"""이미지·동명 TXT 캡션을 latent/Qwen 조건 캐시로 준비합니다."""

import gc
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
CACHE_VERSION = 1


def discover_pairs(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve()
    pairs = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            raise ValueError(f"동명 TXT 캡션이 없습니다: {image_path}")
        pairs.append({"image": str(image_path), "caption": caption_path.read_text(encoding="utf-8-sig").strip()})
    if not pairs:
        raise ValueError(f"학습 이미지가 없습니다: {root}")
    return pairs


def read_image(path, resolution):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(image, (resolution, resolution), method=Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1) / 127.5 - 1


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_cache(config, dataset_dir, cache_dir, weights, device="cuda"):
    from ..runtime import load_vae, load_text_encoder, load_tokenizers, encode_prompt
    from .config import resolve_dtype

    pairs = discover_pairs(dataset_dir)
    dtype = resolve_dtype(config, device)
    signatures = []
    for pair in pairs:
        stat = Path(pair["image"]).stat()
        signatures.append({**pair, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    model_signatures = {}
    for name, path in weights.items():
        path = Path(path).resolve()
        stat = path.stat()
        model_signatures[name] = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    identity = {
        "version": CACHE_VERSION, "pairs": signatures, "weights": model_signatures,
        "resolution": config["dataset"]["resolution"], "text_dtype": str(dtype),
        "preprocess": "exif_rgb_center_crop", "max_sequence_length": 512,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_path = Path(cache_dir).expanduser().resolve() / fingerprint
    manifest_path = cache_path / "manifest.json"
    if manifest_path.is_file():
        dataset = CachedDataset(cache_path)
        print(f"기존 캐시 사용: {len(dataset)}개, {cache_path}")
        return cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    records = [{"file": f"{index:07d}.safetensors", "image": pair["image"]}
               for index, pair in enumerate(pairs)]
    print(f"이미지 {len(pairs)}개: 중앙 정사각형 crop → {identity['resolution']} 해상도")

    # VAE는 FP32로 계산하여 FP16 전처리의 overflow를 피합니다.
    vae = load_vae(weights["vae"], device=device, dtype=torch.float32)
    try:
        with torch.no_grad():
            for pair, record in tqdm(list(zip(pairs, records)), desc="VAE latent 캐시"):
                image = read_image(pair["image"], identity["resolution"])
                latent = vae.encode(image.unsqueeze(0), device=device)[0].cpu().contiguous()
                if not torch.isfinite(latent).all():
                    raise RuntimeError(f"VAE latent에 비유한 값이 있습니다: {pair['image']}")
                save_file({"latent": latent}, str(cache_path / record["file"]))
    finally:
        del vae
        release_cuda_memory()

    qwen, t5 = load_tokenizers()
    text_encoder = load_text_encoder(weights["text_encoder"], device=device, dtype=dtype)
    try:
        # Qwen 결과만 캐시합니다. DiT 내부의 llm_adapter는 LoRA 학습 대상일 수 있습니다.
        for pair, record in tqdm(list(zip(pairs, records)), desc="텍스트 조건 캐시"):
            embeds, ids = encode_prompt(text_encoder, qwen, t5, pair["caption"], device=device, dtype=dtype)
            # safetensors의 파일 매핑을 해제한 뒤 같은 파일에 저장합니다.
            # Windows에서는 매핑된 파일에 덮어쓰면 os error 1224가 발생합니다.
            tensors = {name: tensor.clone() for name, tensor in
                       load_file(str(cache_path / record["file"])).items()}
            tensors.update(prompt_embeds=embeds[0].cpu().contiguous(), t5_ids=ids[0].cpu().contiguous())
            save_file(tensors, str(cache_path / record["file"]))
        embeds, ids = encode_prompt(text_encoder, qwen, t5, "", device=device, dtype=dtype)
        save_file({"prompt_embeds": embeds[0].cpu().contiguous(), "t5_ids": ids[0].cpu().contiguous()},
                  str(cache_path / "empty.safetensors"))
    finally:
        del text_encoder, qwen, t5
        release_cuda_memory()
    manifest = {"fingerprint": fingerprint, "identity": identity, "records": records}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("캐시 준비 완료:", cache_path)
    return cache_path


class CachedDataset(Dataset):
    def __init__(self, cache_path):
        self.cache_path = Path(cache_path).resolve()
        self.manifest = json.loads((self.cache_path / "manifest.json").read_text(encoding="utf-8"))
        self.records = self.manifest["records"]
        if not self.records:
            raise ValueError("캐시에 학습 이미지가 없습니다.")
        for record in self.records:
            path = (self.cache_path / record["file"]).resolve()
            if not path.is_relative_to(self.cache_path) or not path.is_file():
                raise ValueError(f"캐시 파일이 없거나 경로가 잘못되었습니다: {record['file']}")
        self.empty = load_file(str(self.cache_path / "empty.safetensors"))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_file(str(self.cache_path / self.records[index]["file"]))


class SampleStream:
    """epoch별 shuffle과 cursor를 저장하여 재개 시 같은 다음 샘플을 선택합니다."""

    def __init__(self, dataset, repeat, seed, epoch=0, cursor=0):
        self.dataset = dataset
        self.repeat = repeat
        self.seed = seed
        self.epoch = epoch
        self.cursor = cursor
        self._set_order()

    def _set_order(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        self.order = torch.randperm(len(self.dataset) * self.repeat, generator=generator).tolist()

    def take(self, batch_size):
        batch = []
        for _ in range(batch_size):
            if self.cursor == len(self.order):
                self.epoch += 1
                self.cursor = 0
                self._set_order()
            index = self.order[self.cursor] % len(self.dataset)
            batch.append(self.dataset[index])
            self.cursor += 1
        return batch

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}
