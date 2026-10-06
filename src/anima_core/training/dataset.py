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


from .preprocessing import discover_pairs, preprocess_images


CACHE_VERSION = 4


def read_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        pixels = np.asarray(image, dtype=np.float32).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(1) / 127.5 - 1


def release_cuda_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def prepare_cache(config, dataset_dir, cache_dir, weights, device="cuda", preprocessed_dir=None):
    from ..runtime import load_vae, load_text_encoder, load_tokenizers, encode_prompt
    from .config import resolve_dtype

    # 직접 호출해도 노트북과 동일한 PNG 전처리를 거칩니다.
    output_root = Path(preprocessed_dir).resolve().parent if preprocessed_dir else Path(cache_dir) / "preprocessed"
    processed_path = preprocess_images(config, dataset_dir, output_root)
    if preprocessed_dir and processed_path != Path(preprocessed_dir).resolve():
        raise ValueError("전처리 설정 또는 원본이 바뀌었습니다. 전처리 셀을 다시 실행하세요.")
    pairs = discover_pairs(processed_path)
    image_manifest = json.loads((processed_path / "preprocessing.json").read_text(encoding="utf-8"))
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
        "max_pixels": config["dataset"]["max_pixels"], "text_dtype": str(dtype),
        "preprocessing_fingerprint": image_manifest["fingerprint"],
        "preprocess": "exif_rgb_dynamic_center_crop_bilinear", "max_sequence_length": 512,
    }
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    cache_path = Path(cache_dir).expanduser().resolve() / fingerprint
    manifest_path = cache_path / "manifest.json"
    if manifest_path.is_file():
        dataset = CachedDataset(cache_path)
        print(f"기존 캐시 사용: {len(dataset)}개, {cache_path}")
        return cache_path
    cache_path.mkdir(parents=True, exist_ok=True)
    records = [{"file": f"{index:07d}.safetensors", "image": pair["image"],
                "image_size": image_manifest["records"][index]["size"]}
               for index, pair in enumerate(pairs)]
    print(f"전처리 PNG {len(pairs)}개 → VAE latent 캐시")

    # VAE는 FP32로 계산하여 FP16 전처리의 overflow를 피합니다.
    vae = load_vae(weights["vae"], device=device, dtype=torch.float32)
    try:
        with torch.no_grad():
            for pair, record in tqdm(list(zip(pairs, records)), desc="VAE latent 캐시"):
                image = read_image(pair["image"])
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
        for pair, record in tqdm(list(zip(pairs, records)), desc="텍스트 조건 캐시"):
            embeds, ids, qwen_mask, t5_mask = encode_prompt(
                text_encoder, qwen, t5, pair["caption"], device=device, dtype=dtype,
                return_attention_masks=True,
            )
            # safetensors의 파일 매핑을 해제한 뒤 같은 파일에 저장합니다.
            # Windows에서는 매핑된 파일에 덮어쓰면 os error 1224가 발생합니다.
            tensors = {name: tensor.clone() for name, tensor in
                       load_file(str(cache_path / record["file"])).items()}
            tensors.update(prompt_embeds=embeds[0].cpu().contiguous(), t5_ids=ids[0].cpu().contiguous())
            tensors.update(qwen_mask=qwen_mask[0].cpu().contiguous(), t5_mask=t5_mask[0].cpu().contiguous())
            save_file(tensors, str(cache_path / record["file"]))
        embeds, ids, qwen_mask, t5_mask = encode_prompt(
            text_encoder, qwen, t5, "", device=device, dtype=dtype, return_attention_masks=True,
        )
        save_file({"prompt_embeds": embeds[0].cpu().contiguous(), "t5_ids": ids[0].cpu().contiguous(),
                   "qwen_mask": qwen_mask[0].cpu().contiguous(), "t5_mask": t5_mask[0].cpu().contiguous()},
                  str(cache_path / "empty.safetensors"))
    finally:
        del text_encoder, qwen, t5
        release_cuda_memory()
    from .conditioning import load_llm_adapter

    adapter = load_llm_adapter(weights["dit"], device=device, dtype=dtype)
    try:
        with torch.no_grad():
            for record in tqdm(records + [{"file": "empty.safetensors"}], desc="adapter 조건 캐시"):
                path = cache_path / record["file"]
                tensors = {name: tensor.clone() for name, tensor in load_file(str(path)).items()}
                target_mask = tensors["t5_mask"].unsqueeze(0).to(device=device)
                output = adapter(
                    tensors["prompt_embeds"].unsqueeze(0).to(device=device, dtype=dtype),
                    tensors["t5_ids"].unsqueeze(0).to(device=device),
                    target_attention_mask=target_mask,
                    source_attention_mask=tensors["qwen_mask"].unsqueeze(0).to(device=device),
                )
                output = output.masked_fill(~target_mask.bool().unsqueeze(-1), 0)
                cached = {"crossattn_emb": output[0].cpu().contiguous()}
                if "latent" in tensors:
                    cached["latent"] = tensors["latent"]
                save_file(cached, str(path))
                del output, tensors, cached, target_mask
    finally:
        del adapter
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
        self.bucket_keys = [tuple(record["image_size"]) if "image_size" in record else
                            tuple(load_file(str(self.cache_path / record["file"]))["latent"].shape)
                            for record in self.records]
        self.empty = load_file(str(self.cache_path / "empty.safetensors"))
        if "crossattn_emb" not in self.empty:
            raise ValueError("이전 학습 캐시에는 adapter 출력이 없습니다. 캐시 준비 단계를 다시 실행하세요.")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return load_file(str(self.cache_path / self.records[index]["file"]))


class SampleStream:
    """크기별 배치를 섞고 epoch·cursor로 동일한 다음 배치를 복원합니다."""

    def __init__(self, dataset, repeat, seed, batch_size, epoch=0, cursor=0):
        self.dataset = dataset
        self.repeat = repeat
        self.seed = seed
        self.batch_size = batch_size
        self.epoch = epoch
        self.cursor = cursor
        self._set_order()

    def _set_order(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        buckets = {}
        for index, key in enumerate(self.dataset.bucket_keys):
            buckets.setdefault(key, []).extend([index] * self.repeat)
        batches = []
        for indices in buckets.values():
            shuffled = [indices[i] for i in torch.randperm(len(indices), generator=generator).tolist()]
            batches.extend(shuffled[start:start + self.batch_size]
                           for start in range(0, len(shuffled), self.batch_size))
        self.batches = [batches[i] for i in torch.randperm(len(batches), generator=generator).tolist()]
        self.order = [index for batch in self.batches for index in batch]

    def take(self, batch_size):
        if batch_size != self.batch_size:
            raise ValueError("배치 크기는 SampleStream 생성 시 설정한 값과 같아야 합니다.")
        if self.cursor == len(self.batches):
            self.epoch += 1
            self.cursor = 0
            self._set_order()
        indices = self.batches[self.cursor]
        self.cursor += 1
        return [self.dataset[index] for index in indices]

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}
