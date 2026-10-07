"""원본 비율에 가까운 학습 PNG와 캡션을 별도 폴더에 저장합니다."""

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageOps

from .anima_image import DEFAULT_CROP_ANCHOR, DEFAULT_FREEFIT_MAX_RATIO, select_bucket, resize_to_bucket


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
PREPROCESS_VERSION = 4
SIZE_MULTIPLE = 16


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


def choose_image_size(width, height, target_res, freefit_max_ratio=DEFAULT_FREEFIT_MAX_RATIO):
    """anima-lora가 사용하는 원본 select_bucket 함수로 크기를 선택합니다."""
    return select_bucket(width, height, target_res, max_ratio=freefit_max_ratio)[1]


def resize_image(image, target_res, freefit_max_ratio=DEFAULT_FREEFIT_MAX_RATIO):
    # anime_tools.stages.resize.process_image의 기본 margin=0 경로.
    image = ImageOps.exif_transpose(image).convert("RGB")
    bucket = choose_image_size(*image.size, target_res, freefit_max_ratio)
    return resize_to_bucket(image, bucket, crop_anchor=DEFAULT_CROP_ANCHOR)


def preprocess_images(config, dataset_dir, output_dir):
    """CPU 전처리. 입력별 폴더를 구분하며 manifest를 마지막에 저장합니다."""
    from .config import resolve_preprocess_target_res, validate_config

    validate_config(config)
    root = Path(dataset_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if output_root.is_relative_to(root):
        raise ValueError("전처리 폴더는 원본 데이터셋 폴더 밖에 지정하세요.")
    pairs = discover_pairs(root)
    signatures = []
    for pair in pairs:
        stat = Path(pair["image"]).stat()
        signatures.append({**pair, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    identity = {"version": PREPROCESS_VERSION, "pairs": signatures,
                "target_res": resolve_preprocess_target_res(config),
                "freefit_max_ratio": DEFAULT_FREEFIT_MAX_RATIO,
                "crop_anchor": DEFAULT_CROP_ANCHOR, "size_multiple": SIZE_MULTIPLE,
                "preprocess": "anime_tools_74aa014ba12db74286b17e634056683f7f2b44d7"}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = output_root / fingerprint
    manifest_path = target / "preprocessing.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["identity"] == identity and all(
            (target / record[key]).is_file() for record in manifest["records"] for key in ("file", "caption_file")
        ):
            print(f"기존 전처리 이미지 사용: {len(pairs)}개, {target}")
            return target
    target.mkdir(parents=True, exist_ok=True)
    records = []
    for index, pair in enumerate(pairs):
        filename = f"{index:07d}.png"
        caption_file = f"{index:07d}.txt"
        with Image.open(pair["image"]) as image:
            source_size = ImageOps.exif_transpose(image).size
            processed = resize_image(image, identity["target_res"], identity["freefit_max_ratio"])
        processed.save(target / filename)
        (target / caption_file).write_text(pair["caption"], encoding="utf-8")
        records.append({"file": filename, "caption_file": caption_file, "source": pair["image"],
                        "source_size": source_size, "size": processed.size})
    manifest_path.write_text(json.dumps({"fingerprint": fingerprint, "identity": identity,
                                         "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"전처리 이미지 {len(records)}개 저장: {target}")
    return target
