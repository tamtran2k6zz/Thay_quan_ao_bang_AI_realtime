"""
Script chạy trên Google Colab (GPU) để tự động tinh chỉnh nhãn YOLO Segmentation (.txt)
cho bộ ảnh trong `clothes_dataset/images/{train,val}` bằng SAM2 (Segment Anything 2).

Cách dùng trên Google Colab:
  !pip install -q ultralytics
  !python /content/drive/MyDrive/clothes_dataset/auto_label_sam_colab.py
"""

from pathlib import Path
from PIL import Image
from ultralytics import SAM

CLASS_MAP = {
    "shirt": 0,
    "pants": 1,
    "jacket": 2,
    "dress": 3,
}


def infer_class_id(stem: str) -> int | None:
    prefix = stem.split("_", 1)[0].lower()
    return CLASS_MAP.get(prefix)


def auto_segment_dataset(dataset_root: Path, sam_weights: str = "sam2_b.pt") -> None:
    model = SAM(sam_weights)
    updated = 0

    for split in ("train", "val"):
        img_dir = dataset_root / "images" / split
        lbl_dir = dataset_root / "labels" / split
        lbl_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            continue

        for img_path in sorted(img_dir.glob("*.*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            class_id = infer_class_id(img_path.stem)
            if class_id is None:
                continue

            with Image.open(img_path) as im:
                w, h = im.size

            # Gợi ý vùng trung tâm ảnh sản phẩm cho SAM2 tách chính xác đường viền trang phục
            prompt_bbox = [w * 0.12, h * 0.08, w * 0.88, h * 0.92]
            results = model.predict(str(img_path), bboxes=[prompt_bbox], verbose=False)

            masks = results[0].masks if results else None
            if masks is None or not masks.xyn:
                continue

            # Lấy polygon lớn nhất đại diện cho trang phục chính
            poly = max(masks.xyn, key=lambda pts: len(pts))
            if len(poly) < 3:
                continue

            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in poly)
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            lbl_path.write_text(f"{class_id} {coords}\n", encoding="utf-8")
            updated += 1
            print(f"Labeled {split}/{img_path.stem}.txt (class={class_id}, points={len(poly)})")

    print(f"Done! Refined {updated} YOLO-seg label files.")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    auto_segment_dataset(root)
