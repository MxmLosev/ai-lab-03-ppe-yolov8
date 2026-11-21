import shutil
import sys
from pathlib import Path

from common import CLASS_NAMES, group_key, list_images, md5, repo_root, save_json

SRC = repo_root() / ".cache" / "kaggle_dl" / "css-data"
POOL = repo_root() / "data" / "_pool"


def collect():
    images = []
    for split in ("train", "valid", "test"):
        img_dir = SRC / split / "images"
        for img_path in list_images(img_dir):
            label_path = img_dir.parent / "labels" / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"no label for {img_path.name}, skipped")
                continue
            images.append({"file": img_path.name, "origin_split": split, "group": group_key(img_path.stem)})
    return images


def main():
    if not SRC.exists():
        sys.exit(f"source not found: {SRC}")
    if POOL.exists():
        shutil.rmtree(POOL)
    (POOL / "images").mkdir(parents=True)
    (POOL / "labels").mkdir(parents=True)
    records = collect()
    for rec in records:
        shutil.copy2(SRC / rec["origin_split"] / "images" / rec["file"], POOL / "images" / rec["file"])
        shutil.copy2(SRC / rec["origin_split"] / "labels" / (Path(rec["file"]).stem + ".txt"), POOL / "labels" / (Path(rec["file"]).stem + ".txt"))
        rec["md5"] = md5(POOL / "images" / rec["file"])
    save_json(
        {
            "source": {
                "kaggle": "snehilsanyal/construction-site-safety-image-dataset-roboflow",
                "roboflow": "https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/28",
                "github": "https://github.com/snehilsanyal/Construction-Site-Safety-PPE-Detection",
                "license": "CC BY 4.0",
                "roboflow_version": 28,
            },
            "classes": CLASS_NAMES,
            "count": len(records),
            "records": records,
        },
        POOL / "pool_manifest.json",
    )
    groups = {r["group"] for r in records}
    print(f"pool: {len(records)} images, {len(groups)} groups")


if __name__ == "__main__":
    main()
