import os
import shutil
import random

RAW_DIR = "data/raw/extract"
OUT_DIR = "data/processed"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

IMG_EXTS = (".png", ".jpg", ".jpeg")

# Create output folders
for split in SPLITS:
    for cls in ["genuine", "forged"]:
        os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

def get_images(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXTS)
    ]

total = 0

for person in os.listdir(RAW_DIR):
    person_path = os.path.join(RAW_DIR, person)
    if not os.path.isdir(person_path):
        continue

    label = "forged" if person.endswith("_forg") else "genuine"
    images = get_images(person_path)

    if len(images) == 0:
        continue

    random.shuffle(images)

    n = len(images)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    split_map = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in split_map.items():
        for src in files:
            dst = os.path.join(
                OUT_DIR,
                split,
                label,
                os.path.basename(src)
            )
            shutil.copy(src, dst)
            total += 1

print(f"✅ Dataset split completed | Total images copied: {total}")
