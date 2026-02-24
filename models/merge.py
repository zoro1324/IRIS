import os
import shutil
import yaml


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

source_root = "unzipped_data"
target_root = "dataset"

class_names = [
    "Bicycle",
    "Bus",
    "Car",
    "Chair",
    "Cow",
    "Dogs",
    "Motorcycle",
    "Person",
    "Stair",
    "Table",
    "Trash",
    "Truck",
]

classes = {
    name: {"id": idx, "slug": slugify(name)}
    for idx, name in enumerate(class_names)
}

split_map = {
    "train": "train",
    "val": "val",
}

# Create directory structure for YOLO dataset
for split in split_map.values():
    os.makedirs(os.path.join(target_root, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(target_root, split, "labels"), exist_ok=True)

for animal, meta in classes.items():
    class_id = meta["id"]
    slug = meta["slug"]
    print(f"\nProcessing {animal}...")

    for src_split, dst_split in split_map.items():
        img_dir = os.path.join(source_root, animal, src_split, "images")
        lbl_dir = os.path.join(source_root, animal, src_split, "labels")

        if not os.path.isdir(img_dir):
            print(f"  Skipping {src_split} (directory not found)")
            continue

        files = [
            f
            for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"))
        ]
        print(f"  {src_split}: {len(files)} images found")

        for file in files:
            name, ext = os.path.splitext(file)
            new_name = f"{slug}_{name}"

            src_img = os.path.join(img_dir, file)
            dst_img = os.path.join(target_root, dst_split, "images", f"{new_name}{ext}")

            src_lbl = os.path.join(lbl_dir, f"{name}.txt")
            dst_lbl = os.path.join(target_root, dst_split, "labels", f"{new_name}.txt")

            shutil.copy(src_img, dst_img)
            print(f"    Copied: {new_name}{ext}")

            if os.path.exists(src_lbl):
                with open(src_lbl, "r") as f:
                    lines = f.readlines()

                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        parts[0] = str(class_id)
                        new_lines.append(" ".join(parts))

                with open(dst_lbl, "w") as f:
                    f.write("\n".join(new_lines))
            else:
                print(f"      Warning: No label file for {new_name}{ext}")

print("\nDataset reorganization complete!")

# Create data.yaml for YOLO
data_yaml = {
    "path": os.path.abspath(target_root),
    "train": "train/images",
    "val": "val/images",
    "nc": len(class_names),
    "names": {idx: name for name, meta in classes.items() for idx in [meta["id"]]}
}

with open(os.path.join(target_root, "data.yaml"), "w") as f:
    yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

print(f"\nCreated data.yaml with {len(class_names)} classes")
print(f"Dataset structure:")
print(f"  - {target_root}/train/images/")
print(f"  - {target_root}/val/images/")
print(f"  - {target_root}/train/labels/")
print(f"  - {target_root}/val/labels/")