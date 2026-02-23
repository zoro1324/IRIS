import os
import zipfile
import shutil
import random
import glob

# Set random seed for reproducibility
random.seed(42)

# Set base directories
script_dir = os.path.dirname(os.path.abspath(__file__))
zips_dir = os.path.join(script_dir, "zips")
output_root = os.path.join(script_dir, "unzipped_data")

# Clean existing unzipped_data to start fresh, or just create it
if os.path.exists(output_root):
    shutil.rmtree(output_root)
os.makedirs(output_root, exist_ok=True)

# Find all zip files
zip_files = [f for f in os.listdir(zips_dir) if f.endswith(".zip")]

for zip_file in zip_files:
    zip_name = os.path.splitext(zip_file)[0]
    zip_path = os.path.join(zips_dir, zip_file)
    print(f"Processing {zip_name}...")
    
    # Path for extracting this specific zip
    extract_path = os.path.join(output_root, zip_name)
    temp_extract_dir = os.path.join(output_root, f"{zip_name}_temp")
    
    # Extract to a temporary directory first to easily find all files
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_extract_dir)
        
    # Find all images
    images = []
    for ext in ('*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG'):
        images.extend(glob.glob(os.path.join(temp_extract_dir, '**', ext), recursive=True))
        
    # Find all labels
    txt_files = glob.glob(os.path.join(temp_extract_dir, '**', '*.txt'), recursive=True)
    label_dict = {}
    for t in txt_files:
        name = os.path.splitext(os.path.basename(t))[0]
        # Skip roboflow metadata txt files
        if name.lower() not in ['readme.roboflow', 'readme', 'classes', '_darknet.labels']:
            label_dict[name] = t
            
    # Pair images and labels
    pairs = []
    for img_path in images:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        txt_path = label_dict.get(base_name)
            
        if txt_path and os.path.exists(txt_path):
            pairs.append((img_path, txt_path))
            
    # Shuffle for random selection
    random.shuffle(pairs)
    
    # 350 for train, up to 150 for val
    num_train = 350
    num_val = 150
    
    train_pairs = pairs[:num_train]
    val_pairs = pairs[num_train:num_train + num_val]
    
    print(f"  Found {len(pairs)} valid image/label pairs.")
    print(f"  Allocating {len(train_pairs)} to train, {len(val_pairs)} to val.")
    
    # Define paths for final placement
    train_img_dir = os.path.join(extract_path, "train", "images")
    train_lbl_dir = os.path.join(extract_path, "train", "labels")
    val_img_dir = os.path.join(extract_path, "val", "images")
    val_lbl_dir = os.path.join(extract_path, "val", "labels")
    
    # Create directories
    os.makedirs(train_img_dir, exist_ok=True)
    os.makedirs(train_lbl_dir, exist_ok=True)
    os.makedirs(val_img_dir, exist_ok=True)
    os.makedirs(val_lbl_dir, exist_ok=True)
    
    # Function to copy files into their new structure
    def place_pairs(selected_pairs, dest_img_dir, dest_lbl_dir):
        for img_p, txt_p in selected_pairs:
            img_name = os.path.basename(img_p)
            txt_name = os.path.basename(txt_p)
            shutil.copy(img_p, os.path.join(dest_img_dir, img_name))
            shutil.copy(txt_p, os.path.join(dest_lbl_dir, txt_name))
            
    # Place files
    place_pairs(train_pairs, train_img_dir, train_lbl_dir)
    place_pairs(val_pairs, val_img_dir, val_lbl_dir)
    
    # Remove the temporary extracted directory explicitly to discard other files
    shutil.rmtree(temp_extract_dir)

print("All zips processed successfully.")