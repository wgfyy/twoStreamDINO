import json
import random
import os

def generate_val_small():
    # 1. Configuration Paths
    input_path = '/root/autodl-tmp/M4-SAR-coco-OBB/annotations/train.json'
    output_path = '/root/autodl-tmp/M4-SAR-coco-OBB/annotations/train_small.json'
    target_samples = 1000
    
    print(f"Reading from: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # 2. Load the original JSON
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    images = data['images']
    annotations = data['annotations']
    categories = data.get('categories', [])
    info = data.get('info', {})
    licenses = data.get('licenses', [])
    
    total_images = len(images)
    print(f"Total images in val set: {total_images}")
    
    # 3. Sample images
    if total_images <= target_samples:
        print(f"Total images ({total_images}) is less than or equal to target ({target_samples}). Keeping all.")
        sampled_images = images
    else:
        print(f"Randomly sampling {target_samples} images...")
        random.seed(42)  # Set seed for reproducibility
        sampled_images = random.sample(images, target_samples)
    
    # 4. Filter annotations based on selected image IDs
    # Create a set for O(1) lookup
    sampled_img_ids = set(img['id'] for img in sampled_images)
    
    sampled_annotations = []
    for ann in annotations:
        if ann['image_id'] in sampled_img_ids:
            sampled_annotations.append(ann)
            
    print(f"Sampled Dataset Statistics:")
    print(f"  Images: {len(sampled_images)}")
    print(f"  Annotations: {len(sampled_annotations)}")
    
    # 5. Construct new data dictionary
    new_data = {
        'info': info,
        'licenses': licenses,
        'images': sampled_images,
        'annotations': sampled_annotations,
        'categories': categories
    }
    
    # 6. Save to new JSON file
    print(f"Saving to: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(new_data, f)
        
    print("Done successfully.")

if __name__ == "__main__":
    generate_val_small()