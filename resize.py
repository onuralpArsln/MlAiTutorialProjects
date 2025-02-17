from PIL import Image
import os

folder = "archive"
output_folder = "resized_archive"

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(folder):
    filepath = os.path.join(folder, filename)

    # Delete non-JPG files
    if not filename.lower().endswith('.jpg'):
        print(f"Deleting non-JPG file: {filename}")
        os.remove(filepath)
        continue

    # Process JPG images
    if os.path.isfile(filepath):
        try:
            with Image.open(filepath) as img:
                img = img.convert("RGB")  # Ensure compatibility
                img.thumbnail((256, 256))  # Maintain aspect ratio
                
                output_path = os.path.join(output_folder, filename)
                img.save(output_path, "JPEG", optimize=True, quality=85)
                
                print(f"Resized: {filename}")
        except Exception as e:
            print(f"Skipping and deleting {filename}: {e}")
            os.remove(filepath)  # Delete problematic file

print("Done!")
