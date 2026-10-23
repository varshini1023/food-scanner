import os

train_dir = r"C:\Users\MOHAN\OneDrive\Desktop\south indian food dataset\food_scanner_data\train"
test_dir = r"C:\Users\MOHAN\OneDrive\Desktop\south indian food dataset\food_scanner_data\test"

image_extensions = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

def count_images(folder):
    print(f"\nChecking: {folder}")

    if not os.path.isdir(folder):
        print("ERROR: Folder not found!")
        return

    total = 0

    for class_name in sorted(os.listdir(folder)):
        class_path = os.path.join(folder, class_name)

        if not os.path.isdir(class_path):
            continue

        count = sum(
            1
            for filename in os.listdir(class_path)
            if filename.lower().endswith(image_extensions)
        )

        total += count
        print(f"{class_name}: {count} images")

    print(f"TOTAL: {total} images")


count_images(train_dir)
count_images(test_dir)