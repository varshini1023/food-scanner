import os
import hashlib
import csv
from PIL import Image, ImageOps

# Your training dataset location
TRAIN_DIR = (
    r"C:\Users\MOHAN\OneDrive\Desktop"
    r"\south indian food dataset\food_scanner_data\train"
)

# Report will be saved in your current project folder
REPORT_FILE = "dataset_quality_report.csv"

# Food classes expected in your dataset
EXPECTED_CLASSES = [
    "bisibelebath",
    "chappati",
    "dosa",
    "idly",
    "jamun",
    "momos",
    "pakoda",
    "poori",
    "upma"
]

VALID_EXTENSIONS = (
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
)


def get_image_hash(image_path):
    """Calculate an exact file hash."""
    hasher = hashlib.sha256()

    with open(image_path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def get_difference_hash(image_path):
    """Create a simple visual hash for similarity checking."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image)
        image = image.convert("L").resize((9, 8))

        pixels = list(image.getdata())
        result = 0

        for row in range(8):
            for col in range(8):
                left = pixels[row * 9 + col]
                right = pixels[row * 9 + col + 1]

                result = (result << 1) | int(left > right)

        return result


def hash_distance(hash1, hash2):
    """Count the differing bits between two visual hashes."""
    return (hash1 ^ hash2).bit_count()


def main():
    if not os.path.isdir(TRAIN_DIR):
        print("ERROR: Training folder not found!")
        print("Check this path:")
        print(TRAIN_DIR)
        return

    records = []
    image_data = []
    exact_hashes = {}
    visual_hashes = {}

    print("\nChecking training dataset...")
    print("Please wait; larger datasets take longer.\n")

    # Scan each expected class
    for class_name in EXPECTED_CLASSES:
        class_dir = os.path.join(TRAIN_DIR, class_name)

        if not os.path.isdir(class_dir):
            print(f"WARNING: Missing class folder: {class_name}")
            continue

        files = [
            filename
            for filename in os.listdir(class_dir)
            if filename.lower().endswith(VALID_EXTENSIONS)
        ]

        print(f"{class_name}: {len(files)} image files")

        for filename in files:
            image_path = os.path.join(class_dir, filename)
            relative_path = os.path.relpath(image_path, TRAIN_DIR)

            # Check whether the image can be opened
            try:
                with Image.open(image_path) as image:
                    image.verify()

                # Calculate hashes after validating the image
                exact_hash = get_image_hash(image_path)
                visual_hash = get_difference_hash(image_path)

            except Exception as error:
                records.append([
                    "CORRUPTED_OR_UNREADABLE",
                    relative_path,
                    "",
                    str(error)
                ])
                continue

            # Record exact duplicate files
            if exact_hash in exact_hashes:
                original_path = exact_hashes[exact_hash]

                original_class = original_path.split(
                    os.sep
                )[0]

                current_class = class_name

                if original_class != current_class:
                    issue = "EXACT_DUPLICATE_ACROSS_CLASSES"
                else:
                    issue = "EXACT_DUPLICATE_SAME_CLASS"

                records.append([
                    issue,
                    relative_path,
                    original_path,
                    "Files have identical contents"
                ])
            else:
                exact_hashes[exact_hash] = relative_path

            image_data.append({
                "path": relative_path,
                "class": class_name,
                "hash": visual_hash
            })

    # Look for visually similar images in different classes.
    # This flags candidates; it does not prove mislabelling.
    print("\nChecking for visually similar images...")

    for i in range(len(image_data)):
        first = image_data[i]

        for j in range(i + 1, len(image_data)):
            second = image_data[j]

            # Prioritize suspicious cross-class similarities
            if first["class"] == second["class"]:
                continue

            distance = hash_distance(
                first["hash"],
                second["hash"]
            )

            if distance <= 5:
                records.append([
                    "POSSIBLY_SIMILAR_ACROSS_CLASSES",
                    first["path"],
                    second["path"],
                    f"Visual hash difference: {distance}"
                ])

    # Save report
    with open(
        REPORT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)

        writer.writerow([
            "Issue",
            "Image 1",
            "Image 2_or_Details",
            "Notes"
        ])

        writer.writerows(records)

    print("\n--------------------------------")
    print("DATASET CHECK COMPLETE")
    print("--------------------------------")
    print(f"Images scanned: {len(image_data)}")
    print(f"Potential issues found: {len(records)}")
    print(f"Report saved to: {os.path.abspath(REPORT_FILE)}")

    print("\nNo images were deleted or modified.")
    print("Review the report before making changes.")


if __name__ == "__main__":
    main()