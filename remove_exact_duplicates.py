import os
import hashlib
import csv

TRAIN_DIR = (
    r"C:\Users\MOHAN\OneDrive\Desktop"
    r"\south indian food dataset\food_scanner_data\train"
)

REPORT_FILE = "duplicate_removal_log.csv"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def file_hash(path):
    hasher = hashlib.sha256()

    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def main():
    seen = {}
    duplicates = []

    for class_name in sorted(os.listdir(TRAIN_DIR)):
        class_dir = os.path.join(TRAIN_DIR, class_name)

        if not os.path.isdir(class_dir):
            continue

        for filename in sorted(os.listdir(class_dir)):
            if not filename.lower().endswith(VALID_EXTENSIONS):
                continue

            path = os.path.join(class_dir, filename)

            try:
                digest = file_hash(path)
            except OSError as error:
                print(f"Could not read {path}: {error}")
                continue

            key = (class_name, digest)

            if key in seen:
                duplicates.append((path, seen[key]))
            else:
                seen[key] = path

    print(f"Exact duplicate copies found: {len(duplicates)}")

    for duplicate, original in duplicates:
        print(f"\nDuplicate: {duplicate}")
        print(f"Original:  {original}")

    # Save a log before making any changes.
    with open(
        REPORT_FILE, "w", newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Duplicate removed", "Original kept"])

        for duplicate, original in duplicates:
            writer.writerow([duplicate, original])

    # Delete only exact duplicates within the same class.
    confirm = input(
        "\nType DELETE to remove these duplicate copies: "
    )

    if confirm != "DELETE":
        print("Cancelled. No images were deleted.")
        return

    removed = 0

    for duplicate, original in duplicates:
        if os.path.exists(duplicate) and os.path.exists(original):
            os.remove(duplicate)
            removed += 1

    print(f"\nRemoved {removed} exact duplicate copies.")
    print(f"Removal log saved to: {REPORT_FILE}")


if __name__ == "__main__":
    main()
