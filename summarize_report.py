import csv
from collections import Counter

report_file = "suspicious_images_report.csv"

issues = Counter()
confusions = Counter()

with open(report_file, "r", encoding="utf-8-sig") as file:
    reader = csv.DictReader(file)

    for row in reader:
        issue = row["Issue"]
        actual = row["Actual folder"]
        predicted = row["Model prediction"]

        issues[issue] += 1

        if issue == "PREDICTION_DIFFERS_FROM_FOLDER":
            confusions[(actual, predicted)] += 1

print("\n========== REPORT SUMMARY ==========")

print("\nIssue counts:")
for issue, count in issues.most_common():
    print(f"{issue}: {count}")

print("\nMost common classification mistakes:")
for (actual, predicted), count in confusions.most_common():
    print(f"{actual} predicted as {predicted}: {count}")

print("\n====================================")