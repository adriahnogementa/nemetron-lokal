import json
from collections import Counter
from pathlib import Path

DIRECTORY = Path("data_de")
COLUMN = "violated_categories"


counter = Counter()

for files in sorted(DIRECTORY.glob("*.jsonl")):
    with files.open(encoding="utf-8") as rows:
        for row_number, row in enumerate(rows, start=1):
            row = row.strip()
            if not row:
                continue

            try:
                value = json.loads(row)
            except json.JSONDecodeError:
                print(f"Kaputte Zeile übersprungen: {files.name}, Zeile {row_number}")
                continue

            categories = value.get(COLUMN)

            if categories is None:
                continue

            if isinstance(categories, list):
                seperated_categories = categories
            else:
                seperated_categories = str(categories).split(",")

            for category in seperated_categories:
                category = str(category).strip()
                if category:
                    counter[category] += 1

for category, quantity in counter.most_common():
     print(f"{category}\t{quantity}")

print(f"\n{len(counter)} verschiedene Werte, {sum(counter.values())} Vorkommen insgesamt")
