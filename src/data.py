from datasets import load_dataset


def load_data():
    dataset = load_dataset("nvidia/Nemotron-3.5-Content-Safety-Dataset")
    dataset = dataset["test"]
    dataset = dataset.filter(lambda row: row["image_path"] is None)
    dataset = dataset.select(range(50))
    dataset = dataset.select_columns(["prompt","input_label","row_id"])
    return dataset
