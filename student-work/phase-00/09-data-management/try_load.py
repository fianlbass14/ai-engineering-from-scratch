from pathlib import Path

from datasets import load_dataset

OUTPUT_DIR = Path(__file__).parent

# dataset = load_dataset("stanfordnlp/imdb")
# print(dataset)
# print(dataset["train"][0])

# dataset = load_dataset(
#     "wikimedia/wikipedia", "20231101.ab", split="train", streaming=True
# )

# for i, example in enumerate(dataset):
#     print(example["title"])
#     if i >= 4:
#         break


dataset = load_dataset("stanfordnlp/imdb", split = "train")
print(dataset)
dataset.to_csv(OUTPUT_DIR / "imdb_train.csv")
dataset.to_json(OUTPUT_DIR / "imdb_train.json")
dataset.to_parquet(OUTPUT_DIR / "imdb_train.parquet")



# def conv(data, output_dir:str, name:str):
#     output_path = Path(output_dir)
#     output_path.mkdir(parents=True, exist_ok=True)

#     csv_path = output_path / f"{name}.csv"
#     parquet_path = output_path / f"{name}.parquet"

#     data.to_csv(str(csv_path))
#     data.to_parquet(str(parquet_path))

#     csv_size = Path(str(csv_path)).stat().st_size
#     parquet_size = Path(str(parquet_path)).stat().st_size

#     result = csv_size/parquet_size

#     return result


# exercise3
csv_size = (OUTPUT_DIR / "imdb_train.csv").stat().st_size
parquet_size = (OUTPUT_DIR / "imdb_train.parquet").stat().st_size
result = csv_size/parquet_size
print(result)

# exercise4
# dataset = load_dataset("stanfordnlp/imdb", split = "train")
split = dataset.train_test_split(test_size = 0.15, seed = 123)
train_val = split["train"].train_test_split(test_size =0.15/0.85, seed = 123)

train_ds = train_val["train"]
val_ds = train_val["test"]
test_ds = split["test"]

total = len(train_ds) + len(val_ds) + len(test_ds)
print(f"Train: {len(train_ds)} ({len(train_ds)/total:.1%})")
print(f"Val:   {len(val_ds)} ({len(val_ds)/total:.1%})")
print(f"Test:  {len(test_ds)} ({len(test_ds)/total:.1%})")
print(f"Total matches original: {total == len(dataset)}")


