import sys
from pathlib import Path
import time
from multiprocessing import Pool, cpu_count
import pandas as pd

project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

from src.structured_cleaner import EcomDataCleaner
from config.feature_map import get_dataset
from src.text_preprocessing import TextPreprocessing


def process_chunk(args):
    chunk, columns, part_path = args
    text_pre = TextPreprocessing()

    chunk["raw_text"] = text_pre.create_raw_text(chunk, columns)
    chunk["raw_text_normalized"] = text_pre.Normalize_text(chunk["raw_text"])
    chunk["tokens"] = text_pre.Tokenize(chunk["raw_text_normalized"])

    chunk["tokens"] = chunk["tokens"].apply(lambda t: " ".join(t) if isinstance(t, list) else t)
    chunk.to_parquet(part_path, index=False, engine="pyarrow")

    return part_path, len(chunk)


def test_with_real_data():

    cleaner = EcomDataCleaner()

    input_dir = Path("Dataset/Raw_input")
    output_dir = Path("Dataset/Cleaned_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting complete data pipeline...")
    print("=" * 50)

    if not input_dir.exists() or not list(input_dir.glob("*.*")):
        print("No input files were found in Dataset.")
        return

    files = list(input_dir.glob("*.*"))
    total_files = len(files)

    for file_idx, file_path in enumerate(files, start=1):

        dataset_name = file_path.stem
        config = get_dataset(dataset_name)

        if config is None:
            print(f"Skipping '{dataset_name}': dataset is not defined in feature_map.")
            continue

        print(f"\n[File {file_idx}/{total_files}] Processing: {file_path.name}")

        if file_path.suffix == ".csv":
            df = pd.read_csv(file_path, low_memory=False)
        elif file_path.suffix in [".xlsx", ".xls"]:
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file format: {file_path.suffix}")
            continue

        initial_rows = len(df)
        initial_nulls = df.isnull().sum().sum()

        cleaned_df = cleaner.clean_structured_data(df, dataset_name)

        final_rows = len(cleaned_df)
        final_nulls = cleaned_df.isnull().sum().sum()

        print("\nCleaning Summary")
        print(f"Rows: {initial_rows:,} -> {final_rows:,}")
        print(f"Removed rows: {initial_rows - final_rows:,}")
        print(f"Null values: {initial_nulls:,} -> {final_nulls:,}")

        # -----------------------------
        # Text preprocessing
        # -----------------------------
        chunk_size = 50_000
        parts_dir = output_dir / f"{dataset_name}_parts"
        parts_dir.mkdir(exist_ok=True)

        chunk_list = [
            (
                cleaned_df.iloc[i:i + chunk_size].copy(),
                config["preprocess_columns"],
                parts_dir / f"part_{i // chunk_size:04d}.parquet"
            )
            for i in range(0, len(cleaned_df), chunk_size)
        ]
        total_chunks = len(chunk_list)

        del cleaned_df, df

        num_workers = max(1, cpu_count() // 2)
        print(f"\ncores available: {cpu_count()} | using: {num_workers}")
        print(f"total chunks: {total_chunks} (chunk size: {chunk_size:,})")

        start_time = time.time()
        saved_parts = []

        with Pool(processes=num_workers) as pool:
            for i, (part_path, row_count) in enumerate(
                pool.imap_unordered(process_chunk, chunk_list), start=1
            ):
                saved_parts.append(part_path)
                elapsed = time.time() - start_time
                percent = i / total_chunks * 100
                remaining_sec = (elapsed / i) * (total_chunks - i)

                print(f"  Progress: {percent:5.1f}%  |  "
                      f"Chunk {i}/{total_chunks} ({row_count:,} rows)  |  "
                      f"Elapsed: {elapsed/60:.1f} min  |  "
                      f"ETA: {remaining_sec/60:.1f} min")

        print(f"Text preprocessing done in {(time.time()-start_time)/60:.1f} min")
        print(f"Saved {len(saved_parts)} part files in: {parts_dir}")

    print("\n" + "=" * 50)
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    test_with_real_data()