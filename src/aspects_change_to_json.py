

# تبدیل به جیسون

from pathlib import Path
import pandas as pd
import ast
import json


# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "Dataset" / "Cleaned_output" / "comment"
OUTPUT_PATH = BASE_DIR / "Dataset" / "Cleaned_output" / "comment_json"



OUTPUT_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# تبدیل مقدار به JSON استاندارد
# ============================================================

def convert_to_json(value):

    # NULL / NaN
    if pd.isna(value):
        return None

    # اگر قبلاً list یا dict واقعی باشد
    if isinstance(value, (list, dict)):
        return json.dumps(
            value,
            ensure_ascii=False
        )

    # تبدیل به string
    value = str(value).strip()

    if not value:
        return None

    try:
        # تبدیل stringهایی مثل:
        # "['عالی']"
        # "{'1': 'قیمت'}"
        #
        # به Python list/dict
        parsed = ast.literal_eval(value)

        # فقط list/dict را JSON می‌کنیم
        if isinstance(parsed, (list, dict)):
            return json.dumps(
                parsed,
                ensure_ascii=False
            )

        # اگر چیز دیگری بود
        return json.dumps(
            parsed,
            ensure_ascii=False
        )

    except (ValueError, SyntaxError):

        # اگر مقدار قابل parse نبود،
        # آن را به عنوان string معمولی نگه می‌داریم
        return json.dumps(
            value,
            ensure_ascii=False
        )


# ============================================================
# پیدا کردن فایل‌های Parquet
# ============================================================

files = sorted(INPUT_PATH.glob("*.parquet"))

print(f"Number of files: {len(files)}")


# ============================================================
# پردازش فایل‌ها
# ============================================================

for file in files:

    print("\n" + "=" * 70)
    print(f"Processing: {file.name}")
    print("=" * 70)

    df = pd.read_parquet(file)

    print(f"Rows: {len(df)}")

    # --------------------------------------------------------
    # تبدیل advantages
    # --------------------------------------------------------

    df["advantages"] = df["advantages"].apply(
        convert_to_json
    )

    # --------------------------------------------------------
    # تبدیل disadvantages
    # --------------------------------------------------------

    df["disadvantages"] = df["disadvantages"].apply(
        convert_to_json
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\nAdvantages sample:")
    print(
        df["advantages"]
        .dropna()
        .head()
        .to_string(index=False)
    )

    print("\nDisadvantages sample:")
    print(
        df["disadvantages"]
        .dropna()
        .head()
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # ذخیره
    # --------------------------------------------------------

    output_file = OUTPUT_PATH / file.name

    df.to_parquet(
        output_file,
        index=False
    )

    print(f"\nSaved to:")
    print(output_file)


print("\nDONE")

