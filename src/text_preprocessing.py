# from pathlib import Path
from matplotlib import text
import pandas as pd
import re
import string

class TextPreprocessing:
    def __init__(self):
        pass

    # @staticmethod
    def Overview(self, df):
        print("Overview : \n" , df.head())
        print("\nIs Null : \n" , df.isna().sum())

    # @staticmethod
    def create_raw_text(self, df, columns):
        raw_text = pd.Series("", index=df.index, dtype="string")

        for col in columns:

            if col not in df.columns:
                continue

            text = (
                df[col]
                .fillna("")
                .astype("string")
                .str.strip()
            )

            raw_text = raw_text.str.cat(text, sep=" ")

        raw_text = (
            raw_text
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        print("\nRaw Text Samples:\n")
        print(raw_text.head())

        return raw_text
    
    # @staticmethod
    def Normalize_text(self, input_text):

        if isinstance(input_text, pd.Series):
            normalized = input_text.astype(str).copy()
        else:
            normalized = pd.Series([str(input_text)])

        translation_table = str.maketrans({
            "ۀ": "ه",
            "ك": "ک",
            "ي": "ی",
            "ئ": "ی",
            "أ": "ا",
            "إ": "ا",
            "آ": "ا",
            "ؤ": "و",
            "\r": "",
            "\\r": "",
            "\n": "",
            "\u200c": " ",
            "\u00ad": " "
        })
        normalized = normalized.apply(lambda x: x.translate(translation_table))

        normalized = normalized.str.replace(r"<[^>]+>", " ", regex=True)  # HTML
        normalized = normalized.str.replace(r"http\S+|www\S+", " ", regex=True)  # URL
        normalized = normalized.str.replace(r"\S+@\S+", " ", regex=True)  # Email

        punctuation = "،؛؟«»" + string.punctuation
        normalized = normalized.str.replace("[" + re.escape(punctuation) + "]", " ", regex=True)

        normalized = normalized.apply(
            lambda x: re.sub(r'(.)\1{2,}', r'\1', x) if isinstance(x, str) else x
        )  # Repeated Characters

        normalized = normalized.str.replace(r"\s+", " ", regex=True)
        normalized = normalized.str.strip()

        print("\nNormalized Samples:\n")
        for i, text in enumerate(normalized.head(3), start=1):
            print(f"{i}. {text[:100]}")

        return normalized


    # # @staticmethod
    # def fast_tokenize(self, text):
    #     return text.split()

    # # @staticmethod
    # def Tokenize(self, input_text):
    #     if isinstance(input_text, pd.Series):
    #         return input_text.apply(self.fast_tokenize)
    #     return self.fast_tokenize(str(input_text))

    ## @staticmethod
    # def Export(self, df, output_path):
    #     output_path = Path(output_path)
    #     output_path.parent.mkdir(parents=True, exist_ok=True)

    #     if output_path.suffix == ".csv":
    #         df.to_csv(output_path, index=False, encoding="utf-8")

    #     elif output_path.suffix in [".xlsx", ".xls"]:
    #         df.to_excel(output_path, index=False)

    #     elif output_path.suffix == ".parquet":
    #         df.to_parquet(output_path, index=False, engine="pyarrow")

    #     else:
    #         raise ValueError(f"Unsupported file format: {output_path.suffix}")

    #     print(f"\nDataset saved: {output_path}")