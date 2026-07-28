

import pandas as pd


class FeatureEngine:

    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.tables = {}

    # =====================
    # LOAD DATA
    # =====================
    def load_data(self):
        for name, config in self.feature_map.items():
            file = config["file_name"]

            if file.endswith(".csv"):
                df = pd.read_csv(file, encoding='cp1256')
            else:
                df = pd.read_excel(file)

            self.tables[name] = df

    # =====================
    # PREPROCESS
    # =====================
    def preprocess(self):
        for name, config in self.feature_map.items():
            df = self.tables[name]

            # datetime
            for col in config.get("datetime_columns", []):
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")

            self.tables[name] = df

    # =====================
    # FEATURE BUILD
    # =====================
    def build_features(self, df, config):

        df = df.copy()
        group_cols = config.get("join_columns", [])

        agg_dict = {}

        # -------- numeric --------


        for col in config.get("numeric_columns", []):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                agg_dict[col] = ["sum", "mean", "max", "min", "count"]

        # -------- datetime --------
        for col in config.get("datetime_columns", []):
            if col in df.columns:
                df[col + "_hour"] = df[col].dt.hour
                agg_dict[col] = ["min", "max"]

        # -------- categorical --------
        for col in config.get("categorical_columns", []):
            if col in df.columns:
                agg_dict[col] = ["nunique"]

        # -------- text --------
        for col in config.get("text_columns", []):
            if col in df.columns:
                df[col + "_len"] = df[col].fillna("").astype(str).apply(len)
                agg_dict[col + "_len"] = ["mean"]

        # -------- aggregation --------
        if len(group_cols) > 0 and len(agg_dict) > 0:
            features = df.groupby(group_cols).agg(agg_dict)
            features.columns = ["_".join(col) for col in features.columns]
            features = features.reset_index()
        else:
            features = df.copy()

        return features

    # =====================
    # BUILD ALL
    # =====================
    def build_all_features(self):
        result = {}

        for name, config in self.feature_map.items():
            df = self.tables[name]
            feats = self.build_features(df, config)
            result[name] = feats

        return result

    # =====================
    # MERGE
    # =====================
    def merge_all(self, feature_tables):

        base = "orders"  # جدول اصلی
        final_df = feature_tables[base]

        for name, df in feature_tables.items():
            if name == base:
                continue

            join_cols = self.feature_map[name].get("join_columns", [])
            common = [c for c in join_cols if c in final_df.columns]

            if len(common) > 0:
                final_df = final_df.merge(df, on=common, how="left")

        return final_df

    # =====================
    # RUN
    # =====================
    def run(self):
        self.load_data()
        self.preprocess()
        feats = self.build_all_features()
        final = self.merge_all(feats)
        return final

