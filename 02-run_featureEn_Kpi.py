"""
Main pipeline:
  1. Load all parquet parts from Dataset/Cleaned_output/<dataset>_parts/
  2. Feature engineering
  3. Build feature table
  4. Compute KPIs
  5. Save everything under Dataset/Feature_KPI_output/
"""

import os
import pandas as pd

from config.feature_map import FEATURE_MAP, OUTPUT_DIR
from feature_engineer import FeatureEngineer
from kpi import KPIEngine


def build_feature_table(tables):
    if "user_behavior_enriched" not in tables:
        print("[WARN] user_behavior_enriched not found – cannot build feature_table")
        return None

    df = tables["user_behavior_enriched"].copy()

    # user features
    if "user_features" in tables:
        user_df = tables["user_features"].rename(
            columns={
                col: f"user_{col}"
                for col in tables["user_features"].columns
                if col != "user_id"
            }
        )
        df = df.merge(user_df, on="user_id", how="left")

    # product behavior features
    if "product_behavior" in tables:
        product_df = tables["product_behavior"].rename(
            columns={
                col: f"product_{col}"
                for col in tables["product_behavior"].columns
                if col != "product_id"
            }
        )
        df = df.merge(product_df, on="product_id", how="left")

    # product master
    if "product_master" in tables and "id" in tables["product_master"].columns:
        master = tables["product_master"].copy()
        drop_cols = [c for c in master.columns if c in df.columns and c != "id"]
        master = master.drop(columns=drop_cols, errors="ignore")
        df = df.merge(
            master,
            left_on="product_id",
            right_on="id",
            how="left",
            suffixes=("", "_master"),
        )

    return df


def main():
    print("=" * 60)
    print("START Feature Engineering + KPI Pipeline (Parquet)")
    print("=" * 60)

    # 1. Feature Engineering
    fe = FeatureEngineer(FEATURE_MAP)
    tables = fe.run(save=True)

    # 2. Build feature table
    feature_table = build_feature_table(tables)

    if feature_table is not None:
        print("\n=== FEATURE TABLE ===")
        print(feature_table.head())
        print("shape:", feature_table.shape)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        feature_path = os.path.join(OUTPUT_DIR, "feature_table.parquet")
        feature_table.to_parquet(feature_path, index=False)
        print(f"Saved feature_table → {feature_path}")

    # 3. KPIs
    kpi_engine = KPIEngine(tables)
    kpis = kpi_engine.run()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)

    for kpi_name, kpi_value in kpis.items():
        print(f"\n=== {kpi_name.upper()} ===")
        if isinstance(kpi_value, dict):
            for k, v in kpi_value.items():
                if isinstance(v, (pd.DataFrame, pd.Series)):
                    print(f"  {k}: shape/len = {getattr(v, 'shape', len(v))}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(kpi_value)


if __name__ == "__main__":
    main()