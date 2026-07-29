



from feature_engineer import FeatureEngineer
from map import FEATURE_MAP
from kpi import KPIEngine


import pandas as pd

def build_feature_table(tables):
    df = tables["user_behavior_enriched"]

    # ✅ user features
    if "user_features" in tables:
        user_df = tables["user_features"].rename(columns={
            col: f"user_{col}" for col in tables["user_features"].columns if col != "user_id"
        })

        df = df.merge(user_df, on="user_id", how="left")

    # ✅ product features
    if "product_features" in tables:
        product_df = tables["product_features"].rename(columns={
            col: f"product_{col}" for col in tables["product_features"].columns if col != "id"
        })

        df = df.merge(product_df, on="id", how="left")

    return df


def main():
    print("START")

    
    fe = FeatureEngineer(FEATURE_MAP)
    tables = fe.run()


    feature_table = build_feature_table(tables)

    print("\n=== FEATURE TABLE ===")
    print(feature_table.head())
    print(feature_table.shape)

    kpi_engine = KPIEngine(tables)
    kpis = kpi_engine.run()

    print("DONE")

    for kpi_name, kpi_value in kpis.items():
        print(f"\n=== {kpi_name.upper()} ===")
        print(kpi_value)

if __name__ == "__main__":
    main()




