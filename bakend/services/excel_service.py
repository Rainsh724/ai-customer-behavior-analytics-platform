from io import BytesIO
import pandas as pd

def create_customer_excel(user_ids):
    df = pd.DataFrame({"user_id": user_ids})
    output = BytesIO()
    df.to_excel(output, index=False, engine="openpyxl")
    output.seek(0)
    return output
