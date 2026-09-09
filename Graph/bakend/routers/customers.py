import psycopg2
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from database import get_connection
from services.excel_service import create_customer_excel

router = APIRouter(tags=["Customer Export"])

ALLOWED_CLUSTERS = {
    "vip_champions",
    "night_weekend_buyers",
    "active_loyals",
    "low_intent_shoppers",
    "churned_customers",
}

@router.get("/api/customer-segments/{cluster_name}/export")
def export_customers(cluster_name: str):
    if cluster_name not in ALLOWED_CLUSTERS:
        raise HTTPException(status_code=400, detail="Unknown customer cluster.")

    conn = cursor = None
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT user_id
            FROM kpi.ml_user_clusters
            WHERE cluster_name = %s
            ORDER BY user_id
        """, (cluster_name,))

        user_ids = [row[0] for row in cursor.fetchall()]
        excel_file = create_customer_excel(user_ids)

        return StreamingResponse(
            excel_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f'attachment; filename="{cluster_name}.xlsx"'},
        )
    except psycopg2.Error as exc:
        raise HTTPException(status_code=500, detail="Database error while exporting customers.") from exc
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
