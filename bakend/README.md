# Rahin Customer Excel Export Backend

این Backend فقط برای دانلود شناسه مشتریان هر cluster از جدول `kpi.ml_user_clusters` است.

## نصب
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## تنظیم دیتابیس
مقادیر PostgreSQL را در Environment Variables قرار دهید:
```powershell
$env:DB_HOST="localhost"
$env:DB_NAME="YOUR_DATABASE_NAME"
$env:DB_USER="postgres"
$env:DB_PASSWORD="YOUR_DATABASE_PASSWORD"
$env:DB_PORT="5432"
```

## اجرا
```powershell
uvicorn main:app --reload
```

## Endpoint ها
```text
/api/customer-segments/vip_champions/export
/api/customer-segments/night_weekend_buyers/export
/api/customer-segments/active_loyals/export
/api/customer-segments/low_intent_shoppers/export
/api/customer-segments/churned_customers/export
```

## SQL
برای هر درخواست فقط این Query اجرا می‌شود:
```sql
SELECT user_id
FROM kpi.ml_user_clusters
WHERE cluster_name = %s
ORDER BY user_id;
```

هیچ JOIN ای انجام نمی‌شود.

## اتصال React
```javascript
const API_BASE =
  import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

async function downloadSegment(clusterName) {
  const response = await fetch(
    `${API_BASE}/api/customer-segments/${encodeURIComponent(clusterName)}/export`
  );

  if (!response.ok) throw new Error("خطا در دریافت فایل");

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `${clusterName}.xlsx`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}
```

LangGraph در این endpoint نقشی ندارد.
