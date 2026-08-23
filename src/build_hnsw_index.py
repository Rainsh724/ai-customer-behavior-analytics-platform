"""
اسکریپت ساخت ایندکس HNSW برای جدول comments_embedding (~6.5M ردیف)
با psycopg2 - شامل:
  - اتصال پایدار با keepalive (جلوگیری از قطعی کانکشن روی کوئری‌های طولانی)
  - اجرای CREATE INDEX CONCURRENTLY (بدون قفل کردن جدول، قابل retry امن)
  - مانیتورینگ زنده پیشرفت از pg_stat_progress_create_index در یک ترد جدا
  - تشخیص و پاکسازی خودکار ایندکس INVALID در صورت شکست قبلی

نصب پیش‌نیاز:
    pip install psycopg2-binary
"""

import sys
import time
import threading
import psycopg2

# ---------------------- تنظیمات ----------------------
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "dbname": "postgres",
    "user": "postgres",
    "password": "HiddenPatern",
    # keepalive برای جلوگیری از قطع شدن کانکشن روی عملیات چند ساعته
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

TABLE_NAME = "comments_embedding"
COLUMN_NAME = "embedded_comment"
INDEX_NAME = "idx_comments_embedding_hnsw"

# --- تنظیم‌شده برای سیستم: Core i7 نسل 9 / 6 هسته فیزیکی (12 ترد) / 16GB RAM
#     با Postgres روی همین سیستم (نه سرور جدا).
#     dimension=768 => حجم خام دیتا ~20GB که بزرگ‌تر از کل RAM سیستمه، پس
#     maintenance_work_mem رو محافظه‌کارانه تنظیم می‌کنیم تا OS/Postgres
#     دچار کمبود حافظه و swap نشن (که خودش باعث کندی بیشتر می‌شه).
#     قبل از اجرا حتماً برنامه‌های سنگین دیگه (مرورگر و...) رو ببندید.
MAINTENANCE_WORK_MEM = "6GB"  # اگر مانیتور RAM نشون داد جا دارید، تا 8GB امتحان کنید
MAX_PARALLEL_MAINTENANCE_WORKERS = 3  # با 6 هسته فیزیکی، بیشتر از این فایده چندانی نداره
MAX_PARALLEL_WORKERS = 6  # باید >= MAX_PARALLEL_MAINTENANCE_WORKERS باشه

MIN_FREE_RAM_GB_WARNING = 3.0  # اگر کمتر از این مونده بود، هشدار می‌ده

HNSW_M = 16
HNSW_EF_CONSTRUCTION = 64

MONITOR_INTERVAL_SEC = 15
# -------------------------------------------------------


def check_free_ram():
    """چک می‌کنه قبل از شروع، RAM آزاد کافی برای maintenance_work_mem + سیستم هست یا نه.
    نیاز به psutil داره: pip install psutil
    اگه نصب نبود، فقط یک هشدار می‌ده و ادامه پیدا می‌کنه."""
    try:
        import psutil

        free_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"[*] RAM آزاد فعلی: {free_gb:.1f} GB")
        if free_gb < MIN_FREE_RAM_GB_WARNING:
            print(
                f"[!] هشدار: RAM آزاد کمتر از {MIN_FREE_RAM_GB_WARNING}GB است. "
                "پیشنهاد می‌شه برنامه‌های دیگه رو ببندید یا "
                "MAINTENANCE_WORK_MEM رو کاهش بدید تا سیستم swap نکنه."
            )
            answer = input("ادامه بدم؟ (y/n): ").strip().lower()
            if answer != "y":
                print("لغو شد.")
                sys.exit(0)
    except ImportError:
        print(
            "[i] psutil نصب نیست، چک RAM آزاد رد شد "
            "(برای فعال کردنش: pip install psutil)."
        )


def get_connection():
    conn = psycopg2.connect(**DB_CONFIG)
    conn.autocommit = True  # CREATE INDEX CONCURRENTLY نباید داخل تراکنش اجرا بشه
    return conn


def cleanup_invalid_index(conn):
    """اگه از تلاش قبلی، ایندکس INVALID باقی مونده، پاکش می‌کنیم تا دوباره بسازیم."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT indexrelid::regclass
            FROM pg_index
            WHERE indisvalid = false
              AND indexrelid::regclass::text = %s
            """,
            (INDEX_NAME,),
        )
        row = cur.fetchone()
        if row:
            print(f"[!] ایندکس ناقص قبلی پیدا شد ({row[0]}) - در حال حذف...")
            cur.execute(f"DROP INDEX CONCURRENTLY IF EXISTS {INDEX_NAME};")
            print("[+] پاکسازی انجام شد.")


def monitor_progress(stop_event):
    """هر چند ثانیه یک‌بار وضعیت ساخت ایندکس رو از سرور می‌خونه،
    چاپ می‌کنه و بر اساس نرخ واقعی پیشرفت، ETA تخمین می‌زنه."""
    try:
        mon_conn = get_connection()
    except Exception as e:
        print(f"[monitor] اتصال مانیتور برقرار نشد: {e}")
        return

    history = []  # لیست (timestamp, tuples_done) برای محاسبه نرخ

    with mon_conn.cursor() as cur:
        while not stop_event.is_set():
            try:
                cur.execute(
                    """
                    SELECT phase, blocks_done, blocks_total,
                           tuples_done, tuples_total
                    FROM pg_stat_progress_create_index;
                    """
                )
                rows = cur.fetchall()
                if rows:
                    for phase, bd, bt, td, tt in rows:
                        now = time.time()
                        pct = f"{(td / tt * 100):.1f}%" if tt else "?"
                        eta_str = ""

                        if td and tt:
                            history.append((now, td))
                            # فقط با نمونه‌های اخیر (۵ دقیقه گذشته) نرخ رو حساب کن
                            history[:] = [
                                (t, d) for t, d in history if now - t <= 300
                            ]
                            if len(history) >= 2:
                                t0, d0 = history[0]
                                dt, dd = now - t0, td - d0
                                if dt > 0 and dd > 0:
                                    rate = dd / dt  # tuple بر ثانیه
                                    remaining = tt - td
                                    eta_sec = remaining / rate
                                    eta_h = eta_sec / 3600
                                    eta_str = (
                                        f" | نرخ: {rate:.0f} tuple/s "
                                        f"| ETA تقریبی: {eta_h:.1f} ساعت دیگر"
                                    )

                        print(
                            f"[progress] فاز: {phase} | "
                            f"تاپل: {td}/{tt} ({pct}) | بلوک: {bd}/{bt}{eta_str}"
                        )
                else:
                    print("[progress] هنوز اطلاعات پیشرفت در دسترس نیست...")
            except Exception as e:
                print(f"[monitor] خطا در خواندن progress: {e}")
            time.sleep(MONITOR_INTERVAL_SEC)

    mon_conn.close()


def build_index():
    check_free_ram()
    conn = get_connection()

    try:
        with conn.cursor() as cur:
            print("[*] تنظیم پارامترهای session...")
            cur.execute(f"SET maintenance_work_mem = '{MAINTENANCE_WORK_MEM}';")
            cur.execute(
                f"SET max_parallel_maintenance_workers = {MAX_PARALLEL_MAINTENANCE_WORKERS};"
            )
            cur.execute(f"SET max_parallel_workers = {MAX_PARALLEL_WORKERS};")
            # برای دیدن جزئیات worker ها در لاگ سرور (اختیاری)
            cur.execute("SET client_min_messages = NOTICE;")

        cleanup_invalid_index(conn)

        stop_event = threading.Event()
        monitor_thread = threading.Thread(
            target=monitor_progress, args=(stop_event,), daemon=True
        )
        monitor_thread.start()

        start = time.time()
        print(f"[*] شروع ساخت ایندکس {INDEX_NAME} ... (این ممکنه ساعت‌ها طول بکشه)")

        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS {INDEX_NAME}
                ON {TABLE_NAME}
                USING hnsw ({COLUMN_NAME} vector_cosine_ops)
                WITH (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION});
                """
            )

        stop_event.set()
        monitor_thread.join(timeout=2)

        elapsed = time.time() - start
        print(f"[✓] ایندکس با موفقیت ساخته شد. زمان کل: {elapsed/3600:.2f} ساعت")

    except Exception as e:
        print(f"[✗] خطا در ساخت ایندکس: {e}")
        print(
            "    نکته: چون از CONCURRENTLY استفاده شده، جدول قفل نشده و می‌تونید "
            "بعد از رفع مشکل (مثلاً افزایش دیسک/RAM) دوباره همین اسکریپت رو اجرا کنید."
        )
        sys.exit(1)
    finally:
        conn.close()


if __name__ == "__main__":
    build_index()
