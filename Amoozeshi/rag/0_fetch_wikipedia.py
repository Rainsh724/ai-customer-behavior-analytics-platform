import json
import time
import datetime
from pathlib import Path
from urllib.parse import quote

import requests


TITLES = [
    "هوش مصنوعی",
    "پردازش زبان‌های طبیعی",
    "یادگیری ماشین",
]

API_URL = "https://fa.wikipedia.org/w/api.php"

HEADERS = {
    "User-Agent": "BootcampRAG/1.0"
}

OUT_DIR = Path("Amoozeshi/rag/data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = OUT_DIR / "pages.jsonl"


def fetch_page(title: str):
    """
    یک صفحه از ویکی‌پدیا را با API می‌گیرد و متن ساده آن را برمی‌گرداند.
    """
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "format": "json",
        "titles": title,
    }

    response = requests.get(
        API_URL,
        params=params,
        headers=HEADERS,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    if not pages:
        return None

    page = list(pages.values())[0]

    if "missing" in page:
        return None

    page_title = page.get("title", title)

    return {
        "title": page_title,
        "pageid": page.get("pageid"),
        "source": f"https://fa.wikipedia.org/wiki/{quote(page_title.replace(' ', '_'))}",
        "fetched_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "text": page.get("extract", ""),
    }


def main():
    fetched_count = 0

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for title in TITLES:
            try:
                page = fetch_page(title)

                if page and page["text"].strip():
                    f.write(json.dumps(page, ensure_ascii=False) + "\n")
                    fetched_count += 1
                    print(f"✅ {page['title']} fetched. chars={len(page['text'])}")
                else:
                    print(f"❌ {title} not found or empty.")

            except Exception as e:
                print(f"⚠️ Error for {title}: {e}")

            # برای رعایت ادب نسبت به سرور ویکی‌پدیا
            time.sleep(1.0)

    print(f"Done. {fetched_count} pages saved to {OUT_FILE}")


if __name__ == "__main__":
    main()