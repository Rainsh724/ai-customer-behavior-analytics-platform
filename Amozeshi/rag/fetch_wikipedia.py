import json
import os
import requests
from urllib.parse import quote

WIKI_API = "https://fa.wikipedia.org/w/api.php"

TITLES = [
    "هوش مصنوعی",
    "یادگیری ماشین",
    "پردازش زبان طبیعی",
    "بینایی ماشین",
    "مدل زبانی بزرگ"
]

HEADERS = {
    "User-Agent": "PersianWikiDatasetProject/1.0"
}

def fetch_article(title):
    params = {
        "action": "query",
        "format": "json",
        "formatversion": "2",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
        "redirects": True,
    }
    response = requests.get(WIKI_API, headers=HEADERS, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()
    page = data.get("query", {}).get("pages", [])
    if not page:
        return None
    page = page[0]

    if page.get("missing"):
        return None

    text = page.get("extract", "").strip()

    if not text:
        return None

    final_title = page.get("title", title)
    encoded_title = quote(final_title.replace(" ", "_"))

    return {
        "page_id": page.get("pageid"),
        "title": final_title,
        "content": text,
        "source": f"https://fa.wikipedia.org/wiki/{encoded_title}"
    }

def main():
    articles = []
    for title in TITLES:
        print(f"مقاله : {title} در حال بارگذاری...")
        try:
            article = fetch_article(title)
            if article:
                articles.append(article)
                print(f"مقاله بارگذاری شد: {title}")
            else:
                print(f"مقاله یافت نشد برای عنوان: {title}")
        except requests.RequestException as e:
            print(f"خطا در بارگذاری مقاله: {title} - {e}")

    with open("wiki_articles.json", "w", encoding="utf-8") as file:
        json.dump(articles, file, ensure_ascii=False, indent=2)

    print(f"تعداد مقالات بارگذاری شده: {len(articles)}")

if __name__ == "__main__":
    main()