import json
import sqlite3
from datetime import datetime
from pathlib import Path


DB_PATH = Path("wiki_agent_memory.db")


def get_connection():
    return sqlite3.connect(str(DB_PATH))


def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS searches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            results_summary TEXT NOT NULL,
            page_titles TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT NOT NULL,
            summary TEXT NOT NULL,
            source_titles TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS comparisons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic1 TEXT NOT NULL,
            topic2 TEXT NOT NULL,
            comparison_result TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            related_topics TEXT DEFAULT '',
            created_at TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print("Wiki Agent Memory initialized.")


def save_search(query: str, results_summary: str, page_titles: list):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO searches (query, results_summary, page_titles, created_at) VALUES (?, ?, ?, ?)",
        (query, results_summary, json.dumps(page_titles, ensure_ascii=False), now)
    )
    conn.commit()
    conn.close()


def save_summary(topic: str, summary: str, source_titles: list):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO summaries (topic, summary, source_titles, created_at) VALUES (?, ?, ?, ?)",
        (topic, summary, json.dumps(source_titles, ensure_ascii=False), now)
    )
    conn.commit()
    conn.close()


def save_comparison(topic1: str, topic2: str, result: str):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    cursor.execute(
        "INSERT INTO comparisons (topic1, topic2, comparison_result, created_at) VALUES (?, ?, ?, ?)",
        (topic1, topic2, result, now)
    )
    conn.commit()
    conn.close()


def save_conversation(role: str, content: str, related_topics: list = None):
    conn = get_connection()
    cursor = conn.cursor()
    now = datetime.now().isoformat()
    topics_json = json.dumps(related_topics or [], ensure_ascii=False)
    cursor.execute(
        "INSERT INTO conversations (role, content, related_topics, created_at) VALUES (?, ?, ?, ?)",
        (role, content, topics_json, now)
    )
    conn.commit()
    conn.close()


def get_recent_searches(limit: int = 5) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT query, results_summary, page_titles, created_at FROM searches ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "query": r[0],
            "results_summary": r[1],
            "page_titles": json.loads(r[2]) if r[2] else [],
            "created_at": r[3],
        }
        for r in rows
    ]


def get_recent_summaries(limit: int = 5) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT topic, summary, source_titles, created_at FROM summaries ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "topic": r[0],
            "summary": r[1],
            "source_titles": json.loads(r[2]) if r[2] else [],
            "created_at": r[3],
        }
        for r in rows
    ]


def get_recent_conversations(limit: int = 10) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content, related_topics, created_at FROM conversations ORDER BY created_at DESC LIMIT ?",
        (limit,)
    )
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "role": r[0],
            "content": r[1],
            "related_topics": json.loads(r[2]) if r[2] else [],
            "created_at": r[3],
        }
        for r in rows
    ]


def search_in_memory(query: str) -> list:
    conn = get_connection()
    cursor = conn.cursor()
    
    results = []
    
    cursor.execute(
        "SELECT query, results_summary FROM searches WHERE query LIKE ? ORDER BY created_at DESC LIMIT 3",
        (f"%{query}%",)
    )
    for row in cursor.fetchall():
        results.append({
            "type": "search",
            "query": row[0],
            "summary": row[1][:200],
        })
    
    cursor.execute(
        "SELECT topic, summary FROM summaries WHERE topic LIKE ? OR summary LIKE ? ORDER BY created_at DESC LIMIT 3",
        (f"%{query}%", f"%{query}%")
    )
    for row in cursor.fetchall():
        results.append({
            "type": "summary",
            "topic": row[0],
            "summary": row[1][:200],
        })
    
    conn.close()
    return results


def get_memory_context() -> str:
    recent_searches = get_recent_searches(3)
    recent_summaries = get_recent_summaries(3)
    
    context_parts = []
    
    if recent_searches:
        context_parts.append("جستجوهای اخیر کاربر:")
        for s in recent_searches:
            context_parts.append(f"  - «{s['query']}»")
    
    if recent_summaries:
        context_parts.append("خلاصه‌های تولید شده:")
        for s in recent_summaries:
            context_parts.append(f"  - {s['topic']}: {s['summary'][:100]}")
    
    return "\n".join(context_parts) if context_parts else ""


init_db()