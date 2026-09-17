"""
اسکریپت یک‌باره برای ساختن کاربر جدید توی جدول app_users.
اجرا:
    python create_user.py
"""


import getpass
import bcrypt
import psycopg2


def get_admin_conn():
    return psycopg2.connect(
        host="localhost",
        port=5432,
        database="ai_project",
        user="postgres",
        password="",
    )


def main():
    username = input("نام کاربری: ").strip()
    display_name = input("نام نمایشی (اختیاری، Enter برای رد شدن): ").strip() or None
    password = getpass.getpass("رمز عبور: ")
    password_confirm = getpass.getpass("تکرار رمز عبور: ")

    if not username or not password:
        print("نام کاربری و رمز عبور نمی‌توانند خالی باشند.")
        return

    if password != password_confirm:
        print("رمز عبور و تکرار آن یکسان نیستند.")
        return

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    with get_admin_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.app_users
                    (username, password_hash, display_name)
                VALUES (%s, %s, %s)
                """,
                (username, password_hash, display_name),
            )

    print(f"کاربر '{username}' با موفقیت ساخته شد.")


if __name__ == "__main__":
    main()