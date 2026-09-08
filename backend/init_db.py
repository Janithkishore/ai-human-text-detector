"""Initialize the MySQL database and application tables.

Run from the backend directory after copying .env.example to .env:
    python init_db.py
"""

from app import app, db, ensure_mysql_database


if __name__ == "__main__":
    ensure_mysql_database()
    with app.app_context():
        db.create_all()
    print("MySQL database and detection table are ready.")
