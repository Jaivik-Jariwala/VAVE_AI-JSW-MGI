# Filename: create_user_db.py
import sqlite3
import os
from werkzeug.security import generate_password_hash
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
USER_DB_PATH = BASE_DIR / "users.db"

def init_user_db():
    """Creates the users.db database and users table."""
    if USER_DB_PATH.exists():
        print(f"Database {USER_DB_PATH} already exists. Deleting and recreating...")
        os.remove(USER_DB_PATH)

    conn = None
    try:
        conn = sqlite3.connect(USER_DB_PATH)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """)
        print("Created 'users' table.")
        
        # --- Add Default Admin User ---
        default_username = "admin"
        default_password = "password"
        hashed_password = generate_password_hash(default_password)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (default_username, hashed_password)
        )
        print(f"Inserted default user: 'admin' with password: 'password'")

        # --- ADDED: New Superuser ---
        su_username = "superuser"
        su_password = "admin"  # Using "admin" as the password per your request
        su_hashed_password = generate_password_hash(su_password)
        
        cursor.execute(
            "INSERT INTO users (username, password_hash) VALUES (?, ?)",
            (su_username, su_hashed_password)
        )
        print(f"Inserted new user: 'superuser' with password: 'admin'")
        
        
        conn.commit()
        print(f"User database '{USER_DB_PATH}' created successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_user_db()