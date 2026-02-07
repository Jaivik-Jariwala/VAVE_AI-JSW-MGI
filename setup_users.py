# setup_users.py
import sqlite3
from werkzeug.security import generate_password_hash

conn = sqlite3.connect("users.db")
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'User',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# Add role column if missing (for migration)
try:
    cursor.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'User'")
except sqlite3.OperationalError:
    pass # Column likely already exists

# Create Default SuperAdmin
try:
    pw = generate_password_hash("admin123")
    cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  ("superadmin", pw, "SuperAdmin"))
    print("SuperAdmin created: superadmin / admin123")
except sqlite3.IntegrityError:
    print("SuperAdmin already exists.")

conn.commit()
conn.close()