import sqlite3
from werkzeug.security import generate_password_hash

# 1. Connect and Clean Slate
print("Resetting database...")
conn = sqlite3.connect("users.db")
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS users")

# 2. Create Table with ALL required columns
cursor.execute("""
    CREATE TABLE users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT DEFAULT 'User',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")

# 3. Create SuperAdmin
super_pass = generate_password_hash("admin123")
cursor.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
              ("superadmin", super_pass, "SuperAdmin"))

print("Database reset complete. Login with: superadmin / admin123")
conn.commit()
conn.close()