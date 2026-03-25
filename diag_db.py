
import os
import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()

def get_db_connection():
    password = os.getenv("DB_PASS") or os.getenv("DB_PASSWORD")
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=password,
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432")
    )
    return conn

def diagnose():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Check table columns
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'chat_history';")
        cols = [c[0] for c in cur.fetchall()]
        print(f"COLUMNS: {','.join(cols)}")
        
        # Check if table_data has any content
        if 'table_data' in cols:
            cur.execute("SELECT COUNT(*) FROM chat_history WHERE table_data IS NOT NULL;")
            count = cur.fetchone()[0]
            print(f"TABLE_DATA_COUNT: {count}")
        else:
            print("TABLE_DATA_MISSING")

        cur.close()
        conn.close()
    except Exception as e:
        print(f"DIAG_ERROR: {e}")

if __name__ == "__main__":
    diagnose()
