import sqlite3
import os

db_path = "/app/data/edge_client.db"
if not os.path.exists(db_path):
    print("DB NOT FOUND")
    exit()

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    cursor.execute("SELECT user_id, name, nrp, global_label FROM users_local LIMIT 10")
    rows = cursor.fetchall()
    print(f"Total Users: {len(rows)}")
    for r in rows:
        print(f"ID: {r[0]} | Name: {r[1]} | NRP: {r[2]} | Global Label: {r[3]}")
    
    cursor.execute("SELECT COUNT(*) FROM users_local WHERE global_label IS NULL")
    null_count = cursor.fetchone()[0]
    print(f"Users with NULL global_label: {null_count}")
    
except Exception as e:
    print(f"ERROR: {e}")
conn.close()
