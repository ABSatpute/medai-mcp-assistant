import sqlite3
from datetime import datetime

DB_PATH = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_threads (
        thread_id TEXT PRIMARY KEY,
        title TEXT,
        created_at TEXT,
        updated_at TEXT
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS chat_messages (
        message_id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        location_lat REAL,
        location_lon REAL,
        created_at TEXT,
        FOREIGN KEY (thread_id) REFERENCES chat_threads(thread_id) ON DELETE CASCADE
    )''')
    
    c.execute('CREATE INDEX IF NOT EXISTS idx_thread_created ON chat_messages(thread_id, created_at)')
    conn.commit()
    conn.close()

def save_message(thread_id, role, content, location=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    now = datetime.now().isoformat()
    
    c.execute('INSERT OR IGNORE INTO chat_threads (thread_id, created_at, updated_at) VALUES (?, ?, ?)',
              (thread_id, now, now))
    
    lat = location['latitude'] if location else None
    lon = location['longitude'] if location else None
    
    c.execute('''INSERT INTO chat_messages (thread_id, role, content, location_lat, location_lon, created_at)
                 VALUES (?, ?, ?, ?, ?, ?)''', (thread_id, role, content, lat, lon, now))
    
    c.execute('UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?', (now, thread_id))
    conn.commit()
    conn.close()

def load_messages(thread_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content, location_lat, location_lon 
                 FROM chat_messages WHERE thread_id = ? ORDER BY created_at''', (thread_id,))
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for role, content, lat, lon in rows:
        msg = {'role': role, 'content': content}
        if lat and lon:
            msg['location'] = {'latitude': lat, 'longitude': lon}
        messages.append(msg)
    return messages

def get_all_threads():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT thread_id, title, updated_at FROM chat_threads 
                 ORDER BY updated_at DESC''')
    rows = c.fetchall()
    conn.close()
    return [(tid, title or "Untitled Chat", updated) for tid, title, updated in rows]

def update_thread_title(thread_id, title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE chat_threads SET title = ? WHERE thread_id = ?', (title, thread_id))
    conn.commit()
    conn.close()
