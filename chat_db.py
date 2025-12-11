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
    
    # Generate title from first user message
    title = None
    if role == 'user':
        c.execute('SELECT COUNT(*) FROM chat_messages WHERE thread_id = ? AND role = "user"', (thread_id,))
        user_msg_count = c.fetchone()[0]
        if user_msg_count == 0:  # This is the first user message
            title = content[:50] + "..." if len(content) > 50 else content
    
    c.execute('INSERT OR IGNORE INTO chat_threads (thread_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
              (thread_id, title or "New Chat", now, now))
    
    # Update title if this is first user message
    if title:
        c.execute('UPDATE chat_threads SET title = ? WHERE thread_id = ?', (title, thread_id))
    
    lat = location['latitude'] if location else None
    lon = location['longitude'] if location else None
    
    c.execute('''INSERT INTO chat_messages (thread_id, role, content, location_lat, location_lon, created_at)
                 VALUES (?, ?, ?, ?, ?, ?)''', (thread_id, role, content, lat, lon, now))
    
    c.execute('UPDATE chat_threads SET updated_at = ? WHERE thread_id = ?', (now, thread_id))
    conn.commit()
    conn.close()

def load_messages(thread_id, limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT role, content, location_lat, location_lon 
                 FROM chat_messages WHERE thread_id = ? ORDER BY created_at DESC LIMIT ?''', (thread_id, limit))
    rows = c.fetchall()
    conn.close()
    
    messages = []
    for role, content, lat, lon in reversed(rows):  # Reverse to get chronological order
        msg = {'role': role, 'content': content}
        if lat and lon:
            msg['location'] = {'latitude': lat, 'longitude': lon}
        messages.append(msg)
    return messages

def get_all_threads(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''SELECT thread_id, title, updated_at FROM chat_threads 
                 ORDER BY updated_at DESC LIMIT ?''', (limit,))
    rows = c.fetchall()
    conn.close()
    return [(tid, title or "Untitled Chat", updated) for tid, title, updated in rows]

def delete_thread(thread_id):
    """Delete a specific chat thread and all its messages"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('DELETE FROM chat_threads WHERE thread_id = ?', (thread_id,))
    conn.commit()
    conn.close()

def cleanup_old_threads(keep_count=10):
    """Delete old threads beyond keep_count"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''DELETE FROM chat_threads WHERE thread_id NOT IN (
                 SELECT thread_id FROM chat_threads ORDER BY updated_at DESC LIMIT ?)''', (keep_count,))
    conn.commit()
    conn.close()

def update_thread_title(thread_id, title):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE chat_threads SET title = ? WHERE thread_id = ?', (title, thread_id))
    conn.commit()
    conn.close()
