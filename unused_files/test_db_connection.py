#!/usr/bin/env python3
"""Test database connection for MCP server"""

import sys
from dotenv import load_dotenv

load_dotenv()

def test_mysql_connection():
    try:
        from sqlalchemy import create_engine, text
        
        # Same connection string as MCP server
        DB_URI = "mysql+pymysql://root:Akash#3112@localhost:3306/medical_chatbot"
        print(f"Testing connection: {DB_URI}")
        
        engine = create_engine(DB_URI)
        
        # Test connection
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ MySQL connection: SUCCESS")
            
            # Check if tables exist
            tables = conn.execute(text("SHOW TABLES")).fetchall()
            print(f"✅ Found {len(tables)} tables:")
            for table in tables:
                print(f"   - {table[0]}")
                
            # Test medicines table
            try:
                medicines = conn.execute(text("SELECT COUNT(*) FROM medicines")).fetchone()
                print(f"✅ Medicines table: {medicines[0]} records")
            except Exception as e:
                print(f"❌ Medicines table error: {e}")
                
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("\nPossible issues:")
        print("1. MySQL server not running")
        print("2. Database 'medical_chatbot' doesn't exist")
        print("3. Wrong credentials (root:Akash#3112)")
        print("4. MySQL not on localhost:3306")

if __name__ == "__main__":
    test_mysql_connection()
