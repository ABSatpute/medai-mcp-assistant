#!/usr/bin/env python3
"""
Test script to check what medicines are in the database
and test the medicine search functionality
"""

import asyncio
import sys
import os
from sqlalchemy import create_engine, text

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app import call_mcp_tool, fuzzy_match_medicine

DB_URI = "mysql+pymysql://root:Akash#3112@localhost:3306/medical_chatbot"

async def test_database_connection():
    """Test if we can connect to the database"""
    try:
        engine = create_engine(DB_URI)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) as count FROM medicines")).fetchone()
            print(f"Database connected. Total medicines: {result.count}")
            return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False

async def check_medicine_samples():
    """Check what medicines are actually in the database"""
    try:
        result = await call_mcp_tool("medical-database", "execute_sql", {
            "sql_query": "SELECT medicine_name FROM medicines LIMIT 10"
        })
        print(f"Sample medicines in database:")
        print(result)
        return result
    except Exception as e:
        print(f"Error fetching medicines: {e}")
        return None

async def test_medicine_search(medicine_name):
    """Test searching for a specific medicine"""
    print(f"\nTesting search for: '{medicine_name}'")
    
    # Test fuzzy matching
    matched_name = fuzzy_match_medicine(medicine_name)
    print(f"   Fuzzy match result: '{matched_name}'")
    
    # Test database search
    try:
        result = await call_mcp_tool("medical-database", "execute_sql", {
            "sql_query": f"""
                SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity 
                FROM medicines m 
                JOIN store_stock ss ON m.medicine_id = ss.medicine_id 
                JOIN medical_stores ms ON ss.store_id = ms.store_id 
                WHERE m.medicine_name LIKE '%{medicine_name}%' 
                AND ss.stock_quantity > 0
                LIMIT 5
            """
        })
        print(f"   Database search result:")
        print(f"   {result}")
    except Exception as e:
        print(f"   Database search failed: {e}")

async def test_prescription_medicines():
    """Test the specific medicines from your prescription"""
    prescription_medicines = [
        "Betaloc 100mg",
        "Dorzolamidum 10 mg", 
        "Cimetidine 50 mg",
        "Oxprelol 50mg",
        "Liposomal Amphotericin B"
    ]
    
    print(f"\nTesting prescription medicines:")
    for medicine in prescription_medicines:
        await test_medicine_search(medicine)

async def main():
    print("MedAI Database Test")
    print("=" * 50)
    
    # Test database connection
    if not await test_database_connection():
        return
    
    # Check sample medicines
    await check_medicine_samples()
    
    # Test prescription medicines
    await test_prescription_medicines()
    
    print(f"\nTest completed!")

if __name__ == "__main__":
    asyncio.run(main())
