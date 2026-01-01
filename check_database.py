import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import call_mcp_tool

async def check_database():
    print("=== DATABASE DIAGNOSIS ===")
    
    # Step 1: Check total medicines count
    print("\n1. Total medicines in database:")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT COUNT(*) as total FROM medicines"
    })
    print(result)
    
    # Step 2: Check medicine table structure
    print("\n2. Medicine table structure:")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "DESCRIBE medicines"
    })
    print(result)
    
    # Step 3: Sample medicine names
    print("\n3. Sample medicine names (first 10):")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT medicine_name FROM medicines LIMIT 10"
    })
    print(result)
    
    # Step 4: Check if prescription medicines exist
    print("\n4. Checking prescription medicines:")
    prescription_medicines = [
        "Betaloc",
        "Dorzolamidum", 
        "Cimetidine",
        "Oxprelol",
        "Amphotericin"
    ]
    
    for medicine in prescription_medicines:
        result = await call_mcp_tool("medical-database", "execute_sql", {
            "sql_query": f"SELECT medicine_name FROM medicines WHERE medicine_name LIKE '%{medicine}%'"
        })
        print(f"   {medicine}: {result}")
    
    print("\n=== DIAGNOSIS COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(check_database())
