import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import call_mcp_tool

async def test_information_queries():
    print("=== TESTING MEDICINE INFORMATION QUERIES ===")
    
    # Test 1: Get information about a medicine that exists
    print("\n1. Testing information query for existing medicine:")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT medicine_name, medicine_type, brand_name, description, price, pack_size FROM medicines WHERE medicine_name LIKE '%Crocin%' LIMIT 1"
    })
    print(f"Result: {result}")
    
    # Test 2: Test the schema is working
    print("\n2. Testing schema - all columns should work:")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT medicine_name, medicine_type, brand_name, description, price, created_at, pack_size FROM medicines LIMIT 1"
    })
    print(f"Result: {result}")
    
    # Test 3: Test if "uses" column still causes error (should fail)
    print("\n3. Testing 'uses' column (should fail):")
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT medicine_name, uses FROM medicines LIMIT 1"
    })
    print(f"Result: {result}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_information_queries())
