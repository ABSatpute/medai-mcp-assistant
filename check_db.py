import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import call_mcp_tool

async def check_database():
    print("Checking database contents...")
    
    # Check total medicines
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT COUNT(*) as total FROM medicines"
    })
    print(f"Total medicines: {result}")
    
    # Check sample medicine names
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "SELECT medicine_name FROM medicines LIMIT 10"
    })
    print(f"Sample medicines: {result}")
    
    # Check database schema
    result = await call_mcp_tool("medical-database", "execute_sql", {
        "sql_query": "DESCRIBE medicines"
    })
    print(f"Medicine table structure: {result}")

if __name__ == "__main__":
    asyncio.run(check_database())
