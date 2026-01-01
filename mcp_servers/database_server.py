from mcp.server import Server
from mcp.types import Tool, TextContent
from sqlalchemy import create_engine, text
import json

app = Server("medical-database")

DB_URI = "mysql+pymysql://root:Akash#3112@localhost:3306/medical_chatbot"
engine = create_engine(DB_URI)

DATABASE_SCHEMA = """
Database: medical_chatbot

Tables:
1. medicines
   - medicine_id (INT, PRIMARY KEY)
   - medicine_name (VARCHAR)
   - medicine_type (ENUM: 'tablet','syrup','capsule','injection','other')
   - brand_name (VARCHAR)
   - description (TEXT)
   - price (DECIMAL)
   - created_at (TIMESTAMP)
   - pack_size (INT)

2. medical_stores
   - store_id (INT, PRIMARY KEY)
   - store_name (VARCHAR)
   - address (TEXT)
   - phone_number (VARCHAR)
   - latitude (DECIMAL)
   - longitude (DECIMAL)

3. store_stock
   - stock_id (INT, PRIMARY KEY)
   - store_id (INT, FOREIGN KEY -> medical_stores)
   - medicine_id (INT, FOREIGN KEY -> medicines)
   - stock_quantity (INT)
"""

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="execute_sql",
            description=f"Execute SQL query on medical database. Database schema:\n{DATABASE_SCHEMA}",
            inputSchema={
                "type": "object",
                "properties": {
                    "sql_query": {
                        "type": "string",
                        "description": "SQL SELECT query to execute"
                    }
                },
                "required": ["sql_query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    from decimal import Decimal
    from datetime import datetime
    
    if name == "execute_sql":
        sql_query = arguments["sql_query"]
        
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = [dict(row._mapping) for row in result]
            
            if not rows:
                return [TextContent(type="text", text="No results found")]
            
            # Convert Decimal and datetime to JSON serializable types
            for row in rows:
                for key, value in row.items():
                    if isinstance(value, Decimal):
                        row[key] = float(value)
                    elif isinstance(value, datetime):
                        row[key] = value.isoformat()
            
            return [TextContent(type="text", text=json.dumps(rows, indent=2))]
        except Exception as e:
            return [TextContent(type="text", text=f"SQL Error: {str(e)}")]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())
