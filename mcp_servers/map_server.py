from mcp.server import Server
from mcp.types import Tool, TextContent
from sqlalchemy import create_engine, text
import json
from decimal import Decimal

app = Server("medical-map")

DB_URI = "mysql+pymysql://root:Akash#3112@localhost:3306/medical_chatbot"
engine = create_engine(DB_URI)

def decimal_to_float(obj):
    """Convert Decimal to float for JSON serialization"""
    if isinstance(obj, Decimal):
        return float(obj)
    raise TypeError

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_nearby_stores",
            description="Get nearby medical stores with coordinates for map display, optionally filtered by medicine availability",
            inputSchema={
                "type": "object",
                "properties": {
                    "latitude": {
                        "type": "number",
                        "description": "User latitude (default: 18.558091)"
                    },
                    "longitude": {
                        "type": "number",
                        "description": "User longitude (default: 73.793439)"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Number of stores to return (default: 10)"
                    },
                    "medicine_filter": {
                        "type": "string",
                        "description": "Optional: Filter stores that have this medicine in stock"
                    }
                }
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "get_nearby_stores":
        lat = arguments.get("latitude", 18.558091)
        lon = arguments.get("longitude", 73.793439)
        limit = arguments.get("limit", 10)
        medicine_filter = arguments.get("medicine_filter")
        
        print(f"   Received latitude: {lat}")
        print(f"   Received longitude: {lon}")
        print(f"   Limit: {limit}")
        print(f"   Medicine filter: {medicine_filter}")
        
        if medicine_filter:
            # Query stores that have the specific medicine in stock
            query = text("""
                SELECT DISTINCT ms.store_id, ms.store_name, ms.address, ms.phone_number, 
                       ms.latitude, ms.longitude,
                       (6371 * acos(cos(radians(:lat)) * cos(radians(ms.latitude)) * 
                       cos(radians(ms.longitude) - radians(:lon)) + sin(radians(:lat)) * 
                       sin(radians(ms.latitude)))) AS distance,
                       m.medicine_name, m.price, ss.stock_quantity
                FROM medical_stores ms
                JOIN store_stock ss ON ms.store_id = ss.store_id
                JOIN medicines m ON ss.medicine_id = m.medicine_id
                WHERE m.medicine_name LIKE :medicine AND ss.stock_quantity > 0
                ORDER BY distance
                LIMIT :limit
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {
                    "lat": lat, "lon": lon, "limit": limit, 
                    "medicine": f"%{medicine_filter}%"
                })
                rows = [dict(row._mapping) for row in result]
        else:
            # Query all nearby stores
            query = text("""
                SELECT store_id, store_name, address, phone_number, latitude, longitude,
                       (6371 * acos(cos(radians(:lat)) * cos(radians(latitude)) * 
                       cos(radians(longitude) - radians(:lon)) + sin(radians(:lat)) * 
                       sin(radians(latitude)))) AS distance
                FROM medical_stores
                ORDER BY distance
                LIMIT :limit
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {"lat": lat, "lon": lon, "limit": limit})
                rows = [dict(row._mapping) for row in result]
        
        print(f"   Found {len(rows)} stores")
        if rows:
            print(f"   Nearest store: {rows[0]['store_name']} at {rows[0]['distance']:.2f} km")
        
        # Convert to JSON-serializable format
        json_str = json.dumps(rows, default=decimal_to_float, indent=2)
        
        return [TextContent(type="text", text=json_str)]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())
