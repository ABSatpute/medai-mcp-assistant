from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from dotenv import load_dotenv
import asyncio
import json
import os
import sys
import uuid
import base64
import re
from datetime import datetime
from chat_db import init_db, save_message, load_messages, get_all_threads, update_thread_title, cleanup_old_threads, delete_thread

load_dotenv()
init_db()
cleanup_old_threads(10)

# Get absolute paths for MCP servers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = sys.executable

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medai_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

def analyze_prescription_image(image_base64):
    """Direct prescription analysis using main LLM"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    messages = [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": """Analyze this prescription image and extract all visible information. Return ONLY a JSON object with this structure:
{
  "medicines": ["medicine name 1", "medicine name 2"],
  "doctor_info": "Doctor name and details",
  "instructions": "Dosage and instructions",
  "patient_info": "Patient details if visible"
}

Return only the JSON, no other text."""}
        ])
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
            
        return json.loads(content)
    except Exception as e:
        return {"medicines": [], "error": f"Analysis failed: {str(e)}"}

def generate_title(messages):
    if len(messages) < 1:
        return "New Chat"
    
    first_user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "New Chat")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Generate a short 5-word title for this medical query:\n{first_user_msg}"
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content[:60]
    except:
        words = first_user_msg.split()
        if len(words) <= 6:
            return first_user_msg
        return " ".join(words[:6]) + "..."

async def call_mcp_tool(server_name: str, tool_name: str, arguments: dict):
    """Call MCP tool on appropriate server"""
    server_map = {
        "medical-database": "database_server.py",
        "medical-map": "map_server.py"
    }
    
    if server_name not in server_map:
        return f"Error: Unknown server '{server_name}'"
    
    server_path = os.path.join(BASE_DIR, "mcp_servers", server_map[server_name])
    params = StdioServerParameters(command=PYTHON_PATH, args=[server_path])
    
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                
                if tool_name not in tool_names:
                    return f"Error: Tool '{tool_name}' not found on server '{server_name}'"
                
                result = await session.call_tool(tool_name, arguments)
                
                if result and result.content:
                    return result.content[0].text
                else:
                    return f"Tool returned empty result"
    except Exception as e:
        return f"Error calling tool: {str(e)}"

async def process_query(query: str, image_base64: str = None, conversation_history: list = None, user_location: dict = None):
    """Process user query as medical store expert - EXACT copy from mcp_app.py"""
    
    # Force tool usage for button clicks
    if "Find medicines near me" in query:
        result = await call_mcp_tool("medical-database", "execute_sql", {
            "sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE ss.stock_quantity > 0 LIMIT 10"
        })
        
        format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        formatted_response = format_llm.invoke([HumanMessage(content=f"User asked: '{query}'. Database result: {result}. Provide a helpful response about available medicines.")])
        return formatted_response.content, result
    
    if "Show me nearby medical stores" in query:
        arguments = {"latitude": 18.566039, "longitude": 73.766370, "limit": 5}
        if user_location:
            arguments["latitude"] = user_location["latitude"]
            arguments["longitude"] = user_location["longitude"]
            
        result = await call_mcp_tool("medical-map", "get_nearby_stores", arguments)
        
        format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        formatted_response = format_llm.invoke([HumanMessage(content=f"User asked: '{query}'. Store data: {result}. Provide a helpful response about nearby medical stores.")])
        return formatted_response.content, result
    
    # Continue with normal LLM processing for other queries
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Build conversation context
    context = ""
    if conversation_history:
        recent_context = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        context = "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in recent_context])
    
    if user_location:
        location_info = f"User Location: {user_location.get('latitude', 'Unknown')}, {user_location.get('longitude', 'Unknown')}"
    else:
        location_info = "User location not available"
    
    system_prompt = f"""You are MedAI, a medical store expert.

{location_info}

You have access to these tools:
- execute_sql: Query medical database for medicine availability, prices, stock
- get_nearby_stores: Find nearby medical stores with coordinates and map display

CRITICAL: You MUST use tools for these queries. Never give generic responses.

DECISION RULES:
- ANY query about "nearby stores", "medical stores", "pharmacies", "store locator" → ALWAYS use get_nearby_stores
- ANY query about "medicines", "drugs", "find medicine", "medicine search" → ALWAYS use execute_sql
- General health info → direct answer

Context: {context}

EXAMPLES - FOLLOW EXACTLY:
User: "Show me nearby medical stores" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Find medicines near me" → {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE ss.stock_quantity > 0 LIMIT 10"}}}}

User: "Store locator" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Medicine search" → {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE ss.stock_quantity > 0 LIMIT 10"}}}}

IMPORTANT: If user asks about stores/pharmacies/locations → use get_nearby_stores
If user asks about medicines/drugs/availability → use execute_sql

Return ONLY JSON, no markdown, no explanation."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    
    try:
        content = response.content.strip()
        
        # Clean up JSON response
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
        
        # Extract JSON if wrapped in text
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        decision = json.loads(content)
        
        if decision.get("use_tool"):
            tool_name = decision["tool"]
            arguments = decision["arguments"]
            
            # Determine server
            if tool_name == "get_nearby_stores":
                server_name = "medical-map"
                if user_location:
                    arguments.setdefault("latitude", user_location["latitude"])
                    arguments.setdefault("longitude", user_location["longitude"])
            else:
                server_name = "medical-database"
            
            result = await call_mcp_tool(server_name, tool_name, arguments)
            
            # Let LLM format the response intelligently
            format_prompt = f"""User asked: "{query}"
Database result: {result}

As a medical store expert, provide a natural, helpful response based on the user's question and the database results. Don't just show raw data - explain it conversationally."""

            format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            formatted_response = format_llm.invoke([HumanMessage(content=format_prompt)])
            
            return formatted_response.content, result
        else:
            return decision.get("answer", response.content), None
            
    except json.JSONDecodeError:
        return response.content, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/debug')
def debug():
    return render_template('../debug_basic.html')

@app.route('/debug-full')
def debug_full():
    return render_template('debug_index.html')

@app.route('/api/threads')
def get_threads():
    """Get all chat threads"""
    threads = get_all_threads()
    return jsonify([{
        'id': thread[0],
        'title': thread[1],
        'updated_at': thread[2]
    } for thread in threads])

@app.route('/api/thread/<thread_id>/messages')
def get_messages(thread_id):
    """Get messages for a specific thread"""
    messages = load_messages(thread_id)
    return jsonify(messages)

@app.route('/api/thread/<thread_id>/delete', methods=['DELETE'])
def delete_thread_api(thread_id):
    """Delete a specific thread"""
    delete_thread(thread_id)
    return jsonify({'success': True})

@socketio.on('send_message')
def handle_message(data):
    """Handle new message from client - EXACT logic from mcp_app.py"""
    print(f"🔥 BACKEND: Received message event")
    print(f"📨 Data: {data}")
    
    thread_id = data.get('thread_id', str(uuid.uuid4()))
    message = data.get('message', '')
    image_data = data.get('image', None)
    user_location = data.get('location', None)
    
    print(f"📝 Thread ID: {thread_id}")
    print(f"💬 Message: {message}")
    print(f"📍 Location: {user_location}")
    
    # Save user message
    try:
        save_message(thread_id, 'user', message)
        print(f"✅ User message saved to database")
    except Exception as e:
        print(f"❌ Error saving message: {e}")
    
    # Emit user message back to client
    print(f"📤 Emitting user message back to client")
    emit('message_received', {
        'role': 'user',
        'content': message or '📷 Uploaded prescription image',
        'timestamp': datetime.now().isoformat()
    })
    
    # Process with AI
    try:
        print(f"🤖 Starting AI processing...")
        
        # Get conversation history for context
        conversation_history = load_messages(thread_id)
        print(f"📚 Loaded {len(conversation_history)} previous messages")
        
        if image_data:
            print(f"🖼️ Processing image data...")
            # Extract base64 data
            image_base64 = image_data.split(',')[1] if ',' in image_data else image_data
            
            # Analyze prescription
            prescription_data = analyze_prescription_image(image_base64)
            
            if prescription_data.get("medicines"):
                medicines = prescription_data["medicines"]
                medicine_list = ", ".join(medicines)
                
                response = f"I've analyzed your prescription and found **{medicine_list}**. How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them."
                raw_data = None
                print(f"✅ Prescription analyzed: {medicine_list}")
            else:
                response = "I couldn't identify medicines in this image. Please upload a clearer prescription."
                raw_data = None
                print(f"❌ No medicines found in prescription")
        else:
            print(f"💭 Processing text query with MCP...")
            # Use process_query function
            response, raw_data = asyncio.run(process_query(
                message, 
                None, 
                conversation_history,
                user_location
            ))
            print(f"✅ MCP processing complete")
        
        # Save assistant response
        try:
            save_message(thread_id, 'assistant', response)
            print(f"✅ Assistant response saved to database")
        except Exception as e:
            print(f"❌ Error saving response: {e}")
        
        # Check if we have store data for map display
        if raw_data:
            try:
                data = json.loads(raw_data)
                if isinstance(data, list) and len(data) > 0 and 'latitude' in data[0]:
                    print(f"🗺️ Emitting map data with {len(data)} stores")
                    # Emit map data
                    emit('show_map', {
                        'stores': data,
                        'user_location': user_location
                    })
                else:
                    print(f"📊 Raw data available but not map data")
            except Exception as e:
                print(f"❌ Error processing raw data: {e}")
        
        # Emit assistant response
        print(f"📤 Emitting assistant response")
        emit('message_received', {
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Auto-generate title for new threads
        if len(conversation_history) <= 1:
            print(f"🏷️ Generating title for new thread...")
            title = generate_title([{'role': 'user', 'content': message}])
            update_thread_title(thread_id, title)
            emit('title_updated', {
                'thread_id': thread_id,
                'title': title
            })
            print(f"✅ Title generated: {title}")
        
        print(f"🎉 Message processing complete!")
        
    except Exception as e:
        print(f"💥 ERROR in message processing: {e}")
        import traceback
        traceback.print_exc()
        
        emit('message_received', {
            'role': 'assistant',
            'content': f"Sorry, I encountered an error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        })

if __name__ == '__main__':
    print("🏥 Starting MedAI on http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
