import streamlit as st
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
import base64
import folium
from streamlit_folium import st_folium
from dotenv import load_dotenv
import os
import sys
import uuid
from chat_db import init_db, save_message, load_messages, get_all_threads, update_thread_title

load_dotenv()
init_db()

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = sys.executable

st.set_page_config(page_title="MedAI MCP Agent", layout="wide")

# Check URL params FIRST before initializing session state
try:
    query_params = st.query_params
    if 'lat' in query_params and 'lon' in query_params:
        new_lat = float(query_params['lat'])
        new_lon = float(query_params['lon'])
        # Force set in session state
        st.session_state.user_location = {"latitude": new_lat, "longitude": new_lon}
        st.session_state.location_detected = True
        st.query_params.clear()
except Exception as e:
    pass

# Initialize session state
if 'current_thread_id' not in st.session_state:
    st.session_state.current_thread_id = str(uuid.uuid4())
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'last_stores' not in st.session_state:
    st.session_state.last_stores = None
if 'show_map' not in st.session_state:
    st.session_state.show_map = False
if 'user_location' not in st.session_state:
    st.session_state.user_location = None
if 'location_detected' not in st.session_state:
    st.session_state.location_detected = False

# Load messages from database if thread exists
if not st.session_state.messages:
    st.session_state.messages = load_messages(st.session_state.current_thread_id)

# Utility functions
def generate_title(messages):
    if len(messages) < 3:
        return messages[0]['content'][:50] if messages else "New Chat"
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    last_three = [m['content'] for m in messages[-3:]]
    prompt = f"Generate a short 5-word title:\n{' | '.join(last_three)}"
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content[:60]
    except:
        return messages[0]['content'][:50]

def new_chat():
    st.session_state.current_thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.show_map = False

def load_chat(thread_id):
    st.session_state.current_thread_id = thread_id
    st.session_state.messages = load_messages(thread_id)
    st.session_state.show_map = False

async def get_mcp_tools():
    """Connect to MCP servers and get available tools"""
    tools = []
    
    servers = [
        os.path.join(BASE_DIR, "mcp_servers/database_server.py"),
        os.path.join(BASE_DIR, "mcp_servers/map_server.py"),
        os.path.join(BASE_DIR, "mcp_servers/prescription_server.py")
    ]
    
    for server_path in servers:
        params = StdioServerParameters(command=PYTHON_PATH, args=[server_path])
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools_response = await session.list_tools()
                tools.extend(tools_response.tools)
    
    return tools

async def call_mcp_tool(tool_name: str, arguments: dict):
    """Call MCP tool on appropriate server"""
    servers = [
        os.path.join(BASE_DIR, "mcp_servers/database_server.py"),
        os.path.join(BASE_DIR, "mcp_servers/map_server.py"),
        os.path.join(BASE_DIR, "mcp_servers/prescription_server.py")
    ]
    
    for server_path in servers:
        params = StdioServerParameters(command=PYTHON_PATH, args=[server_path])
        try:
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    
                    # Check if this server has the tool
                    tools_response = await session.list_tools()
                    tool_names = [t.name for t in tools_response.tools]
                    
                    if tool_name not in tool_names:
                        continue
                    
                    # Call the tool
                    result = await session.call_tool(tool_name, arguments)
                    
                    if result and result.content:
                        return result.content[0].text
                    else:
                        return f"Tool returned empty result"
        except Exception as e:
            print(f"Error on {server_path}: {e}")
            continue
    
    return "Error: Tool not found"

async def process_query(query: str, image_base64: str = None, conversation_history: list = None, user_location: dict = None):
    """Process user query with MCP tools"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Default location if not provided
    if not user_location:
        user_location = {"latitude": 18.566039, "longitude": 73.766370}
    
    # Get available tools
    tools = await get_mcp_tools()
    
    # Build tool descriptions
    tool_info = []
    for t in tools:
        schema = json.dumps(t.inputSchema, indent=2)
        tool_info.append(f"Tool: {t.name}\nDescription: {t.description}\nSchema: {schema}")
    
    # Build conversation context
    context = ""
    if conversation_history:
        recent = conversation_history[-4:]
        context = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])
    
    # Create different system prompts based on input type
    if image_base64 and query == "Analyze this prescription image":
        # Image-only prompt
        system_prompt = f"""You are MedAI, a medical assistant with access to these tools:

{chr(10).join(tool_info)}

The user uploaded a prescription image. Extract all prescription data using extract_prescription_data tool.

Response format:
{{"use_tool": true, "tool": "extract_prescription_data", "arguments": {{"image_base64": "..."}}}}"""
    
    elif image_base64 and query != "Analyze this prescription image":
        # Image + custom text prompt
        system_prompt = f"""You are MedAI, a medical assistant with access to these tools:

{chr(10).join(tool_info)}

The user uploaded a prescription image AND asked: "{query}"

CRITICAL RULES:
- If query mentions "availability", "check", "find": do multi-step (extract_prescription_data → execute_sql)
- If query mentions "near me", "nearby": do multi-step (extract_prescription_data → get_nearby_stores → execute_sql)
- Otherwise: just extract prescription data

Multi-step format:
{{"use_tool": true, "steps": [{{"tool": "extract_prescription_data", "arguments": {{"image_base64": "..."}}}}, {{"tool": "execute_sql", "arguments": {{"sql_query": "..."}}}}]}}

Single tool format:
{{"use_tool": true, "tool": "extract_prescription_data", "arguments": {{"image_base64": "..."}}}}"""
    
    else:
        # Text-only prompt (original)
        system_prompt = f"""You are MedAI, a medical assistant with access to these tools:

{chr(10).join(tool_info)}

CRITICAL RULES:
- For ANY query with "near me", "nearby", "closest", "nearest": MUST use get_nearby_stores tool
- For "find stores": use get_nearby_stores (NOT execute_sql)
- For medicine availability at specific stores: use execute_sql
- For price queries: use execute_sql
- For prescription images WITHOUT availability request: use extract_prescription_data only
- For prescription images WITH availability request ("check availability", "find medicines", "availability"): do multi-step
- When user uploads image: automatically use extract_prescription_data even if not explicitly requested

MULTI-STEP QUERY PATTERNS:
1. "nearest store(s)" queries: ALWAYS use get_nearby_stores first, then execute_sql with store_id filter
2. "compare prices at nearest stores": 
   - Step 1: get_nearby_stores(limit=N)
   - Step 2: execute_sql with WHERE ms.store_id IN (store_ids_from_step1)
3. Prescription analysis ONLY (image without availability request):
   - Single step: extract_prescription_data(image_base64)
4. Prescription availability check (image + "availability"/"check"/"find" in query):
   - Step 1: extract_prescription_data(image_base64)
   - Step 2: execute_sql to check medicine availability using extracted medicine names
5. Prescription + nearby availability (image + "near me"/"nearby"):
   - Step 1: extract_prescription_data(image_base64)
   - Step 2: get_nearby_stores(limit=5)
   - Step 3: execute_sql with store filter for medicine availability

SQL Rules:
- For "available" or "in stock": add "AND ss.stock_quantity > 0"
- Use LIKE '%%term%%' for text searches
- For counts: SELECT COUNT(*) as count FROM ...
- NEVER use LIMIT inside subqueries
- For nearest stores, use get_nearby_stores tool, NOT distance calculation in SQL

Conversation context:
{context}

Response format for single tool:
{{"use_tool": true, "tool": "tool_name", "arguments": {{...}}}}

Response format for multi-step (use when query mentions "nearest" + other criteria):
{{"use_tool": true, "steps": [{{"tool": "get_nearby_stores", "arguments": {{"limit": N}}}}, {{"tool": "execute_sql", "arguments": {{"sql_query": "..."}}}}]}}

For direct answers:
{{"use_tool": false, "answer": "your response"}}"""
    
    # Add image context if present
    if image_base64:
        query += "\n[User uploaded a prescription image]"
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=query)
    ]
    
    response = llm.invoke(messages)
    content = response.content.strip()
    
    # Extract JSON
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0].strip()
    elif "```" in content:
        content = content.split("```")[1].split("```")[0].strip()
    
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        content = match.group()
    
    try:
        decision = json.loads(content)
        
        print(f"DEBUG: LLM Decision: {decision}")
        print(f"DEBUG: Has image: {image_base64 is not None}")
        print(f"DEBUG: Query: {query}")
        
        if decision.get("use_tool"):
            # Use the location passed as parameter
            location = user_location
            
            # DEBUG: Print which tool is being used
            if "steps" in decision:
                print(f"DEBUG: Multi-step execution")
                for i, step in enumerate(decision["steps"]):
                    print(f"  Step {i+1}: {step['tool']}")
            else:
                print(f"DEBUG: Single tool execution: {decision.get('tool')}")
                print(f"DEBUG: User location being passed: {location}")
            
            # Handle multi-step execution
            if "steps" in decision:
                results = []
                store_ids = []
                
                for i, step in enumerate(decision["steps"]):
                    # Add location for map tools
                    if step["tool"] == "get_nearby_stores":
                        step["arguments"]["latitude"] = location["latitude"]
                        step["arguments"]["longitude"] = location["longitude"]
                    
                    # Add image if needed
                    if step["tool"] == "extract_prescription_data" and image_base64:
                        step["arguments"]["image_base64"] = image_base64
                    elif image_base64 and "image_base64" in str(step.get("arguments", {})):
                        step["arguments"]["image_base64"] = image_base64
                    
                    # If this is step 2 and we have store IDs from step 1, inject them
                    if i > 0 and store_ids and step["tool"] == "execute_sql":
                        sql = step["arguments"]["sql_query"]
                        ids_str = ",".join(map(str, store_ids))
                        if "WHERE" in sql.upper():
                            sql = sql.replace("WHERE", f"WHERE ms.store_id IN ({ids_str}) AND")
                        else:
                            sql += f" WHERE ms.store_id IN ({ids_str})"
                        step["arguments"]["sql_query"] = sql
                    
                    result = await call_mcp_tool(step["tool"], step["arguments"])
                    results.append(result)
                    
                    # Extract store IDs from map tool result
                    if step["tool"] == "get_nearby_stores":
                        try:
                            store_data = json.loads(result)
                            store_ids = [s.get("store_id") for s in store_data if "store_id" in s]
                        except:
                            pass
                
                # Return the final result
                return results[-1]
            
            # Single tool execution
            if decision["tool"] == "extract_prescription_data" and image_base64:
                decision["arguments"]["image_base64"] = image_base64
            elif image_base64 and "image_base64" in str(decision.get("arguments", {})):
                decision["arguments"]["image_base64"] = image_base64
            
            if decision["tool"] == "get_nearby_stores":
                decision["arguments"]["latitude"] = location["latitude"]
                decision["arguments"]["longitude"] = location["longitude"]
            
            result = await call_mcp_tool(decision["tool"], decision["arguments"])
            
            # Check if it's a count result
            try:
                data = json.loads(result)
                if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                    keys = list(data[0].keys())
                    if len(keys) <= 2 and any(k in ['count', 'total', 'sum', 'avg', 'max', 'min'] for k in [k.lower() for k in keys]):
                        value = list(data[0].values())[0]
                        return f"There are **{value}** results matching your query."
            except:
                pass
            
            return result
        else:
            return decision.get("answer", content)
    except:
        return content

# UI
st.title("🏥 MedAI - MCP Agent")
st.caption("Production-ready medical AI with MCP tools")

# Sidebar
with st.sidebar:
    st.header("💬 Chat Sessions")
    
    if st.button("➕ New Chat", use_container_width=True):
        new_chat()
        st.rerun()
    
    st.divider()
    
    # Load threads from database
    threads = get_all_threads()
    if threads:
        st.subheader("Previous Chats")
        for thread_id, title, _ in threads:
            is_current = thread_id == st.session_state.current_thread_id
            
            if st.button(
                f"{'🟢 ' if is_current else ''}{title}", 
                key=f"thread_{thread_id}",
                use_container_width=True
            ):
                load_chat(thread_id)
                st.rerun()
    
    st.divider()
    
    st.header("ℹ️ About")
    st.write("MedAI uses MCP (Model Context Protocol) to provide:")
    st.write("- 💊 Prescription analysis")
    st.write("- 🔍 Medicine search")
    st.write("- 💰 Price comparison (Local + Online)")
    st.write("- 🤖 AI-powered web scraping")
    st.write("- 🏪 Store locator")
    st.write("- 💬 Medical Q&A")
    
    if st.button("🗑️ Clear Current Chat"):
        st.session_state.messages = []
        st.session_state.show_map = False
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        content = msg['content']
        # Try to parse as JSON for table display
        try:
            data = json.loads(content)
            if isinstance(data, list) and len(data) > 0:
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
            else:
                st.markdown(content)
        except:
            st.markdown(content)

# Location detection with button trigger
with st.expander("📍 Your Location", expanded=not st.session_state.location_detected):
    if st.session_state.user_location:
        st.write(f"**Current:** {st.session_state.user_location['latitude']:.6f}, {st.session_state.user_location['longitude']:.6f}")
    
    if st.session_state.location_detected:
        st.success("✅ Location detected!")
        if st.button("🔄 Detect Again"):
            st.session_state.location_detected = False
            st.rerun()
    else:
        st.error("❌ Location not detected - Click button below")
        
        if st.button("🎯 Detect My Location", type="primary"):
            from streamlit.components.v1 import html
            html("""
                <script>
                if (navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        (pos) => {
                            window.location.href = '?lat=' + pos.coords.latitude + '&lon=' + pos.coords.longitude;
                        },
                        (err) => {
                            alert('Location access denied: ' + err.message);
                        }
                    );
                } else {
                    alert('Geolocation not supported by your browser');
                }
                </script>
            """, height=0)

# File upload above chat input
uploaded_file = st.file_uploader("📎 Upload prescription image", type=["png","jpg","jpeg"])

# Chat input at bottom
user_input = st.chat_input("Ask me anything about health, medicines, or upload a prescription...")

if user_input or uploaded_file:
    # Hide map when new query is asked
    st.session_state.show_map = False
    
    # Handle image-only upload
    if uploaded_file and not user_input:
        user_input = "Analyze this prescription image"
    
    # Save user message to database
    location = st.session_state.user_location
    save_message(st.session_state.current_thread_id, 'user', user_input, location)
    
    # Add to session state
    user_msg = {'role': 'user', 'content': user_input, 'location': location}
    st.session_state.messages.append(user_msg)
    
    with st.chat_message('user'):
        st.markdown(user_input)
    
    # Process with MCP
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            image_base64 = None
            if uploaded_file:
                image_bytes = uploaded_file.read()
                image_base64 = base64.b64encode(image_bytes).decode()
            
            response = asyncio.run(process_query(
                user_input, 
                image_base64, 
                st.session_state.messages,
                st.session_state.user_location
            ))
        
        # Try to parse as JSON for table display
        try:
            data = json.loads(response)
            print(f"DEBUG: Parsed JSON data type: {type(data)}")
            print(f"DEBUG: Data keys: {data.keys() if isinstance(data, dict) else 'Not a dict'}")
            
            # Check if it's prescription data
            if isinstance(data, dict) and ('doctor_info' in data or 'patient_info' in data or 'medicines' in data):
                # Display prescription data in formatted way
                st.subheader("📋 Prescription Analysis")
                
                # Dynamic display of all sections
                for section_key, section_data in data.items():
                    if section_key == 'doctor_info' and section_data:
                        st.write("**👨‍⚕️ Doctor Information**")
                        for key, value in section_data.items():
                            if value:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    elif section_key == 'patient_info' and section_data:
                        st.write("**👤 Patient Information**")
                        for key, value in section_data.items():
                            if value:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    elif section_key == 'prescription_details' and section_data:
                        st.write("**📄 Prescription Details**")
                        for key, value in section_data.items():
                            if value:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    elif section_key == 'medicines' and section_data:
                        st.write("**💊 Prescribed Medicines**")
                        if isinstance(section_data, list) and section_data:
                            import pandas as pd
                            df = pd.DataFrame(section_data)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("⚠️ No medicines detected")
                    
                    elif section_key == 'medical_info' and section_data:
                        st.write("**🩺 Medical Information**")
                        for key, value in section_data.items():
                            if value:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                    
                    elif section_key == 'additional_data' and section_data:
                        st.write("**📝 Additional Information**")
                        for key, value in section_data.items():
                            if value:
                                st.write(f"• {key.replace('_', ' ').title()}: {value}")
                
                # Save the formatted response
                save_message(st.session_state.current_thread_id, 'assistant', response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
            
            # Only show table if it's a list of dicts (database results)
            elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                import pandas as pd
                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                
                # Check if this is store data (has latitude/longitude)
                if 'latitude' in data[0] and 'longitude' in data[0]:
                    st.session_state.last_stores = data
                    st.session_state.show_map = True
                    st.info("📍 Map displayed below. It will hide when you ask a new question, but the data remains here.")
                
                # Save the actual JSON data
                save_message(st.session_state.current_thread_id, 'assistant', response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
            else:
                # Not a table - show as text
                st.markdown(response)
                save_message(st.session_state.current_thread_id, 'assistant', response)
                st.session_state.messages.append({'role': 'assistant', 'content': response})
                
        except (json.JSONDecodeError, ValueError):
            # Plain text response
            st.markdown(response)
            save_message(st.session_state.current_thread_id, 'assistant', response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
    
    # Auto-generate title after 3 messages
    if len(st.session_state.messages) >= 3:
        threads = get_all_threads()
        current_thread = next((t for t in threads if t[0] == st.session_state.current_thread_id), None)
        if current_thread and current_thread[1] == "Untitled Chat":
            title = generate_title(st.session_state.messages)
            update_thread_title(st.session_state.current_thread_id, title)

# Display map OUTSIDE chat context (like reference code)
if 'show_map' in st.session_state and st.session_state.show_map and st.session_state.last_stores:
    st.header("🗺️ Store Locations Map")
    
    stores = st.session_state.last_stores
    
    # Check if user location is available
    if not st.session_state.user_location:
        st.warning("⚠️ Location not detected - using default location for map")
        user_lat = 18.566039
        user_lon = 73.766370
    else:
        user_lat = st.session_state.user_location['latitude']
        user_lon = st.session_state.user_location['longitude']
    
    st.info(f"📍 Your location: {user_lat:.6f}, {user_lon:.6f}")
    
    if st.session_state.location_detected:
        st.success("✅ Using browser-detected location")
    else:
        st.warning("⚠️ Click '📍 Your Location' to detect")
    
    # Store selection for routing
    selected_store = st.selectbox(
        "Select store to show route:",
        options=range(len(stores)),
        format_func=lambda i: f"{'🏆 ' if i==0 else ''}{stores[i]['store_name']} ({stores[i].get('distance', 0):.2f} km)"
    )
    
    # Create map
    m = folium.Map(
        location=[user_lat, user_lon], 
        zoom_start=13,
        tiles='OpenStreetMap'
    )
    
    # Add animated user cartoon marker
    user_icon_html = """
    <div style="
        font-size: 40px;
        animation: bounce 1s ease-in-out infinite;
        filter: drop-shadow(0 0 10px rgba(255,0,0,0.8));
    ">🧍</div>
    <style>
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
    </style>
    """
    
    folium.Marker(
        [user_lat, user_lon],
        popup=f"<b>👤 YOUR LOCATION</b><br>Lat: {user_lat:.6f}<br>Lon: {user_lon:.6f}",
        tooltip="👤 YOU ARE HERE",
        icon=folium.DivIcon(html=user_icon_html)
    ).add_to(m)
    
    # Add routing to selected store using OSRM
    import requests
    selected = stores[selected_store]
    
    route_loaded = False
    try:
        # Get route from OSRM (free routing service)
        osrm_url = f"http://router.project-osrm.org/route/v1/driving/{user_lon},{user_lat};{selected['longitude']},{selected['latitude']}?overview=full&geometries=geojson"
        
        st.write(f"DEBUG: Fetching route from OSRM...")
        response = requests.get(osrm_url, timeout=10)
        st.write(f"DEBUG: Response status: {response.status_code}")
        
        if response.status_code == 200:
            route_data = response.json()
            st.write(f"DEBUG: Route data received: {route_data.get('code')}")
            
            if route_data.get('routes'):
                route = route_data['routes'][0]
                coords = route['geometry']['coordinates']
                # Convert from [lon, lat] to [lat, lon]
                route_coords = [[c[1], c[0]] for c in coords]
                
                st.write(f"DEBUG: Route has {len(route_coords)} points")
                
                # Add route line
                folium.PolyLine(
                    route_coords,
                    color='blue',
                    weight=5,
                    opacity=0.8,
                    popup=f"Route: {route['distance']/1000:.2f} km, {route['duration']/60:.0f} min"
                ).add_to(m)
                
                st.success(f"🚗 Route: {route['distance']/1000:.2f} km • ⏱️ {route['duration']/60:.0f} min")
                route_loaded = True
            else:
                st.warning("No routes found in response")
        else:
            st.error(f"OSRM API error: {response.status_code}")
    except Exception as e:
        st.error(f"Could not load route: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
    
    if not route_loaded:
        st.info("Showing straight line distance instead of route")
    
    # Add store markers
    colors = ['green', 'blue', 'purple', 'orange', 'darkred']
    for i, store in enumerate(stores):
        is_nearest = i == 0
        is_selected = i == selected_store
        
        folium.Marker(
            [store['latitude'], store['longitude']],
            popup=f"""
            <div style="width:250px">
                <b>{'🏆 ' if is_nearest else ''}{store.get('store_name', 'Unknown Store')}</b><br>
                {'<span style="color:green">Nearest Store</span><br>' if is_nearest else ''}
                📍 {store.get('address', 'N/A')}<br>
                📞 {store.get('phone_number', 'N/A')}<br>
                📏 {store.get('distance', 0):.2f} km away
            </div>
            """,
            tooltip=f"{'🏆 ' if is_nearest else ''}{store.get('store_name', 'Store')} ({store.get('distance', 0):.2f} km)",
            icon=folium.Icon(
                color='blue' if is_selected else ('green' if is_nearest else colors[i % len(colors)]), 
                icon='star' if is_nearest else 'plus', 
                prefix='fa'
            )
        ).add_to(m)
    
    # Display map
    st_folium(m, width=700, height=500)
    
    # Clear map flag after display
    if st.button("Hide Map"):
        st.session_state.show_map = False
        st.rerun()
