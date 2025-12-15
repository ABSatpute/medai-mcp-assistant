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
from chat_db import init_db, save_message, load_messages, get_all_threads, update_thread_title, cleanup_old_threads, delete_thread

load_dotenv()
init_db()
cleanup_old_threads(10)  # Keep only 10 recent threads

# Get absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = sys.executable

st.set_page_config(page_title="MedAI MCP Agent", layout="wide")

# Add collapsible sidebar functionality
from streamlit.components.v1 import html

# Inject sidebar toggle functionality
html("""
<style>
.sidebar-toggle {
    position: fixed;
    top: 60px;
    left: 10px;
    z-index: 999;
    background: #ff4b4b;
    color: white;
    border: none;
    border-radius: 5px;
    padding: 8px 12px;
    cursor: pointer;
    font-size: 16px;
}
.stSidebar > div {
    transition: transform 0.3s ease;
}
.sidebar-collapsed .stSidebar > div {
    transform: translateX(-100%);
}
</style>

<script>
function toggleSidebar() {
    const body = document.body;
    const isCollapsed = body.classList.contains('sidebar-collapsed');
    
    if (isCollapsed) {
        body.classList.remove('sidebar-collapsed');
        localStorage.setItem('sidebarCollapsed', 'false');
    } else {
        body.classList.add('sidebar-collapsed');
        localStorage.setItem('sidebarCollapsed', 'true');
    }
}

// Restore state
if (localStorage.getItem('sidebarCollapsed') === 'true') {
    document.body.classList.add('sidebar-collapsed');
}
</script>

<button class="sidebar-toggle" onclick="toggleSidebar()">☰</button>
""", height=0)

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
    st.session_state.current_thread_id = None  # Start with no active thread
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
if st.session_state.current_thread_id and not st.session_state.messages:
    st.session_state.messages = load_messages(st.session_state.current_thread_id)

# Utility functions
def analyze_user_intent(query, prescription_medicines=None):
    """Analyze user intent using NLP instead of keyword matching"""
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    context = ""
    if prescription_medicines:
        context = f"User has prescription with medicines: {', '.join(prescription_medicines)}. "
    
    prompt = f"""{context}Analyze this user query and return the intent as JSON:

User query: "{query}"

Return JSON with:
{{
  "intent": "availability|store_location|price|medicine_info|general",
  "confidence": 0.9,
  "entities": ["medicine_name1", "medicine_name2"],
  "action_needed": "check_availability|find_stores|get_price|provide_info|general_response"
}}

Examples:
- "Is this available?" → {{"intent": "availability", "action_needed": "check_availability"}}
- "Where can I buy this?" → {{"intent": "store_location", "action_needed": "find_stores"}}
- "How much does it cost?" → {{"intent": "price", "action_needed": "get_price"}}
- "What is paracetamol?" → {{"intent": "medicine_info", "action_needed": "provide_info", "entities": ["paracetamol"]}}

Return only JSON, no other text."""
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = response.content.strip()
        
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
            
        return json.loads(content)
    except:
        return {"intent": "general", "confidence": 0.5, "action_needed": "general_response"}

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
        
        # Try to extract JSON from response
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        elif content.startswith('```'):
            content = content.replace('```', '').strip()
            
        return json.loads(content)
    except Exception as e:
        return {"medicines": [], "error": f"Analysis failed: {str(e)}", "raw_response": response.content if 'response' in locals() else "No response"}

def generate_title(messages):
    if len(messages) < 1:
        return "New Chat"
    
    # Always use AI to generate title from first user message
    first_user_msg = next((m['content'] for m in messages if m['role'] == 'user'), "New Chat")
    
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = f"Generate a short 5-word title for this medical query:\n{first_user_msg}"
    
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content[:60]
    except:
        # Fallback to smart truncation at word boundary
        words = first_user_msg.split()
        if len(words) <= 6:
            return first_user_msg
        return " ".join(words[:6]) + "..."

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
                
                # Check if this server has the tool
                tools_response = await session.list_tools()
                tool_names = [t.name for t in tools_response.tools]
                
                if tool_name not in tool_names:
                    return f"Error: Tool '{tool_name}' not found on server '{server_name}'"
                
                # Call the tool
                result = await session.call_tool(tool_name, arguments)
                
                if result and result.content:
                    return result.content[0].text
                else:
                    return f"Tool returned empty result"
    except Exception as e:
        return f"Error calling tool: {str(e)}"

async def process_query(query: str, image_base64: str = None, conversation_history: list = None, user_location: dict = None):
    """Process user query as medical store expert"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Check if we have prescription data in session
    prescription_context = ""
    if hasattr(st.session_state, 'prescription_data') and st.session_state.prescription_data:
        medicines = st.session_state.prescription_data.get("medicines", [])
        if medicines:
            prescription_context = f"User has uploaded prescription with medicines: {', '.join(medicines)}. "
    
    # Build conversation context
    context = ""
    if conversation_history:
        recent_context = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        context = "\n".join([f"{msg['role']}: {msg['content'][:100]}..." for msg in recent_context])
    
    system_prompt = f"""You are MedAI, a medical store expert. {prescription_context}

You have access to these tools:
- execute_sql: Query medical database for medicine availability, prices, stock
- get_nearby_stores: Find nearby medical stores with coordinates and map display

DECISION RULES:
- Location queries → get_nearby_stores
- Medicine queries → execute_sql  
- General info → direct answer

Context: {context}

EXAMPLES:
User: "Give me nearby stores" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Find nearest pharmacy" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Where can I buy medicines?" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 10}}}}

User: "Show me store locations" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Pharmacies near me" → {{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

User: "Is Dolo available?" → {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE m.medicine_name LIKE '%Dolo%' AND ss.stock_quantity > 0"}}}}

User: "Price of paracetamol?" → {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT medicine_name, price FROM medicines WHERE medicine_name LIKE '%paracetamol%'"}}}}

User: "Do you have aspirin?" → {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE m.medicine_name LIKE '%aspirin%' AND ss.stock_quantity > 0"}}}}

User: "What is paracetamol used for?" → {{"use_tool": false, "answer": "Paracetamol is used for pain relief and fever reduction."}}

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
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            content = json_match.group()
        
        print(f"DEBUG: Raw LLM response: {response.content}")
        print(f"DEBUG: Cleaned content: {content}")
        
        decision = json.loads(content)
        print(f"DEBUG: Parsed decision: {decision}")
        
        # Show SQL query for verification
        if decision.get("use_tool") and decision.get("tool") == "execute_sql":
            sql_query = decision["arguments"].get("sql_query", "")
            print(f"🔍 GENERATED SQL: {sql_query}")
        
        if decision.get("use_tool"):
            tool_name = decision["tool"]
            arguments = decision["arguments"]
            
            print(f"DEBUG: Calling tool: {tool_name} with args: {arguments}")
            
            # Determine server
            if tool_name == "get_nearby_stores":
                server_name = "medical-map"
                if user_location:
                    arguments.setdefault("latitude", user_location["latitude"])
                    arguments.setdefault("longitude", user_location["longitude"])
            else:
                server_name = "medical-database"
            
            result = await call_mcp_tool(server_name, tool_name, arguments)
            
            # Check if this is map server result for map display
            if tool_name == "get_nearby_stores":
                try:
                    data = json.loads(result)
                    if isinstance(data, list) and len(data) > 0 and 'latitude' in data[0]:
                        st.session_state.last_stores = data
                        st.session_state.show_map = True
                        st.info("📍 Map displayed below.")
                except:
                    pass
            
            # Let LLM format the response intelligently
            format_prompt = f"""User asked: "{query}"
Database result: {result}

As a medical store expert, provide a natural, helpful response based on the user's question and the database results. Don't just show raw data - explain it conversationally."""

            format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            formatted_response = format_llm.invoke([HumanMessage(content=format_prompt)])
            
            return formatted_response.content
        else:
            return decision.get("answer", response.content)
            
    except json.JSONDecodeError:
        return response.content

# UI
st.title("🏥 MedAI - MCP Agent")
st.caption("Production-ready medical AI with MCP tools")

# Show current chat title
if st.session_state.current_thread_id:
    threads = get_all_threads()
    current_thread = next((t for t in threads if t[0] == st.session_state.current_thread_id), None)
    current_title = current_thread[1] if current_thread else "New Chat"
    st.subheader(f"💬 {current_title}")
else:
    st.subheader("💬 No Active Chat")
    st.info("Click 'New Chat' to start a conversation")

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
            
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{'🟢 ' if is_current else ''}{title}", 
                    key=f"thread_{thread_id}",
                    use_container_width=True
                ):
                    load_chat(thread_id)
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"del_{thread_id}", help="Delete chat"):
                    delete_thread(thread_id)
                    if is_current:  # If deleting current chat, clear session
                        st.session_state.current_thread_id = None
                        st.session_state.messages = []
                        st.session_state.show_map = False
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
if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = 0

uploaded_file = st.file_uploader("📎 Upload prescription image", type=["png","jpg","jpeg"], key=f"uploader_{st.session_state.file_uploader_key}")

# Chat input at bottom - disable when no active conversation
has_active_thread = st.session_state.current_thread_id is not None
has_messages = len(st.session_state.messages) > 0

if not has_active_thread:
    st.info("💬 Click 'New Chat' to start a conversation")
    user_input = st.chat_input("Start a new chat to begin conversation", disabled=True)
else:
    user_input = st.chat_input("Ask me anything about health, medicines, or upload a prescription...")

if user_input:
    # Create new thread if none exists
    if not st.session_state.current_thread_id:
        st.session_state.current_thread_id = str(uuid.uuid4())
    
    # Only process when user actually submits a question
    # Hide map when new query is asked
    st.session_state.show_map = False
    
    # Intelligent message handling  
    if uploaded_file:
        display_message = f"📷 {user_input}"
        processing_query = f"I uploaded a prescription image and want to: {user_input}"
    else:
        display_message = user_input
        processing_query = user_input
    
    # Save user message to database
    location = st.session_state.user_location
    save_message(st.session_state.current_thread_id, 'user', display_message, location)
    
    # Add to session state
    user_msg = {'role': 'user', 'content': display_message, 'location': location}
    st.session_state.messages.append(user_msg)
    
    with st.chat_message('user'):
        st.markdown(display_message)
    
    # Process with MCP
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            image_base64 = None
            if uploaded_file:
                image_bytes = uploaded_file.read()
                image_base64 = base64.b64encode(image_bytes).decode()
            
            # Handle image analysis
            if image_base64:
                st.info("🔍 Analyzing prescription image...")
                prescription_data = analyze_prescription_image(image_base64)
                
                if prescription_data.get("medicines"):
                    # Store prescription data for future queries
                    st.session_state.prescription_data = prescription_data
                    
                    medicines = prescription_data["medicines"]
                    medicine_list = ", ".join(medicines)
                    
                    # Enhanced query with prescription context
                    enhanced_query = f"""PRESCRIPTION CONTEXT: Found medicines: {medicine_list}
USER REQUEST: {processing_query}
Respond as medical expert based on user intent."""
                    
                    response = asyncio.run(process_query(
                        enhanced_query,
                        None,
                        st.session_state.messages,
                        st.session_state.user_location
                    ))
                else:
                    response = "I couldn't identify medicines in this image. Please upload a clearer prescription or tell me the medicine names."
            else:
                # Intelligent text query processing
                enhanced_text_query = processing_query
                
                # Add prescription context if available
                if hasattr(st.session_state, 'prescription_data') and st.session_state.prescription_data:
                    medicines = st.session_state.prescription_data.get("medicines", [])
                    if medicines:
                        medicine_list = ", ".join(medicines)
                        enhanced_text_query = f"""CONTEXT: User previously uploaded prescription with: {medicine_list}
CURRENT QUERY: {processing_query}
Respond as medical expert considering both prescription context and current question."""
                
                response = asyncio.run(process_query(
                    enhanced_text_query, 
                    None, 
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
    
    # Clear uploaded file after processing (just the upload box, keep chat data)
    if uploaded_file:
        st.session_state.file_uploader_key += 1
        st.rerun()
    
    # Auto-generate title after 1 message (immediate AI title)
    if len(st.session_state.messages) >= 1 and st.session_state.current_thread_id:
        threads = get_all_threads()
        current_thread = next((t for t in threads if t[0] == st.session_state.current_thread_id), None)
        if current_thread and (current_thread[1] == "New Chat" or "..." in current_thread[1]):
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
