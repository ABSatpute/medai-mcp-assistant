# app.py - Unified advanced backend (merged & cleaned)
import os
import sys
import json
import re
import uuid
import base64
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import razorpay
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import boto3
from io import BytesIO

# Local DB helpers (assumed present in project)
from chat_db import (
    init_db,
    save_message,
    load_messages,
    get_all_threads,
    update_thread_title,
    cleanup_old_threads,
    delete_thread,
)

# E-commerce models
from models.user import User
from models.cart import Cart
from models.order import Order

load_dotenv()

# Initialize DB and cleanup old threads
init_db()
cleanup_old_threads(10)

# Paths & runtime info
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON_PATH = sys.executable

# Flask + SocketIO
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "medai_secret_key_2024")
app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY", "medai_jwt_secret_2024")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = False  # Never expire (or set to timedelta for specific time)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize JWT
jwt = JWTManager(app)

# Initialize Razorpay client
try:
    razorpay_client = razorpay.Client(auth=(os.getenv("RAZORPAY_KEY_ID"), os.getenv("RAZORPAY_KEY_SECRET")))
    print("Razorpay client initialized successfully")
except Exception as e:
    print(f"Razorpay initialization failed: {e}")
    razorpay_client = None

# Initialize AWS Polly client
try:
    polly_client = boto3.client('polly', region_name='us-east-1')
    print("AWS Polly client initialized successfully")
except Exception as e:
    print(f"AWS Polly initialization failed: {e}")
    polly_client = None

# ---------------------
# Utilities / LLM calls
# ---------------------
def _clean_llm_json_response(raw: str) -> str:
    """Strip markdown fences and return the JSON-like substring if present."""
    if not raw:
        return raw
    s = raw.strip()
    if s.startswith("```json"):
        s = s.replace("```json", "", 1).replace("```", "").strip()
    elif s.startswith("```"):
        s = s.replace("```", "").strip()
    # Try to extract the first {...} block if there is surrounding text
    m = re.search(r"\{.*\}", s, re.DOTALL)
    return m.group() if m else s

def analyze_prescription_image(image_base64: str) -> dict:
    """
    Use main LLM to extract prescription structure.
    Returns a dict: {"medicines": [...], "doctor_info": ..., "instructions": ..., "patient_info": ...}
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [
        HumanMessage(
            content=[
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                {
                    "type": "text",
                    "text": (
                        "Analyze this prescription image and extract all visible information. "
                        "Return ONLY a JSON object with this structure:\n"
                        "{\n"
                        "  \"medicines\": [\"medicine name 1\", \"medicine name 2\"],\n"
                        "  \"doctor_info\": \"Doctor name and details\",\n"
                        "  \"instructions\": \"Dosage and instructions\",\n"
                        "  \"patient_info\": \"Patient details if visible\"\n"
                        "}\n\nReturn only the JSON, no other text."
                    ),
                },
            ]
        )
    ]
    try:
        response = llm.invoke(messages)
        cleaned = _clean_llm_json_response(response.content)
        return json.loads(cleaned)
    except Exception as e:
        return {"medicines": [], "error": f"Analysis failed: {str(e)}"}

# Cache for medicine names to avoid repeated database calls
_medicine_cache = None
_cache_timestamp = None

def get_medicine_names():
    """Get cached medicine names or fetch from database"""
    global _medicine_cache, _cache_timestamp
    import time
    
    # Cache for 5 minutes
    if _medicine_cache is None or (time.time() - _cache_timestamp) > 300:
        try:
            result = asyncio.run(call_mcp_tool("medical-database", "execute_sql", {
                "sql_query": "SELECT DISTINCT medicine_name FROM medicines"
            }))
            
            if result and result.startswith('['):
                import json
                medicines = json.loads(result)
                _medicine_cache = [med['medicine_name'] for med in medicines]
                _cache_timestamp = time.time()
            else:
                _medicine_cache = []
        except:
            _medicine_cache = []
    
    return _medicine_cache

def fuzzy_match_medicine(user_input: str, threshold: float = 0.6) -> str:
    """
    Fast fuzzy matching for medicine names
    """
    try:
        medicine_names = get_medicine_names()
        if not medicine_names:
            return user_input
        
        user_lower = user_input.lower()
        
        # Fast exact match first
        for med_name in medicine_names:
            if user_lower in med_name.lower() or med_name.lower() in user_lower:
                return med_name
        
        # Simple fuzzy matching for common typos
        def quick_similarity(s1, s2):
            s1, s2 = s1.lower(), s2.lower()
            if abs(len(s1) - len(s2)) > 3:  # Skip if length difference too big
                return 0.0
            
            # Count matching characters in order
            matches = 0
            i = j = 0
            while i < len(s1) and j < len(s2):
                if s1[i] == s2[j]:
                    matches += 1
                    i += 1
                    j += 1
                else:
                    i += 1
            
            return matches / max(len(s1), len(s2))
        
        # Find best match quickly
        best_match = user_input
        best_score = 0.0
        
        for med_name in medicine_names:
            # Check against medicine name and first word
            score = quick_similarity(user_input, med_name)
            first_word = med_name.split()[0] if med_name.split() else med_name
            word_score = quick_similarity(user_input, first_word)
            
            final_score = max(score, word_score)
            
            if final_score > best_score and final_score >= threshold:
                best_score = final_score
                best_match = med_name
        
        if best_score >= threshold:
            print(f"FUZZY MATCH: '{user_input}' -> '{best_match}' (score: {best_score:.2f})")
            return best_match
        else:
            return user_input
            
    except Exception as e:
        print(f"FUZZY MATCH ERROR: {str(e)}")
        return user_input

def generate_title(messages):
    """Generate a short title from recent conversation messages using AI."""
    if not messages:
        return "New Chat"
    
    # Extract conversation context
    conversation_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            conversation_text += f"User: {content}\n"
        elif role == "assistant":
            conversation_text += f"Assistant: {content[:100]}...\n"  # Limit assistant response length
    
    try:
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        prompt = f"""Generate a short 4-5 word title that summarizes this medical conversation. Be specific and concise:

{conversation_text.strip()}

Title should reflect the main medical topic or concern discussed."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        title = response.content.strip().split("\n")[0][:50]
        
        # Clean up the title (remove quotes, extra spaces)
        title = title.strip('"\'').strip()
        
        return title if title else "Medical Consultation"
    except Exception as e:
        print(f"❌ Title generation error: {e}")
        # Fallback - use first user message
        first_user_msg = next((m["content"] for m in messages if m.get("role") == "user"), "New Chat")
        words = first_user_msg.split()
        if len(words) <= 5:
            return first_user_msg[:40]
        return " ".join(words[:5]) + "..."
    except Exception:
        # Fallback - smart truncation at word boundary
        words = first_user_msg.split()
        if len(words) <= 6:
            return first_user_msg
        return " ".join(words[:6]) + "..."

# -----------------------
# MCP (tool) communication
# -----------------------
# OPTIMIZATION: Prescription analysis moved to direct function call for better performance
# Removed prescription-server MCP overhead - 50% faster, lower costs
async def call_mcp_tool(server_name: str, tool_name: str, arguments: dict):
    """
    Call an MCP server tool via stdio_client.
    server_name -> maps to a file inside mcp_servers/
    """
    server_map = {
        "medical-database": "database_server.py",
        "medical-map": "map_server.py",
        # Removed prescription-server - now handled directly in app.py for better performance
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
                    return "Tool returned empty result"
    except Exception as e:
        return f"Error calling tool: {str(e)}"

# -------------------------------
# Advanced process_query (async)
# -------------------------------
async def process_query(query: str, image_base64: str = None, conversation_history: list = None, user_location: dict = None):
    """
    Advanced decision-making pipeline. This is the canonical production logic.
    Returns (assistant_text_response, raw_tool_result_or_none)
    """
    import re
    
    # Force certain direct actions if query contains explicit phrases
    if "Find medicines near me" in query or re.search(r"\b(find|search)\b.*\bmedicine(s)?\b", query, re.IGNORECASE) and "near" in query:
        sql = (
            "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity "
            "FROM medicines m "
            "JOIN store_stock ss ON m.medicine_id = ss.medicine_id "
            "JOIN medical_stores ms ON ss.store_id = ms.store_id "
            "WHERE ss.stock_quantity > 0 LIMIT 10"
        )
        result = await call_mcp_tool("medical-database", "execute_sql", {"sql_query": sql})
        format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        formatted = format_llm.invoke([HumanMessage(content=f"User asked: '{query}'. Database result: {result}. Provide a helpful response about available medicines.")])
        return formatted.content, result

    if re.search(r"\b(nearby|near me|nearby stores|pharmacies|store locator|pharmacy)\b", query, re.IGNORECASE):
        print(f"🗺️ DEBUG - Hardcoded store query triggered for: {query}")
        args = {"latitude": 18.566039, "longitude": 73.766370, "limit": 5}
        if user_location:
            args["latitude"] = user_location.get("latitude", args["latitude"])
            args["longitude"] = user_location.get("longitude", args["longitude"])
        print(f"🗺️ DEBUG - Calling map server with args: {args}")
        result = await call_mcp_tool("medical-map", "get_nearby_stores", args)
        print(f"🗺️ DEBUG - Map server result: {result}")
        format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        formatted = format_llm.invoke([HumanMessage(content=f"User asked: '{query}'. Store data: {result}. Provide a helpful response about nearby medical stores.")])
        return formatted.content, result

    # Build system prompt with strict decision rules (Advanced)
    context = ""
    if conversation_history:
        recent = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
        # Preserve full context for prescription information, truncate others
        context_parts = []
        for m in recent:
            content = m['content']
            # Keep full content if it contains prescription info
            if "Prescription contains:" in content or "found **" in content:
                context_parts.append(f"{m['role']}: {content}")
            else:
                context_parts.append(f"{m['role']}: {content[:200]}")
        context = "\n".join(context_parts)

    location_info = f"User Location: {user_location.get('latitude')}, {user_location.get('longitude')}" if user_location else "User location not available"

    system_prompt = f"""You are MedAI, an intelligent medical assistant with advanced contextual understanding.

USER LOCATION: {location_info}

=== CONTEXT INTELLIGENCE ===
PRESCRIPTION AWARENESS:
- If conversation contains prescription analysis with medicines (look for "found **Medicine1, Medicine2**"), remember them
- When user asks "my medicines", "my prescription", "what medicines" → Extract from context first
- Reference previous analysis: "Based on your prescription analysis, your medicines are..."

CONTEXTUAL REFERENCES:
- "that medicine", "those medicines", "them" → Use medicines from context
- "that store", "those stores" → Use stores from context  
- "availability of that" → Use medicine name from context + execute_sql
- Always check context before asking user to repeat information

=== TOOLS AVAILABLE ===
- execute_sql: Query medical database (medicines, prices, stock, stores)
- get_nearby_stores: Find medical stores with GPS coordinates

=== INTELLIGENT DECISION RULES ===

1. PRESCRIPTION CONTEXT QUERIES:
   - "my medicines", "prescription medicines", "what medicines" → Check context first
   - If medicines found in context → Direct answer with context medicines
   - If no context → Ask for clarification

2. MEDICINE QUERIES:
   - "find [medicine]", "availability", "price", "stock" → execute_sql
   - "availability of that" + context has medicine → execute_sql with context medicine

3. ALTERNATIVE MEDICINE QUERIES:
   - "alternatives", "similar medicines", "generic versions" → Suggest alternatives using medical knowledge, then ASK PERMISSION
   - "check availability" (after alternatives suggested) → Check availability of previously suggested alternatives
   - DO NOT automatically check database - always ask user permission first

4. LOCATION QUERIES:
   - "nearby stores", "pharmacies", "medical stores", "store locator" → get_nearby_stores
   - "stores for [medicine]" → execute_sql (include store info)

4. CONTEXTUAL FOLLOW-UPS:
   - Reference previous conversation naturally
   - Maintain conversation flow without repetitive questions

=== RESPONSE FORMAT ===
TOOL USAGE: {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT..."}}}}
DIRECT ANSWER: {{"use_tool": false, "answer": "Based on your prescription, your medicines are: [list]..."}}

=== EXAMPLES ===
Context: "found **Paracetamol 500mg, Ibuprofen 200mg**"
User: "what medicines in my prescription"
→ {{"use_tool": false, "answer": "Based on your prescription analysis, your medicines are: Paracetamol 500mg and Ibuprofen 200mg. Would you like me to check availability or find nearby stores?"}}

Context: Previous mention of "Crocin"  
User: "find availability of that"
→ {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE m.medicine_name LIKE '%Crocin%' AND ss.stock_quantity > 0"}}}}

User: "suggest alternatives for Remdesivir"
→ {{"use_tool": false, "answer": "Based on medical knowledge, alternatives to Remdesivir could include:\n• Favipiravir\n• Molnupiravir\n• Paxlovid\n\nWould you like me to check which of these are available in our store?"}}

User: "yes, check availability" (after seeing alternatives)
→ {{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name, ss.stock_quantity FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE (m.medicine_name LIKE '%Favipiravir%' OR m.medicine_name LIKE '%Molnupiravir%' OR m.medicine_name LIKE '%Paxlovid%') AND ss.stock_quantity > 0"}}}}

Always end medical responses with: "Always consult your doctor for medical advice."
"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Build full conversation history for LLM
    messages = [SystemMessage(content=system_prompt)]
    
    # Add full conversation history
    if conversation_history:
        for msg in conversation_history:
            if msg['role'] == 'user':
                messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                messages.append(AIMessage(content=msg['content']))
    
    # Add current query
    messages.append(HumanMessage(content=query))
    
    response = llm.invoke(messages)

    cleaned = _clean_llm_json_response(response.content)
    try:
        decision = json.loads(cleaned)
        print(f"DEBUG - LLM Decision: {decision}")  # Debug what LLM decided
    except Exception:
        # LLM didn't return JSON -> treat as plain answer
        print(f"DEBUG - LLM returned plain text: {response.content[:100]}...")  # Debug non-JSON response
        return response.content, None

    if decision.get("use_tool"):
        tool_name = decision.get("tool")
        arguments = decision.get("arguments", {})

        if tool_name == "get_nearby_stores":
            server_name = "medical-map"
            if user_location:
                arguments.setdefault("latitude", user_location["latitude"])
                arguments.setdefault("longitude", user_location["longitude"])
        else:
            server_name = "medical-database"

        result = await call_mcp_tool(server_name, tool_name, arguments)
        
        # DEBUG: Print what MCP tool returned
        print(f"DEBUG - MCP Tool Result: {result}")

        # Extract medicine names from the SQL query for better response formatting
        medicine_names = []
        if "sql_query" in arguments:
            sql = arguments["sql_query"]
            # Extract medicine names from LIKE clauses or IN clauses
            import re
            # Try LIKE pattern first
            likes = re.findall(r"LIKE '%([^%]+)%'", sql)
            if likes:
                medicine_names = likes
            else:
                # Try IN pattern
                in_match = re.search(r"IN \(([^)]+)\)", sql)
                if in_match:
                    # Extract quoted values
                    quoted_values = re.findall(r"'([^']+)'", in_match.group(1))
                    medicine_names = quoted_values

        # Format for user
        format_prompt = f"""User asked: "{query}"
Database result: {result}
Searched medicines: {', '.join(medicine_names) if medicine_names else 'Not specified'}

If the database result shows "No results found":
1. Specifically mention which medicines are not available: {', '.join(medicine_names) if medicine_names else 'the requested medicines'}
2. Offer practical next steps:
   - "Would you like me to check if we can order these for you?"
   - "I can help you find contact information for our pharmacy team"
3. DO NOT suggest more alternatives - user already received alternatives
4. Be specific and solution-oriented

If results are found:
- Provide clear availability with store details and prices

Always end with: 'Always consult your doctor for medical advice.'"""
        format_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        formatted_response = format_llm.invoke([HumanMessage(content=format_prompt)])
        return formatted_response.content, result
    else:
        # Direct answer branch
        return decision.get("answer", response.content), None

# -------------------------
# Flask routes (static UI)
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/debug")
def debug():
    return render_template("../debug_basic.html")

@app.route("/debug-full")
def debug_full():
    return render_template("debug_index.html")

@app.route("/api/threads")
def api_get_threads():
    threads = get_all_threads()
    return jsonify([{"id": t[0], "title": t[1], "updated_at": t[2]} for t in threads])

@app.route("/api/thread/<thread_id>/messages")
def api_get_messages(thread_id):
    return jsonify(load_messages(thread_id))

@app.route("/api/thread/<thread_id>/delete", methods=["DELETE"])
def api_delete_thread(thread_id):
    delete_thread(thread_id)
    return jsonify({"success": True})

@app.route("/api/medicine-names")
def get_medicine_names_api():
    """Get all medicine names for autocomplete suggestions"""
    try:
        medicine_names = get_medicine_names()
        return jsonify({"medicines": medicine_names})
    except Exception as e:
        print(f"❌ Error fetching medicine names: {e}")
        return jsonify({"medicines": []})

# -------------------------
# SocketIO message handler
# -------------------------
@socketio.on("analyze_prescription")
def handle_analyze_prescription(data):
    """Handle prescription image analysis from Find Medicines popup"""
    try:
        image_data = data.get("image", "")
        print(f"🔍 DEBUG - Received prescription analysis request")
        
        if not image_data:
            emit("prescription_analysis_result", {"error": "No image provided"})
            return
        
        # Extract base64 data
        if "base64," in image_data:
            image_base64 = image_data.split("base64,")[1]
        else:
            image_base64 = image_data
        
        # Use direct prescription analysis (optimized - no MCP overhead)
        prescription_data = analyze_prescription_image(image_base64)
        
        print(f"🔍 DEBUG - Prescription analysis result: {prescription_data}")
        
        if prescription_data and not prescription_data.get("error"):
            emit("prescription_analysis_result", {"success": True, "data": prescription_data})
        else:
            error_msg = prescription_data.get("error", "Analysis failed") if prescription_data else "No analysis result"
            emit("prescription_analysis_result", {"error": error_msg})
        
    except Exception as e:
        print(f"❌ ERROR in analyze_prescription: {str(e)}")
        emit("prescription_analysis_result", {"error": str(e)})

@socketio.on("search_medicines")
def handle_search_medicines(data):
    """
    Handle medicine search from Find Medicines popup with store-centric ranking
    
    ENHANCED FUNCTIONALITY:
    - Multi-medicine availability checking
    - Store-centric results (not medicine-centric)
    - Intelligent ranking: Availability (70%) + Stock (20%) + Distance (10%)
    - Decreasing order by best match stores
    """
    try:
        medicines = data.get("medicines", [])
        user_location = data.get("location", {"latitude": 18.566039, "longitude": 73.766370})
        print(f"🔍 DEBUG - Received medicine search request: {medicines}")
        
        if not medicines:
            emit("medicine_search_result", {"error": "No medicines provided"})
            return
        
        # Get all medicine availability data
        all_medicine_data = []
        
        for medicine in medicines:
            result = asyncio.run(call_mcp_tool("medical-database", "execute_sql", {
                "sql_query": f"SELECT m.medicine_name, m.price, ms.store_name, ms.store_id, ss.stock_quantity, ms.latitude, ms.longitude FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE m.medicine_name LIKE '%{medicine}%' AND ss.stock_quantity > 0"
            }))
            
            print(f"🔍 DEBUG - Database result for '{medicine}': Found {len(json.loads(result)) if result.startswith('[') else 0} items")
            
            if result and result.startswith('['):
                medicine_data = json.loads(result)
                for item in medicine_data:
                    item['requested_medicine'] = medicine
                all_medicine_data.extend(medicine_data)
        
        print(f"🔍 DEBUG - Total medicine data items: {len(all_medicine_data)}")

        # Group by store and calculate availability ranking
        store_rankings = {}
        for item in all_medicine_data:
            store_name = item['store_name']
            store_id = item['store_id']
            requested_medicine = item['requested_medicine']
            
            if store_name not in store_rankings:
                store_rankings[store_name] = {
                    "store_name": store_name,
                    "store_id": store_id,
                    "medicines_available": 0,
                    "total_medicines_requested": len(medicines),
                    "medicines": [],
                    "total_stock": 0,
                    "latitude": item.get('latitude', 18.566039),
                    "longitude": item.get('longitude', 73.766370),
                    "found_medicines": set()
                }
            
            # Add medicine to store if not already added for this requested medicine
            medicine_key = f"{requested_medicine}_{item['medicine_name']}"
            if medicine_key not in store_rankings[store_name]["found_medicines"]:
                store_rankings[store_name]["medicines"].append({
                    "name": item['medicine_name'],
                    "price": item['price'],
                    "stock": item['stock_quantity'],
                    "requested_for": requested_medicine
                })
                store_rankings[store_name]["found_medicines"].add(medicine_key)
                store_rankings[store_name]["total_stock"] += item['stock_quantity']
        
        # Calculate medicines_available based on unique requested medicines found
        for store in store_rankings.values():
            unique_requested_medicines = set()
            for medicine in store["medicines"]:
                unique_requested_medicines.add(medicine["requested_for"])
            store["medicines_available"] = len(unique_requested_medicines)
            del store["found_medicines"]
        
        # Calculate availability percentage and distance-based ranking
        for store in store_rankings.values():
            store["availability_percentage"] = round((store["medicines_available"] / store["total_medicines_requested"]) * 100)
            
            # Calculate distance from user location
            try:
                import math
                lat1, lon1 = user_location["latitude"], user_location["longitude"]
                lat2, lon2 = store["latitude"], store["longitude"]
                
                # Haversine formula for distance
                R = 6371
                dlat = math.radians(lat2 - lat1)
                dlon = math.radians(lon2 - lon1)
                a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
                c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                distance = R * c
                store["distance"] = round(distance, 2)
            except:
                store["distance"] = 0
            
            # Calculate ranking score: availability (70%) + stock (20%) + proximity (10%)
            availability_score = store["availability_percentage"]
            stock_score = min(store["total_stock"] / 10, 100)
            proximity_score = max(100 - (store["distance"] * 10), 0)
            
            store["ranking_score"] = (availability_score * 0.7) + (stock_score * 0.2) + (proximity_score * 0.1)
        
        # Sort stores by ranking score (descending order)
        ranked_stores = sorted(store_rankings.values(), key=lambda x: x["ranking_score"], reverse=True)
        
        print(f"🔍 DEBUG - Store rankings: {len(ranked_stores)} stores found")
        
        # Send enhanced results back to popup
        emit("medicine_search_result", {
            "success": True, 
            "results": ranked_stores,
            "search_summary": {
                "medicines_requested": medicines,
                "total_stores_found": len(ranked_stores),
                "best_match": ranked_stores[0]["store_name"] if ranked_stores else "None"
            }
        })
        
    except Exception as e:
        print(f"❌ ERROR in search_medicines: {str(e)}")
        emit("medicine_search_result", {"error": str(e)})

@socketio.on("send_message")
def handle_message(data):
    """
    Handles incoming messages from the frontend.
    Expects:
    {
      "thread_id": "<uuid>", 
      "message": "<text>", 
      "image": "data:image/jpeg;base64,...." (optional),
      "location": {"latitude": 12.34, "longitude": 56.78} (optional),
      "input_method": "text" | "voice" | "image" (optional)
    }
    """
    thread_id = data.get("thread_id", str(uuid.uuid4()))
    message = data.get("message", "")
    image_data = data.get("image", None)
    user_location = data.get("location", None)
    input_method = data.get("input_method", "text")  # Default to text if not specified
    
    # DEBUG: Log what input method was received
    print(f"DEBUG: Received input_method = '{input_method}' for message: '{message[:50]}...'")
    
    # Ensure consistent timestamping
    timestamp = datetime.utcnow().isoformat()

    # Save user message
    try:
        save_message(thread_id, "user", message, user_location)
    except Exception:
        # Non-fatal; continue
        pass

    # Emit back user message (so UI shows it immediately)
    if message.strip() and image_data:
        # Both text and image
        message_content = message
        message_data = {"role": "user", "content": message_content, "timestamp": timestamp, "image": image_data}
    elif image_data:
        # Image only
        message_data = {"role": "user", "content": "📷 Uploaded prescription image", "timestamp": timestamp, "image": image_data}
    else:
        # Text only
        message_data = {"role": "user", "content": message, "timestamp": timestamp}
    
    emit("message_received", message_data)

    # PROCESS: image and/or text
    try:
        conversation_history = load_messages(thread_id)
        raw_data = None
        assistant_response = ""

        if image_data:
            # Extract base64 portion
            image_base64 = image_data.split(",", 1)[1] if "," in image_data else image_data
            prescription_data = analyze_prescription_image(image_base64)
            
            if prescription_data.get("medicines"):
                meds = prescription_data["medicines"]
                med_list = ", ".join(meds)
                
                # If user also sent a text message, process it with prescription context
                if message.strip():
                    # Create enhanced context with prescription info
                    prescription_context = f"[Prescription Analysis: Found medicines: {med_list}]"
                    enhanced_message = f"{message}\n\n{prescription_context}"
                    
                    # Process the user's text instruction with prescription context
                    assistant_response, raw_data = asyncio.run(
                        process_query(enhanced_message, None, conversation_history, user_location)
                    )
                else:
                    # No text message, use default prescription response
                    assistant_response = (
                        f"I've analyzed your prescription and found **{med_list}**. "
                        "How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. "
                        "Always consult your doctor for medical advice.\n\n"
                        f"[Context: Prescription contains: {med_list}]"
                    )
                
                # Persist prescription analysis as assistant message
                raw_data = json.dumps(prescription_data)
            else:
                assistant_response = "I couldn't identify medicines in this image. Please upload a clearer prescription image."
        else:
            # Text only - process normally
            assistant_response, raw_data = asyncio.run(
                process_query(message, None, conversation_history, user_location)
            )

        # Save assistant response
        try:
            save_message(thread_id, "assistant", assistant_response, None)
        except Exception:
            pass

        # Stream the response in chunks instead of sending all at once
        if assistant_response:
            stream_response(assistant_response, thread_id, raw_data, user_location, conversation_history, input_method)

    except Exception as e:
        # Log error server-side and inform user politely
        import traceback
        traceback.print_exc()
        emit("message_received", {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}", "timestamp": datetime.utcnow().isoformat()})

def stream_response(assistant_response, thread_id, raw_data, user_location, conversation_history, input_method="text"):
    """Stream the assistant response with conditional AWS Polly voice synthesis"""
    import time
    
    # Start streaming signal
    emit("response_start", {"thread_id": thread_id})
    
    # Generate audio with AWS Polly ONLY if user used voice input
    if polly_client and input_method == "voice":
        try:
            # Clean text for Polly (remove markdown)
            clean_text = assistant_response.replace('**', '').replace('*', '').replace('`', '')
            
            polly_response = polly_client.synthesize_speech(
                Text=clean_text,
                OutputFormat='mp3',
                VoiceId='Joanna',
                Engine='neural'
            )
            
            # Convert audio to base64 for transmission
            audio_data = polly_response['AudioStream'].read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # Send audio to frontend
            emit("polly_audio_ready", {
                "thread_id": thread_id,
                "audio_data": audio_base64
            })
            
            print(f"AWS Polly audio generated for voice input: {len(audio_data)} bytes")
            
        except Exception as e:
            print(f"AWS Polly error: {e}")
            # Fallback to browser TTS will be handled by frontend
    elif input_method == "voice":
        print(f"Voice input detected but Polly unavailable, frontend will use browser TTS")
    else:
        print(f"{input_method.title()} input - no voice synthesis needed")
    
    # Stream text chunks for visual effect
    words = assistant_response.split()
    chunk_size = 2
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if i + chunk_size < len(words):
            chunk += " "
        
        emit("response_chunk", {
            "thread_id": thread_id,
            "chunk": chunk,
            "is_final": False
        })
        
        time.sleep(0.3)
    
    # Send completion signal
    emit("response_complete", {
        "thread_id": thread_id,
        "full_content": assistant_response,
        "timestamp": datetime.utcnow().isoformat()
    })
    
    # Handle map data and title generation (same as before)
    if raw_data:
        try:
            parsed = json.loads(raw_data)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "latitude" in parsed[0]:
                emit("show_map", {"stores": parsed, "user_location": user_location})
        except Exception as e:
            print(f"📡 DEBUG - Error parsing raw_data: {e}")
    
    message_count = len(conversation_history)
    if message_count <= 1 or message_count % 3 == 0:
        try:
            recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
            title = generate_title(recent_messages)
            update_thread_title(thread_id, title)
            emit("title_updated", {"thread_id": thread_id, "title": title})
        except Exception as e:
            print(f"❌ Title generation failed: {e}")

# -------------------------
# CLI starter  
# -------------------------
@app.route("/api/route/<float:user_lat>/<float:user_lon>/<float:store_lat>/<float:store_lon>")
def get_route(user_lat, user_lon, store_lat, store_lon):
    """Get driving route between user and store"""
    try:
        import requests
        
        # Get route from OSRM
        route_url = f"http://router.project-osrm.org/route/v1/driving/{user_lon},{user_lat};{store_lon},{store_lat}?overview=full&geometries=geojson"
        response = requests.get(route_url, timeout=10)
        
        if response.status_code == 200:
            route_data = response.json()
            if route_data.get('routes'):
                coords = route_data['routes'][0]['geometry']['coordinates']
                route_coords = [[c[1], c[0]] for c in coords]  # Convert [lon,lat] to [lat,lon]
                
                distance = route_data['routes'][0]['distance'] / 1000  # km
                duration = route_data['routes'][0]['duration'] / 60    # minutes
                
                return {
                    "success": True,
                    "route": route_coords,
                    "distance": round(distance, 1),
                    "duration": round(duration)
                }
        
        return {"success": False, "error": "No route found"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.route("/api/map/<float:lat>/<float:lon>")
def get_folium_map(lat, lon):
    """Generate simple Folium map HTML for popup"""
    try:
        import folium
        
        # Create simple map
        m = folium.Map(location=[lat, lon], zoom_start=13)
        
        # Add user marker with different style
        folium.Marker(
            [lat, lon], 
            popup="📍 Your Location",
            icon=folium.Icon(color='red', icon='user', prefix='fa')
        ).add_to(m)
        
        # Get nearby stores
        result = asyncio.run(call_mcp_tool("medical-map", "get_nearby_stores", {
            "latitude": lat, 
            "longitude": lon, 
            "limit": 5
        }))
        
        # Add store markers with detailed popups
        if result and result.startswith('['):
            stores = json.loads(result)
            for store in stores:
                popup_html = f"""
                <div style="min-width: 200px;">
                    <h4 style="margin: 0 0 8px 0; color: #1e293b;">{store['store_name']}</h4>
                    <p style="margin: 4px 0; color: #64748b; font-size: 12px;">
                        📍 {store.get('address', 'Address not available')}
                    </p>
                    <p style="margin: 4px 0; color: #64748b; font-size: 12px;">
                        📞 {store.get('phone_number', 'Phone not available')}
                    </p>
                    <p style="margin: 4px 0; color: #2563eb; font-weight: 600; font-size: 12px;">
                        📏 {store['distance']:.2f} km away
                    </p>
                </div>
                """
                
                folium.Marker(
                    [store['latitude'], store['longitude']], 
                    popup=folium.Popup(popup_html, max_width=250),
                    tooltip=store['store_name'],
                    icon=folium.Icon(color='blue', icon='plus', prefix='fa')
                ).add_to(m)
        
        return m._repr_html_()
    except Exception as e:
        return f"<div>Error loading map: {str(e)}</div>"

# -------------------------
# Authentication Routes
# -------------------------
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        full_name = data.get('full_name')
        phone = data.get('phone')
        
        # Validation
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Check if user exists
        existing_user = User.find_by_email(email)
        if existing_user:
            return jsonify({'error': 'User already exists'}), 400
        
        # Create new user
        user = User(email=email, full_name=full_name, phone=phone)
        user.password_hash = User.hash_password(password)
        
        if user.save():
            # Create access token
            access_token = create_access_token(identity=user.user_id)
            return jsonify({
                'message': 'User registered successfully',
                'access_token': access_token,
                'user': user.to_dict()
            }), 201
        else:
            return jsonify({'error': 'Registration failed'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.find_by_email(email)
        if not user or not User.check_password(password, user.password_hash):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Create access token
        access_token = create_access_token(identity=user.user_id)
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/profile', methods=['GET'])
@jwt_required()
def get_profile():
    """Get user profile"""
    try:
        user_id = get_jwt_identity()
        user = User.find_by_id(user_id)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({'user': user.to_dict()}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Cart Management Routes
# -------------------------
@app.route('/api/cart', methods=['GET'])
@jwt_required()
def get_cart():
    """Get user's cart"""
    try:
        user_id = get_jwt_identity()
        cart = Cart(user_id)
        items = cart.get_items()
        
        return jsonify({
            'items': items,
            'total_amount': cart.total_amount,
            'item_count': cart.get_item_count()
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cart/add', methods=['POST'])
@jwt_required()
def add_to_cart_api():
    """Add item to cart"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        # Handle both ID-based and name-based requests
        medicine_id = data.get('medicine_id')
        store_id = data.get('store_id')
        medicine_name = data.get('medicine_name')
        store_name = data.get('store_name')
        quantity = data.get('quantity', 1)
        unit_price = data.get('unit_price')
        
        # If IDs not provided, look them up by names
        if not medicine_id and medicine_name:
            result = asyncio.run(call_mcp_tool("medical-database", "execute_sql", {
                "sql_query": f"SELECT medicine_id FROM medicines WHERE medicine_name LIKE '%{medicine_name}%' LIMIT 1"
            }))
            if result and result.startswith('['):
                medicines = json.loads(result)
                if medicines:
                    medicine_id = medicines[0]['medicine_id']
        
        if not store_id and store_name:
            result = asyncio.run(call_mcp_tool("medical-database", "execute_sql", {
                "sql_query": f"SELECT store_id FROM medical_stores WHERE store_name LIKE '%{store_name}%' LIMIT 1"
            }))
            if result and result.startswith('['):
                stores = json.loads(result)
                if stores:
                    store_id = stores[0]['store_id']
        
        if not all([medicine_id, store_id, unit_price]):
            return jsonify({'error': 'Missing required fields or could not find medicine/store'}), 400
        
        cart = Cart(user_id)
        if cart.add_item(medicine_id, store_id, quantity, unit_price):
            return jsonify({'message': 'Item added to cart'}), 200
        else:
            return jsonify({'error': 'Failed to add item'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cart/update', methods=['PUT'])
@jwt_required()
def update_cart_item():
    """Update cart item quantity"""
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        quantity = data.get('quantity')
        
        if not item_id or quantity is None:
            return jsonify({'error': 'Item ID and quantity required'}), 400
        
        user_id = get_jwt_identity()
        cart = Cart(user_id)
        
        if cart.update_quantity(item_id, quantity):
            return jsonify({'message': 'Cart updated'}), 200
        else:
            return jsonify({'error': 'Failed to update cart'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/cart/remove', methods=['DELETE'])
@jwt_required()
def remove_from_cart():
    """Remove item from cart"""
    try:
        data = request.get_json()
        item_id = data.get('item_id')
        
        if not item_id:
            return jsonify({'error': 'Item ID required'}), 400
        
        user_id = get_jwt_identity()
        cart = Cart(user_id)
        
        if cart.remove_item(item_id):
            return jsonify({'message': 'Item removed'}), 200
        else:
            return jsonify({'error': 'Failed to remove item'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Payment Routes
# -------------------------
@app.route('/api/payment/create-order', methods=['POST'])
@jwt_required()
def create_payment_order():
    """Create Razorpay order for payment"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        amount = data.get('amount')  # Amount in rupees
        address_data = data.get('address')
        
        if not amount or not address_data:
            return jsonify({'error': 'Amount and address required'}), 400
        
        # Create Razorpay order
        razorpay_order = razorpay_client.order.create({
            'amount': int(amount * 100),  # Amount in paise
            'currency': 'INR',
            'payment_capture': 1
        })
        
        # Save address to database (simplified for now)
        # In production, you'd save to user_addresses table
        
        return jsonify({
            'success': True,
            'order_id': razorpay_order['id'],
            'amount': amount,
            'currency': 'INR',
            'key_id': os.getenv("RAZORPAY_KEY_ID")
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/payment/verify', methods=['POST'])
@jwt_required()
def verify_payment():
    """Verify Razorpay payment and create order"""
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        
        razorpay_order_id = data.get('razorpay_order_id')
        razorpay_payment_id = data.get('razorpay_payment_id')
        razorpay_signature = data.get('razorpay_signature')
        address_data = data.get('address')
        
        # Verify payment signature
        params_dict = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id,
            'razorpay_signature': razorpay_signature
        }
        
        razorpay_client.utility.verify_payment_signature(params_dict)
        
        # Payment verified successfully
        # Get cart items before clearing
        cart = Cart(user_id)
        cart_items = cart.get_items()
        
        if not cart_items:
            return jsonify({'error': 'Cart is empty'}), 400
        
        # Create order using Order model
        order = Order(user_id)
        payment_data = {
            'razorpay_order_id': razorpay_order_id,
            'razorpay_payment_id': razorpay_payment_id
        }
        
        if order.create_order(cart_items, address_data, payment_data):
            # Clear cart after successful order
            cart.clear_cart()
            
            return jsonify({
                'success': True,
                'message': 'Payment successful and order created',
                'order_id': order.order_id,
                'payment_id': razorpay_payment_id
            }), 200
        else:
            return jsonify({'error': 'Failed to create order'}), 500
        
    except razorpay.errors.SignatureVerificationError:
        return jsonify({'error': 'Payment verification failed'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Order Management Routes
# -------------------------
@app.route('/api/orders', methods=['GET'])
@jwt_required()
def get_user_orders():
    """Get user's order history"""
    try:
        user_id = get_jwt_identity()
        orders = Order.get_user_orders(user_id)
        return jsonify({'orders': orders}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/orders/<order_id>', methods=['GET'])
@jwt_required()
def get_order_details(order_id):
    """Get detailed order information"""
    try:
        user_id = get_jwt_identity()
        order_details = Order.get_order_details(order_id, user_id)
        
        if order_details:
            return jsonify({'order': order_details}), 200
        else:
            return jsonify({'error': 'Order not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------
# Enhanced SocketIO Handlers for E-commerce
# -------------------------
@socketio.on("add_to_cart")
def handle_add_to_cart_socket(data):
    """Add item to cart via SocketIO (for logged-in users)"""
    try:
        # For now, we'll use session-based cart for non-authenticated users
        # In production, you'd want to require authentication
        
        medicine_id = data.get("medicine_id")
        store_id = data.get("store_id")
        quantity = data.get("quantity", 1)
        unit_price = data.get("unit_price")
        medicine_name = data.get("medicine_name", "Unknown Medicine")
        store_name = data.get("store_name", "Unknown Store")
        
        # For demo purposes, we'll emit success
        # In production, integrate with user authentication
        emit("cart_updated", {
            "success": True,
            "message": f"Added {medicine_name} to cart",
            "item_count": 1  # This would be actual count from database
        })
        
    except Exception as e:
        emit("cart_updated", {
            "success": False,
            "error": str(e)
        })

if __name__ == "__main__":
    print("Starting MedAI on http://localhost:5000")
    # Disable debug mode to prevent constant restarts that break SocketIO sessions
    socketio.run(app, debug=False, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
