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
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

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
socketio = SocketIO(app, cors_allowed_origins="*")

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
        context = "\n".join([f"{m['role']}: {m['content'][:200]}" for m in recent])

    location_info = f"User Location: {user_location.get('latitude')}, {user_location.get('longitude')}" if user_location else "User location not available"

    system_prompt = f"""
You are MedAI, a medical store expert.

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

Return ONLY JSON when you want to call a tool, otherwise return a JSON with "use_tool": false and "answer": "<text>"

Example tool call responses:
{{"use_tool": true, "tool": "get_nearby_stores", "arguments": {{"latitude": 18.566039, "longitude": 73.766370, "limit": 5}}}}

{{"use_tool": true, "tool": "execute_sql", "arguments": {{"sql_query": "SELECT m.medicine_name, m.price, ms.store_name FROM medicines m JOIN store_stock ss ON m.medicine_id = ss.medicine_id JOIN medical_stores ms ON ss.store_id = ms.store_id WHERE m.medicine_name LIKE '%Crocin%' LIMIT 5"}}}}
"""

    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]
    response = llm.invoke(messages)

    cleaned = _clean_llm_json_response(response.content)
    try:
        decision = json.loads(cleaned)
        print(f"🤖 DEBUG - LLM Decision: {decision}")  # Debug what LLM decided
    except Exception:
        # LLM didn't return JSON -> treat as plain answer
        print(f"🤖 DEBUG - LLM returned plain text: {response.content[:100]}...")  # Debug non-JSON response
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
        print(f"🔍 DEBUG - MCP Tool Result: {result}")

        # Format for user
        format_prompt = f"""User asked: "{query}"
Database result: {result}

As a medical store expert, provide a natural, helpful response based on the user's question and the database results. Don't just show raw data - explain it conversationally. End with: 'Always consult your doctor for medical advice.'"""
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
      "location": {"latitude": 12.34, "longitude": 56.78} (optional)
    }
    """
    thread_id = data.get("thread_id", str(uuid.uuid4()))
    message = data.get("message", "")
    image_data = data.get("image", None)
    user_location = data.get("location", None)

    # Ensure consistent timestamping
    timestamp = datetime.utcnow().isoformat()

    # Save user message
    try:
        save_message(thread_id, "user", message, user_location)
    except Exception:
        # Non-fatal; continue
        pass

    # Emit back user message (so UI shows it immediately)
    emit("message_received", {"role": "user", "content": message or "📷 Uploaded prescription image", "timestamp": timestamp})

    # PROCESS: image or text
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
                assistant_response = (
                    f"I've analyzed your prescription and found **{med_list}**. "
                    "How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. "
                    "Always consult your doctor for medical advice."
                )
                # Persist prescription analysis as assistant message
                raw_data = json.dumps(prescription_data)
            else:
                assistant_response = "I couldn't identify medicines in this image. Please upload a clearer prescription image."
        else:
            # Run the async process_query synchronously here
            assistant_response, raw_data = asyncio.run(
                process_query(message, None, conversation_history, user_location)
            )

        # Save assistant response
        try:
            save_message(thread_id, "assistant", assistant_response, None)
        except Exception:
            pass

        # If raw_data looks like JSON list of stores -> emit show_map
        if raw_data:
            try:
                print(f"📡 DEBUG - Checking if raw_data is map data: {raw_data[:100]}...")
                parsed = json.loads(raw_data)
                if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict) and "latitude" in parsed[0]:
                    print(f"📡 DEBUG - Emitting show_map event with {len(parsed)} stores")
                    emit("show_map", {"stores": parsed, "user_location": user_location})
                else:
                    print(f"📡 DEBUG - Raw data is not store data format")
            except Exception as e:
                print(f"📡 DEBUG - Error parsing raw_data: {e}")
                # raw_data may be string or non-JSON - ignore silently
                pass

        # Emit assistant reply
        emit("message_received", {"role": "assistant", "content": assistant_response, "timestamp": datetime.utcnow().isoformat()})

        # Auto-generate title for new threads and update every 3 messages
        message_count = len(conversation_history)
        if message_count <= 1 or message_count % 3 == 0:
            try:
                # Use last 3 messages for better context (or all if less than 3)
                recent_messages = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
                title = generate_title(recent_messages)
                update_thread_title(thread_id, title)
                emit("title_updated", {"thread_id": thread_id, "title": title})
                print(f"📝 Title updated after {message_count} messages: {title}")
            except Exception as e:
                print(f"❌ Title generation failed: {e}")
                pass

    except Exception as e:
        # Log error server-side and inform user politely
        import traceback
        traceback.print_exc()
        emit("message_received", {"role": "assistant", "content": f"Sorry, I encountered an error: {str(e)}", "timestamp": datetime.utcnow().isoformat()})

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

if __name__ == "__main__":
    print("Starting MedAI on http://localhost:5000")
    socketio.run(app, debug=True, host="0.0.0.0", port=5000, allow_unsafe_werkzeug=True)
