from flask import Flask, render_template, request, jsonify
import asyncio
import base64
import json
import uuid
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# Import existing database logic
from chat_db import init_db, save_message, load_messages, get_all_threads, update_thread_title, cleanup_old_threads, delete_thread

load_dotenv()
init_db()
cleanup_old_threads(10)

app = Flask(__name__)

def analyze_prescription_image(image_base64):
    """Direct prescription analysis"""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    messages = [
        HumanMessage(content=[
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
            {"type": "text", "text": "Extract medicines from this prescription. Return JSON: {\"medicines\": [\"med1\", \"med2\"]}"}
        ])
    ]
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        if content.startswith('```json'):
            content = content.replace('```json', '').replace('```', '').strip()
        return json.loads(content)
    except:
        return {"medicines": []}

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    thread_id = data.get('thread_id', str(uuid.uuid4()))
    message = data.get('message', '')
    image_data = data.get('image', None)
    
    # Save user message
    save_message(thread_id, 'user', message)
    
    # Get conversation context
    history = load_messages(thread_id)
    context = "\n".join([f"{h['role']}: {h['content'][:100]}" for h in history[-4:]])
    
    try:
        if image_data:
            image_base64 = image_data.split(',')[1] if ',' in image_data else image_data
            prescription_data = analyze_prescription_image(image_base64)
            
            if prescription_data.get("medicines"):
                medicines = ", ".join(prescription_data["medicines"])
                response = f"Found medicines: {medicines}. How can I help?"
            else:
                response = "Couldn't identify medicines. Please try a clearer image."
        else:
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            system_prompt = f"You are MedAI, a medical assistant. Context: {context}"
            
            ai_response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ])
            response = ai_response.content
        
        # Save AI response
        save_message(thread_id, 'assistant', response)
        
        return jsonify({
            'response': response,
            'thread_id': thread_id
        })
        
    except Exception as e:
        return jsonify({'response': f'Error: {str(e)}', 'thread_id': thread_id})

@app.route('/api/threads')
def get_threads():
    threads = get_all_threads()
    return jsonify([{'id': t[0], 'title': t[1]} for t in threads])

@app.route('/api/thread/<thread_id>')
def get_thread_messages(thread_id):
    messages = load_messages(thread_id)
    return jsonify(messages)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
