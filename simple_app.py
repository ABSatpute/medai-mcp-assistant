from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
import uuid
from datetime import datetime

load_dotenv()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'medai_secret_key_2024'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('send_message')
def handle_message(data):
    """Simple message handler with LLM"""
    thread_id = data.get('thread_id', str(uuid.uuid4()))
    message = data.get('message', '')
    
    print(f"Received message: {message}")
    
    # Echo user message back
    emit('message_received', {
        'role': 'user',
        'content': message,
        'timestamp': datetime.now().isoformat()
    })
    
    try:
        # Simple LLM response
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        system_prompt = """You are MedAI, a helpful medical assistant. 
        Provide helpful responses about health and medical topics.
        Keep responses conversational and helpful."""
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        
        response = llm.invoke(messages)
        
        # Send LLM response
        emit('message_received', {
            'role': 'assistant',
            'content': response.content,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"Sent response: {response.content[:50]}...")
        
    except Exception as e:
        print(f"Error: {e}")
        emit('message_received', {
            'role': 'assistant',
            'content': f"Sorry, I encountered an error: {str(e)}",
            'timestamp': datetime.now().isoformat()
        })

if __name__ == '__main__':
    print("Starting MedAI on http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
