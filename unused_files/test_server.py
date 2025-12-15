from flask import Flask
from flask_socketio import SocketIO, emit
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'test_key'
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route('/')
def index():
    return '<h1>Test Server Running</h1><script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.0.1/socket.io.js"></script><script>const socket = io(); socket.emit("test", "hello");</script>'

@socketio.on('test')
def handle_test(data):
    print(f"Received: {data}")
    emit('response', f"Echo: {data}")

if __name__ == '__main__':
    print("Starting test server on http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True)
