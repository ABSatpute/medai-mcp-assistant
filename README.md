# MedAI - Medical Assistant with MCP Integration

A production-ready medical AI assistant built with Flask, SocketIO, LangGraph, and Model Context Protocol (MCP) for intelligent medicine search, store location, and health queries.

## Features

- 🏥 **Medical Q&A**: AI-powered health and medicine information
- 💊 **Smart Medicine Search**: Fuzzy matching with store-centric ranking
- 🗺️ **Store Locator**: Find nearest medical stores with live location detection
- 📍 **Distance Calculation**: Real-time distance calculation using Haversine formula
- 🎯 **Route Planning**: Get directions to selected stores via OSRM
- 💬 **Real-time Chat**: WebSocket-based messaging with persistent threads
- 🔧 **Multi-Step Queries**: Complex queries with tool chaining
- 📊 **Interactive Maps**: Folium-based store visualization
- 📋 **Prescription Analysis**: Image-based prescription processing

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/medical_app_langgraph.git
cd medical_app_langgraph

# Create virtual environment
python -m venv medlgenv
medlgenv\Scripts\activate  # Windows

# Install Flask dependencies
pip install -r requirements_flask.txt

# Create .env file
echo OPENAI_API_KEY=your_key_here > .env
```

## Usage

```bash
# Flask app (main implementation)
python app.py

# Alternative Streamlit version
streamlit run mcp_app.py
```

## Configuration

Update default location in `app.py`:
```python
user_location = {"latitude": 18.566039, "longitude": 73.766370}
```

## Architecture

### Backend
- **Flask** with **SocketIO** for real-time WebSocket communication
- **LangGraph** for AI workflow orchestration
- **SQLite** database for chat history persistence
- **MCP servers** for tool integration

### Key Components
- **Real-time Chat**: WebSocket-based messaging with auto-generated thread titles
- **Medicine Search**: Fuzzy matching with store-centric ranking algorithm
- **Location Services**: Haversine distance calculation and OSRM route planning
- **Interactive Maps**: Folium-based store visualization with detailed popups

### MCP Integration
- `medical-database` - SQL queries for medicines and stores
- `medical-map` - Location and mapping services
- `prescription-analyzer` - Image analysis for prescriptions

## License

MIT License
