# MedAI - Medical Assistant with MCP Integration

A production-ready medical AI assistant built with Streamlit, LangGraph, and Model Context Protocol (MCP) for intelligent medicine search, store location, and health queries.

## Features

- 🏥 **Medical Q&A**: AI-powered health and medicine information
- 💊 **Medicine Search**: Find medicines with prices and availability
- 🗺️ **Store Locator**: Find nearest medical stores with live location detection
- 📍 **Distance Calculation**: Real-time distance calculation using Haversine formula
- 🎯 **Route Planning**: Get directions to selected stores
- 💬 **Chat History**: Persistent conversation threads with SQLite
- 🔧 **Multi-Step Queries**: Complex queries with tool chaining
- 📊 **Data Visualization**: Tables and interactive maps

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/medical_app_langgraph.git
cd medical_app_langgraph

# Create virtual environment
python -m venv medlgenv
medlgenv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo OPENAI_API_KEY=your_key_here > .env
```

## Usage

```bash
streamlit run mcp_app.py
```

## Configuration

Update default location in `mcp_app.py`:
```python
user_location = {"latitude": 18.566039, "longitude": 73.766370}
```

## License

MIT License
