import json
import base64
from mcp.server import Server
from mcp.types import Tool, TextContent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Server("prescription-analyzer")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="extract_prescription_data",
            description="Extract complete prescription data from image including medicines, dosages, instructions, doctor info",
            inputSchema={
                "type": "object",
                "properties": {
                    "image_base64": {
                        "type": "string",
                        "description": "Base64 encoded prescription image"
                    }
                },
                "required": ["image_base64"]
            }
        ),
        Tool(
            name="get_medicine_info",
            description="Get information about medicines",
            inputSchema={
                "type": "object",
                "properties": {
                    "medicines": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of medicine names"
                    }
                },
                "required": ["medicines"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    if name == "extract_prescription_data":
        image_data = arguments["image_base64"]
        
        messages = [
            HumanMessage(content=[
                {"type": "text", "text": """Extract ALL visible text and data from this prescription image. Don't limit to predefined fields - capture EVERYTHING you can see.

Return as JSON with this flexible structure (include ANY fields you find):
{
  "doctor_info": {
    // Include ANY doctor-related info you see: name, title, specialization, clinic, hospital, address, phone, email, registration number, etc.
  },
  "patient_info": {
    // Include ANY patient info: name, age, gender, address, phone, patient ID, weight, height, etc.
  },
  "prescription_details": {
    "date": "Any date found",
    "prescription_number": "Prescription ID/number if present",
    "visit_type": "OPD/IPD/Emergency etc.",
    // Add any other prescription metadata
  },
  "medicines": [
    {
      // For each medicine, include ALL details you can see:
      // name, brand, generic, dosage, strength, quantity, frequency, duration, route, instructions, etc.
    }
  ],
  "medical_info": {
    "diagnosis": "Any diagnosis/condition mentioned",
    "symptoms": "Any symptoms listed", 
    "vital_signs": "BP, temperature, pulse, etc.",
    "allergies": "Any allergy information",
    "medical_history": "Any history mentioned"
  },
  "additional_data": {
    // Include ANY other text/data visible: lab results, follow-up dates, warnings, refills, signatures, stamps, etc.
  }
}

CRITICAL INSTRUCTIONS:
- Extract EVERY piece of text you can see
- Don't skip unclear or handwritten text - make your best attempt
- Include numbers, codes, stamps, signatures if visible
- Capture lab values, measurements, dates
- Include any printed or handwritten notes
- Don't limit to medical terms - extract everything

Return ONLY the JSON with all available data."""},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ])
        ]
        
        response = llm.invoke(messages)
        content = response.content.strip()
        
        # Extract JSON from response if it contains extra text
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        # Try to find JSON object in the response
        import re
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            content = match.group()
        
        # If medicines array is empty, try a simpler extraction
        try:
            parsed_data = json.loads(content)
            if not parsed_data.get('medicines') or len(parsed_data.get('medicines', [])) == 0:
                # Try simpler medicine extraction
                simple_messages = [
                    HumanMessage(content=[
                        {"type": "text", "text": "Look at this prescription image. List ONLY the medicine names you can see, even if handwritten or unclear. Return as simple JSON array: [\"medicine1\", \"medicine2\", ...]"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
                    ])
                ]
                
                simple_response = llm.invoke(simple_messages)
                simple_content = simple_response.content.strip()
                
                # Extract array from response
                if "[" in simple_content and "]" in simple_content:
                    array_match = re.search(r'\[.*\]', simple_content, re.DOTALL)
                    if array_match:
                        try:
                            medicine_names = json.loads(array_match.group())
                            # Convert to proper format
                            medicines = [{"name": name, "dosage": "Not specified", "frequency": "Not specified", "duration": "Not specified", "instructions": "Not specified"} for name in medicine_names]
                            parsed_data['medicines'] = medicines
                            content = json.dumps(parsed_data)
                        except:
                            pass
        except:
            pass
        
        return [TextContent(type="text", text=content)]
    
    elif name == "get_medicine_info":
        medicines = arguments["medicines"]
        prompt = f"Provide uses, symptoms treated, and safety notes for: {', '.join(medicines)}"
        response = llm.invoke([HumanMessage(content=prompt)])
        return [TextContent(type="text", text=response.content)]
    
    return [TextContent(type="text", text=f"Unknown tool: {name}")]

if __name__ == "__main__":
    import asyncio
    from mcp.server.stdio import stdio_server
    
    async def main():
        async with stdio_server() as (read_stream, write_stream):
            await app.run(read_stream, write_stream, app.create_initialization_options())
    
    asyncio.run(main())
