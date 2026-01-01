import base64
import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import analyze_prescription_image, process_query

def convert_image_to_base64(image_path):
    """Convert image file to base64 string"""
    try:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode('utf-8')
            return base64_string
    except Exception as e:
        print(f"Error converting image: {e}")
        return None

async def test_actual_prescription():
    print("=== TESTING WITH ACTUAL PRESCRIPTION IMAGE ===")
    
    # Convert your image to base64
    image_path = r"C:\Users\ADMIN\Downloads\97.jpg"
    print(f"Converting image: {image_path}")
    
    image_base64 = convert_image_to_base64(image_path)
    if not image_base64:
        print("Failed to convert image")
        return
    
    print("Image converted to base64")
    
    # Step 1: Analyze prescription
    print("\n1. Analyzing prescription...")
    prescription_data = analyze_prescription_image(image_base64)
    print(f"Prescription analysis result: {prescription_data}")
    
    if not prescription_data.get("medicines"):
        print("No medicines found in prescription")
        return
    
    # Step 2: Create conversation context
    medicines = prescription_data["medicines"]
    med_list = ", ".join(medicines)
    
    conversation_history = [
        {
            "role": "assistant", 
            "content": f"I've analyzed your prescription and found **{med_list}**. How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. Always consult your doctor for medical advice.\n\n[Context: Prescription contains: {med_list}]"
        }
    ]
    
    # Step 3: Test availability check
    print(f"\n2. Testing availability for: {med_list}")
    response1, _ = await process_query(
        query="check if these medicines are available",
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    print(f"Availability Response: {response1}")
    
    # Step 4: Test alternative suggestions
    conversation_history.extend([
        {"role": "user", "content": "check if these medicines are available"},
        {"role": "assistant", "content": response1}
    ])
    
    print(f"\n3. Testing alternative suggestions...")
    response2, _ = await process_query(
        query="suggest alternatives for all medicines",
        conversation_history=conversation_history
    )
    print(f"Alternative Response: {response2}")
    
    # Step 5: Test permission-based availability check
    conversation_history.extend([
        {"role": "user", "content": "suggest alternatives for all medicines"},
        {"role": "assistant", "content": response2}
    ])
    
    print(f"\n4. Testing permission-based availability check...")
    response3, _ = await process_query(
        query="yes, check availability of alternatives",
        conversation_history=conversation_history
    )
    print(f"Permission Response: {response3}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_actual_prescription())
