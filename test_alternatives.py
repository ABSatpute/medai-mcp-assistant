import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_query

async def test_alternatives():
    print("=== TESTING ALTERNATIVE MEDICINE FLOW ===")
    
    # Simulate conversation with Remdesivir prescription
    conversation_history = [
        {
            "role": "assistant", 
            "content": "I've analyzed your prescription and found **Remdesivir**. How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. Always consult your doctor for medical advice.\n\n[Context: Prescription contains: Remdesivir]"
        },
        {
            "role": "user",
            "content": "check this is available or not"
        },
        {
            "role": "assistant", 
            "content": "I'm sorry, but Remdesivir is currently not available in our inventory. However, I can help you with some alternatives: Let me search for generic versions we carry. We have similar medicines that might work for your needs. I can check if we can order Remdesivir for you. Please let me know how you would like to proceed! Always consult your doctor for medical advice."
        }
    ]
    
    # Test alternative request
    print("Testing: 'suggest alternatives'")
    response, raw_data = await process_query(
        query="suggest alternatives",
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    
    print(f"Response: {response}")
    print(f"Raw Data: {raw_data}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_alternatives())
