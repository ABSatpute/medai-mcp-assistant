import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_query

async def test_context_awareness():
    print("=== TESTING CONTEXT AWARENESS ===")
    
    # Simulate conversation history with prescription
    conversation_history = [
        {
            "role": "assistant", 
            "content": "I've analyzed your prescription and found **Betaloc 100mg, Dorzolamidum 10 mg, Cimetidine 50 mg, Oxprelol 50mg**. How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. Always consult your doctor for medical advice.\n\n[Context: Prescription contains: Betaloc 100mg, Dorzolamidum 10 mg, Cimetidine 50 mg, Oxprelol 50mg]"
        }
    ]
    
    # Test query
    query = "is this available"
    
    print(f"Query: '{query}'")
    print(f"Context: {conversation_history}")
    
    # Call process_query to see what happens
    response, raw_data = await process_query(
        query=query,
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    
    print(f"\nAI Response: {response}")
    print(f"Raw Data: {raw_data}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_context_awareness())
