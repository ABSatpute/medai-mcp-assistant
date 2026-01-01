import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_query

async def test_context_persistence():
    print("=== TESTING CONTEXT PERSISTENCE ===")
    
    # Simulate the conversation flow from your example
    conversation_history = [
        {
            "role": "assistant", 
            "content": "I've analyzed your prescription and found **Remdesivir**. How can I help you with these medicines? I can check availability, find nearby stores, or provide information about them. Always consult your doctor for medical advice.\n\n[Context: Prescription contains: Remdesivir]"
        }
    ]
    
    # Test 1: "check this is available or not"
    print("\n1. Testing: 'check this is available or not'")
    response1, _ = await process_query(
        query="check this is available or not",
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    print(f"Response: {response1}")
    
    # Add the response to conversation history
    conversation_history.append({"role": "user", "content": "check this is available or not"})
    conversation_history.append({"role": "assistant", "content": response1})
    
    # Test 2: "give me alternative"
    print("\n2. Testing: 'give me alternative'")
    response2, _ = await process_query(
        query="give me alternative",
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    print(f"Response: {response2}")
    
    # Add to history
    conversation_history.append({"role": "user", "content": "give me alternative"})
    conversation_history.append({"role": "assistant", "content": response2})
    
    # Test 3: "what is the name of medicine in prescription"
    print("\n3. Testing: 'what is the name of medicine in prescription'")
    response3, _ = await process_query(
        query="what is the name of medicine in prescription",
        conversation_history=conversation_history,
        user_location={"latitude": 18.566039, "longitude": 73.766370}
    )
    print(f"Response: {response3}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_context_persistence())
