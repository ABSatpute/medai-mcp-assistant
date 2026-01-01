import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_query

async def test_permission_flow():
    print("=== TESTING PERMISSION-BASED ALTERNATIVE FLOW ===")
    
    # Step 1: User asks for alternatives
    print("\n1. User asks for alternatives:")
    conversation_history = [
        {"role": "assistant", "content": "I've analyzed your prescription and found **Remdesivir**. [Context: Prescription contains: Remdesivir]"}
    ]
    
    response1, _ = await process_query(
        query="suggest alternatives",
        conversation_history=conversation_history
    )
    print(f"AI Response: {response1}")
    
    # Step 2: User gives permission to check
    print("\n2. User gives permission:")
    conversation_history.extend([
        {"role": "user", "content": "suggest alternatives"},
        {"role": "assistant", "content": response1}
    ])
    
    response2, _ = await process_query(
        query="yes, check availability",
        conversation_history=conversation_history
    )
    print(f"AI Response: {response2}")
    
    # Step 3: User asks for alternatives for different medicine
    print("\n3. Different medicine alternatives:")
    conversation_history = [
        {"role": "assistant", "content": "I've analyzed your prescription and found **Paracetamol 500mg**. [Context: Prescription contains: Paracetamol 500mg]"}
    ]
    
    response3, _ = await process_query(
        query="what alternatives do you have",
        conversation_history=conversation_history
    )
    print(f"AI Response: {response3}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_permission_flow())
