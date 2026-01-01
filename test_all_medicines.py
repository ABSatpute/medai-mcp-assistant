import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from app import process_query

async def test_different_medicines():
    print("=== TESTING ALTERNATIVES FOR DIFFERENT MEDICINES ===")
    
    # Test 1: Paracetamol alternatives
    print("\n1. Testing alternatives for Paracetamol:")
    conversation_history = [
        {"role": "assistant", "content": "I've analyzed your prescription and found **Paracetamol 500mg**. [Context: Prescription contains: Paracetamol 500mg]"}
    ]
    
    response, _ = await process_query(
        query="suggest alternatives for this medicine",
        conversation_history=conversation_history
    )
    print(f"Response: {response}")
    
    # Test 2: Aspirin alternatives  
    print("\n2. Testing alternatives for Aspirin:")
    conversation_history = [
        {"role": "assistant", "content": "I've analyzed your prescription and found **Aspirin 75mg**. [Context: Prescription contains: Aspirin 75mg]"}
    ]
    
    response, _ = await process_query(
        query="give me alternative medicines",
        conversation_history=conversation_history
    )
    print(f"Response: {response}")
    
    # Test 3: Insulin alternatives
    print("\n3. Testing alternatives for Insulin:")
    conversation_history = [
        {"role": "assistant", "content": "I've analyzed your prescription and found **Insulin Glargine**. [Context: Prescription contains: Insulin Glargine]"}
    ]
    
    response, _ = await process_query(
        query="what alternatives do you have",
        conversation_history=conversation_history
    )
    print(f"Response: {response}")
    
    # Test 4: Generic query without specific medicine
    print("\n4. Testing generic alternative request:")
    conversation_history = []
    
    response, _ = await process_query(
        query="suggest alternative medicines",
        conversation_history=conversation_history
    )
    print(f"Response: {response}")
    
    print("\n=== TEST COMPLETE ===")

if __name__ == "__main__":
    asyncio.run(test_different_medicines())
