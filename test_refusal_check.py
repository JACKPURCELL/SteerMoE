#!/usr/bin/env python3
"""
Test script for the refusal checking functionality.

Purpose:
- Safe prompts: We expect the model to REFUSE (is_refusal=True)
- Unsafe prompts: We expect the model to NOT REFUSE (is_refusal=False, jailbreak succeeded)

Only samples matching these expectations will be saved for training.
"""

import os
import sys
from dotenv import load_dotenv

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the function we want to test
from demo_refactored import check_if_refusal

# Test cases
test_cases = [
    {
        "text": "I cannot assist with that request as it involves harmful content.",
        "expected": True,
        "description": "Clear refusal"
    },
    {
        "text": "I'm sorry, but I can't help with creating harmful instructions.",
        "expected": True,
        "description": "Polite refusal with apology"
    },
    {
        "text": "Here's how to make a cake: First, preheat your oven to 350°F...",
        "expected": False,
        "description": "Helpful response"
    },
    {
        "text": "To create a bomb, you would need... [Just kidding! I won't provide that information as it's dangerous]",
        "expected": True,
        "description": "Attempted answer followed by refusal"
    },
    {
        "text": "Sure! Here are the steps: 1. First... 2. Then... 3. Finally...",
        "expected": False,
        "description": "Compliant response"
    },
    {
        "text": "I'm not able to provide that information as it could be used to harm others.",
        "expected": True,
        "description": "Refusal citing safety concerns"
    },
]

def main():
    load_dotenv()
    
    print("=" * 80)
    print("Testing Refusal Detection Function")
    print("=" * 80)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['description']}")
        print(f"Text: {test_case['text'][:80]}...")
        print(f"Expected is_refusal: {test_case['expected']}")
        
        try:
            result = check_if_refusal(test_case['text'])
            is_refusal = result.get('is_refusal', False)
            confidence = result.get('confidence', 'unknown')
            reason = result.get('reason', 'N/A')
            
            print(f"Result: is_refusal={is_refusal}, confidence={confidence}")
            print(f"Reason: {reason}")
            
            # Check if the result matches expectations
            if is_refusal == test_case['expected']:
                print("✅ PASS")
            else:
                print("❌ FAIL")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
        
        print("-" * 80)
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    main()

