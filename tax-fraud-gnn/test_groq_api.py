import os
from groq import Groq

# Get API key from environment variable (do not hardcode in production)
api_key = os.environ.get('GROQ_API_KEY')

if not api_key:
    print("ERROR: GROQ_API_KEY environment variable not set!")
    print("Please set the GROQ_API_KEY environment variable before running this test.")
    print("Example: set GROQ_API_KEY=your_actual_key_here")
    exit(1)

print("=" * 60)
print("Testing Groq API Integration")
print("=" * 60)

try:
    # Initialize Groq client
    client = Groq(api_key=api_key)
    print("✓ Groq client initialized successfully")
    
    # Test question about the dataset
    test_question = "What is the fraud rate in a tax fraud detection system with 1000 companies and 7 fraudulent cases?"
    
    print(f"\nTest Question: {test_question}")
    print("\nSending request to Groq API...")
    print("-" * 60)
    
    # Make API call
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful tax fraud detection analyst. Answer questions clearly and concisely."
            },
            {
                "role": "user",
                "content": test_question
            }
        ],
        temperature=0.2,
        max_tokens=300
    )
    
    # Get the answer
    answer = response.choices[0].message.content
    
    print("\nGroq API Response:")
    print("-" * 60)
    print(answer)
    print("-" * 60)
    
    print("\n✓ API test successful!")
    print(f"✓ Model used: {response.model}")
    
    # Safely access usage information
    usage = getattr(response, 'usage', None)
    if usage is not None:
        total_tokens = getattr(usage, 'total_tokens', None)
        if total_tokens is not None:
            print(f"✓ Tokens used: {total_tokens}")
        else:
            print("⚠ Usage information available but total_tokens not found")
    else:
        print("⚠ Usage information not available in response")
    
except Exception as e:
    print(f"\n✗ Error: {e}")
    print("\nAPI test failed!")

print("=" * 60)