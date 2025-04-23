from services.bitnet_service import BitNetService

def test_chat():
    # Initialize the service
    bitnet = BitNetService()
    
    # Test queries
    test_queries = [
        
        "Tell me about artificial intelligence."
    ]
    
    # Run tests
    for i, query in enumerate(test_queries):
        print(f"\nTest {i+1}:")
        print(f"Query: {query}")
        
        response = bitnet.generate_response(query)
        print(f"\nResponse: {response}")
        print("-" * 80)

if __name__ == "__main__":
    test_chat() 