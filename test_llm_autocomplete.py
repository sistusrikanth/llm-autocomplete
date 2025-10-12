#!/usr/bin/env python3
"""
Test script for the LLM-based autocomplete system.
"""

from llm_autocomplete import LLMAutocompleteModel
import time


def test_llm_autocomplete():
    """Test the LLM-based autocomplete functionality."""
    print("🧪 Testing LLM-Based Autocomplete System")
    print("=" * 50)
    
    try:
        # Initialize model
        print("🔄 Initializing LLM model...")
        model = LLMAutocompleteModel()
        
        # Display model info
        info = model.get_model_info()
        print(f"\n📊 Model Information:")
        for key, value in info.items():
            print(f"   {key}: {value}")
        print()
        
        # Test cases
        test_words = [
            "python",
            "machine", 
            "artificial",
            "learning",
            "data",
            "technology",
            "hello",
            "the"
        ]
        
        print("🔍 Testing predictions:")
        print("-" * 30)
        
        for word in test_words:
            print(f"\n📝 Testing word: '{word}'")
            start_time = time.time()
            
            predictions = model.predict_next_words(word, 3)
            
            end_time = time.time()
            generation_time = end_time - start_time
            
            if predictions:
                print(f"   ⏱️  Total generation time: {generation_time:.2f}s")
            else:
                print("   ❌ No predictions generated")
        
        print(f"\n✅ Testing completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\n💡 Troubleshooting tips:")
        print("   1. Make sure you have installed the dependencies:")
        print("      pip install torch transformers tokenizers accelerate")
        print("   2. Check your internet connection (models are downloaded)")
        print("   3. Ensure you have sufficient disk space for model files")


if __name__ == "__main__":
    test_llm_autocomplete()


