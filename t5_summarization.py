"""
Simple T5-based Text Summarization Model
Uses T5-small for efficient text summarization
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time


class T5Summarizer:
    """A lightweight T5-based text summarization model."""
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize the T5 summarization model.
        
        Args:
            model_name: Hugging Face model name (default: t5-small)
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🔄 Loading T5 model: {model_name}")
        print(f"📱 Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model = self.model.to(self.device)
        
        print("✅ Model loaded successfully!")
    
    def summarize(self, text: str, max_length: int = 150, min_length: int = 40) -> str:
        """
        Summarize the input text.
        
        Args:
            text: Input text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            Summarized text
        """
        # T5 requires a task prefix
        input_text = f"summarize: {text}"
        
        # Tokenize input
        inputs = self.tokenizer.encode(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        # Generate summary
        start_time = time.time()
        
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs,
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
        
        # Decode the summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        inference_time = time.time() - start_time
        
        return summary, inference_time


def main():
    """Main function to test T5 summarization."""
    print("🤖 T5-Based Text Summarization")
    print("=" * 70)
    
    # Initialize the summarizer
    summarizer = T5Summarizer()
    
    # Test paragraph
    test_text = """
    Artificial intelligence has transformed the landscape of modern technology, 
    enabling machines to perform tasks that once required human intelligence. 
    Machine learning, a subset of AI, allows computers to learn from data and 
    improve their performance over time without being explicitly programmed. 
    Deep learning, which uses neural networks with multiple layers, has achieved 
    remarkable success in areas such as image recognition, natural language processing, 
    and speech recognition. These advancements have led to practical applications 
    including virtual assistants, autonomous vehicles, medical diagnosis systems, 
    and personalized recommendation engines. As AI continues to evolve, it raises 
    important questions about ethics, privacy, and the future of work in an 
    increasingly automated world.
    """
    
    print("\n📝 Original Text:")
    print("-" * 70)
    print(test_text.strip())
    print(f"\nOriginal length: {len(test_text.split())} words")
    
    # Generate summary
    print("\n🔄 Generating summary...")
    summary, inference_time = summarizer.summarize(test_text)
    
    print("\n📊 Summary:")
    print("-" * 70)
    print(summary)
    print(f"\nSummary length: {len(summary.split())} words")
    print(f"⏱️  Inference time: {inference_time:.2f}s")
    print(f"📉 Compression ratio: {len(test_text.split()) / len(summary.split()):.2f}x")
    
    # Test with different length constraints
    print("\n\n🔬 Testing with different length constraints:")
    print("=" * 70)
    
    for max_len, min_len in [(50, 20), (100, 40), (200, 60)]:
        print(f"\n📏 Max length: {max_len}, Min length: {min_len}")
        summary, inf_time = summarizer.summarize(test_text, max_length=max_len, min_length=min_len)
        print(f"   Summary: {summary}")
        print(f"   Length: {len(summary.split())} words | Time: {inf_time:.2f}s")


if __name__ == "__main__":
    main()

