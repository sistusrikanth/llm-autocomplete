import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Tuple
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class LLMAutocompleteModel:
    """A lightweight LLM-based autocomplete model."""
    
    def __init__(self, model_name: str = "distilgpt2"):
        """
        Initialize the LLM model.
        
        Args:
            model_name: Hugging Face model name. Using DistilGPT-2 as it's lightweight
                       and excellent for text completion tasks.
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        self.generator = None
        
        print(f"🔄 Loading LLM model: {model_name}")
        print(f"📱 Using device: {self.device}")
        
        self._load_model()
    
    def _load_model(self):
        """Load the model and tokenizer."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Move model to device if not using device_map
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_full_text=False
            )
            
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔄 Falling back to a simpler approach...")
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Load a fallback model if the primary one fails."""
        try:
            # Try a smaller model as fallback
            fallback_model = "distilgpt2"
            print(f"🔄 Loading fallback model: {fallback_model}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model)
            self.model = AutoModelForCausalLM.from_pretrained(fallback_model)
            
            if self.device == "cpu":
                self.model = self.model.to(self.device)
            
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_full_text=False
            )
            
            print("✅ Fallback model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading fallback model: {e}")
            raise RuntimeError("Could not load any LLM model")
    
    def predict_next_words(self, word: str, num_predictions: int = 3) -> List[Tuple[str, float]]:
        """
        Predict the next words using the LLM.
        
        Args:
            word: Input word to predict from
            num_predictions: Number of predictions to return
            
        Returns:
            List of (word, confidence) tuples
        """
        if not self.generator:
            return []
        
        try:
            # Prepare the input text
            input_text = word.strip()
            
            # Generate text
            results = self.generator(
                input_text,
                max_new_tokens=5,  # Generate up to 5 tokens
                num_return_sequences=num_predictions,
                temperature=0.8,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True,
                top_p=0.9,
                top_k=50
            )
            
            predictions = []
            for i, result in enumerate(results):
                generated_text = result['generated_text'].strip()
                
                # Extract the first few words from generated text
                words = generated_text.split()
                if words:
                    # Take the first word as the prediction
                    predicted_word = words[0].strip('.,!?;:"')
                    # Filter out the original word
                    if predicted_word and predicted_word.lower() not in word.lower().split():
                        # Calculate a simple confidence score
                        confidence = 1.0 - (i * 0.1)  # Decreasing confidence
                        predictions.append((predicted_word, confidence))
            
            # If we don't have enough unique predictions, generate more
            if len(predictions) < num_predictions:
                additional_results = self.generator(
                    input_text,
                    max_new_tokens=5,
                    num_return_sequences=num_predictions * 2,
                    temperature=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    truncation=True,
                    top_p=0.95,
                    top_k=100
                )
                
                for result in additional_results:
                    generated_text = result['generated_text'].strip()
                    words = generated_text.split()
                    if words:
                        predicted_word = words[0].strip('.,!?;:"')
                        if (predicted_word and 
                            predicted_word.lower() not in word.lower().split() and
                            predicted_word not in [p[0] for p in predictions]):
                            confidence = 0.5 - (len(predictions) * 0.05)
                            predictions.append((predicted_word, confidence))
                            
                            if len(predictions) >= num_predictions:
                                break
            
            return predictions[:num_predictions]
            
        except Exception as e:
            print(f"❌ Error generating predictions: {e}")
            return []
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model is not None,
            'tokenizer_loaded': self.tokenizer is not None,
            'generator_loaded': self.generator is not None
        }


def main():
    """Main function to run the LLM-based autocomplete system."""
    print("🤖 LLM-Based Autocomplete System")
    print("=" * 50)
    
    # Initialize the LLM model
    try:
        model = LLMAutocompleteModel()
        
        # Display model information
        info = model.get_model_info()
        print(f"📊 Model Information:")
        print(f"   Model: {info['model_name']}")
        print(f"   Device: {info['device']}")
        print(f"   Status: {'✅ Ready' if info['generator_loaded'] else '❌ Not Ready'}")
        print()
        
        # Interactive loop
        print("💬 Enter a word to get LLM predictions (type 'quit' to exit):")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("Word: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not user_input:
                    print("Please enter a word.")
                    continue
                
                # Get predictions
                print("🔄 Generating predictions...")
                predictions = model.predict_next_words(user_input, 3)
                
                if predictions:
                    print(f"\n🔮 LLM predictions for '{user_input}':")
                    for i, (word, confidence) in enumerate(predictions, 1):
                        print(f"   {i}. '{word}' (confidence: {confidence:.3f})")
                else:
                    print(f"\n❌ No predictions generated for '{user_input}'")
                
                print()
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    except Exception as e:
        print(f"❌ Failed to initialize LLM model: {e}")
        print("💡 Make sure you have installed the required dependencies:")
        print("   pip install torch transformers tokenizers accelerate")


if __name__ == "__main__":
    main()
