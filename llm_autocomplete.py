import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict, Tuple
import warnings
import os
import torch
import time
torch.manual_seed(int(time.time())) 
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
    
    def predict_next_words(self, word: str, num_predictions: int = 3) -> List[Tuple[str, float, float]]:
        """
        Predict the next words using the LLM by calling inference 3 times separately.
        Each call generates a different next token, and we compute the actual confidence
        from the model's output probabilities.
        
        Args:
            word: Input word to predict from
            num_predictions: Number of predictions to return (default 3)
            
        Returns:
            List of (word, confidence, inference_time) tuples sorted by decreasing confidence
        """
        if not self.model or not self.tokenizer:
            return []
        
        try:
            # Add context to guide the model
            context = "You are a next word predictor for special education kids who cant talk. Open with common sentence starter word."
            
            # Prepare the input text with context
            input_text = context + word.strip()
            
            # Tokenize the input
            inputs = self.tokenizer(input_text, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            predictions = []
            seen_tokens = set()
            
            # Call LLM inference num_predictions times separately
            for i in range(num_predictions):
                inference_start = time.time()
                with torch.no_grad():
                    # Generate one token at a time with sampling to get diversity
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,  # Generate only 1 token
                        do_sample=True,
                        temperature=0.7 + (i * 0.1),  # Increase temperature for more diversity
                        top_p=0.9,
                        top_k=50,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    
                    # Get the generated token
                    generated_token_id = outputs.sequences[0][-1].item()
                    
                    # Skip if we've already seen this token
                    if generated_token_id in seen_tokens:
                        # Retry with higher temperature
                        continue
                    
                    seen_tokens.add(generated_token_id)
                    
                    # Get the logits for the generated token
                    logits = outputs.scores[0][0]  # Shape: [vocab_size]
                    
                    # Convert logits to probabilities using softmax
                    probs = torch.nn.functional.softmax(logits, dim=-1)
                    
                    # Get the probability of the generated token
                    token_prob = probs[generated_token_id].item()
                    
                    # Decode the token to text
                    predicted_word = self.tokenizer.decode(generated_token_id, skip_special_tokens=True).strip()
                    
                    # Clean up the predicted word
                    predicted_word = predicted_word.strip('.,!?;:"')
                    
                    inference_time = time.time() - inference_start
                    
                    # Only add if it's a valid word and not in the input
                    if predicted_word and predicted_word.lower() not in word.lower().split():
                        predictions.append((predicted_word, token_prob, inference_time))
            
            # If we couldn't get enough unique predictions, try again with higher temperature
            retry_count = 0
            max_retries = num_predictions * 3
            
            while len(predictions) < num_predictions and retry_count < max_retries:
                retry_start = time.time()
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=1,
                        do_sample=True,
                        temperature=1.0 + (retry_count * 0.1),
                        top_p=0.95,
                        top_k=100,
                        pad_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    
                    generated_token_id = outputs.sequences[0][-1].item()
                    retry_time = time.time() - retry_start
                    
                    if generated_token_id not in seen_tokens:
                        seen_tokens.add(generated_token_id)
                        
                        logits = outputs.scores[0][0]
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        token_prob = probs[generated_token_id].item()
                        
                        predicted_word = self.tokenizer.decode(generated_token_id, skip_special_tokens=True).strip()
                        predicted_word = predicted_word.strip('.,!?;:"')
                        
                        if predicted_word and predicted_word.lower() not in word.lower().split():
                            if predicted_word not in [p[0] for p in predictions]:
                                predictions.append((predicted_word, token_prob, retry_time))
                    
                    retry_count += 1
            
            # Sort predictions by confidence (probability) in descending order
            predictions.sort(key=lambda x: x[1], reverse=True)
            
            print(f"\n📊 Final predictions (sorted by confidence):")
            for i, (word, conf, inf_time) in enumerate(predictions[:num_predictions], 1):
                print(f"   {i}. '{word}' - Confidence: {conf:.6f}, Inference Time: {inf_time:.4f}s")
            
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
                
                # if not user_input:
                #     print("Please enter a word.")
                #     continue
                
                # Get predictions
                print("🔄 Generating predictions...")
                predictions = model.predict_next_words(user_input, 3)
                
                if not predictions:
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
