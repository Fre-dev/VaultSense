import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any

class BitNetService:
    def __init__(self, model_name: str = "gpt2"):
        self.model_name = model_name
        self.device = "cpu"
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise

    def generate_response(self, 
                         prompt: str, 
                         max_length: int = 100,
                         temperature: float = 0.7,
                         top_p: float = 0.9) -> str:
        """Generate response from the model"""
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            self.model.eval()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):]
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I'm sorry, I encountered an error while processing your request."

    def chat_with_context(self, 
                         query: str, 
                         context: List[Dict[str, Any]] = None,
                         max_length: int = 100) -> str:
        """Generate response using optional context"""
        if context:
            context_text = "\n".join([f"Context: {doc['text']}" for doc in context])
            prompt = f"""Based on the following context, please answer the question.
            
Context:
{context_text}

Question: {query}

Answer:"""
        else:
            prompt = query
        
        return self.generate_response(prompt, max_length=max_length)

def main():
    # Example usage
    bitnet = BitNetService()
    
    # Example query and context
    query = "What is the capital of France?"
    context = [
        {" "}
    ]
    
    response = bitnet.chat_with_context(query, context)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()