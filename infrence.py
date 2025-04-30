from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./hinglish-model"  # Path where you saved your model
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Set model to evaluation mode
model.eval()

# Simple text generation function
def generate_text(prompt, max_length=50, temperature=0.6, top_p=0.9, top_k=50):
    # Tokenize input and generate attention mask
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]  # Explicitly get attention_mask

    with torch.no_grad():
       outputs = model.generate(
    input_ids,
    attention_mask=attention_mask,
    max_length=200,
    num_return_sequences=1,
    do_sample=True,
    temperature=0.7,  # More controlled output
    top_p=0.95,       # More diverse sampling
    top_k=30,         # Reduce the sampling space
    no_repeat_ngram_size=2
)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "Hi Aap Kaise Ho"
generated = generate_text(prompt)
print("Generated Text:\n", generated)
