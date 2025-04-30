import openai

# Set your API key
openai.api_key = "Enter your API key here"

# Use your fine-tuned model ID here
fine_tuned_model = "ft:gpt-3.5-turbo-XYZ123"  # Replace with your actual fine-tuned model ID

# Function to get a response from the fine-tuned model
def chat_with_model(prompt):
    response = openai.ChatCompletion.create(
        model=fine_tuned_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,   # Controls randomness
        max_tokens=150     # Controls the length of the response
    )
    return response['choices'][0]['message']['content']

# Example usage
user_prompt = "Kaise ho?"
reply = chat_with_model(user_prompt)
print(f"Assistant: {reply}")
