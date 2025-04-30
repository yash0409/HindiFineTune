# Hinglish Voice-AI: LLM Fine-Tuning Project

This project implements a fine-tuned language model for handling code-switched (Hinglish) dialogue in a Voice-AI system. The model is designed to understand and generate natural Hinglish conversations.

## Project Overview

This is a LLM Fine-Tuning Mini-Project focused on creating a Voice-AI that can handle Hinglish (Hindi-English code-switched) dialogue. The project includes dataset preparation, model fine-tuning, and inference capabilities.

## Project Structure

```
.
├── dataset_2.jsonl          # Training dataset with Hinglish examples
├── fine_tune.py          # Script for fine-tuning using OpenAI API (GPT-3.5)
├── fine_tune__gpt2.py    # Alternative script using GPT-2 (no API key required)
├── inference.py          # Script for generating responses using GPT-2
└── inference_gpt3turbo.py # Script for generating responses using GPT-3.5-turbo
```

## Implementation Options

This project provides two implementation approaches:

1. **GPT-3.5-turbo Implementation** (requires OpenAI API key)
   - Uses OpenAI's fine-tuning API
   - Better performance but requires paid API access
   - Implemented in `fine_tune.py` and `inference_gpt3turbo.py`

2. **GPT-2 Implementation** (no API key required)
   - Uses Microsoft's DialoGPT-medium model
   - Free to use, no API key needed
   - Implemented in `fine_tune__gpt2.py` and `inference.py`
   - **Why GPT-2?**: Initially planned to use GPT-3.5-turbo, but switched to GPT-2 due to:
     - No requirement for paid API access
     - Complete control over the model and training process
     - Ability to run locally without API dependencies

## Dataset Design

The dataset (`dataset.jsonl`) contains 10-20 carefully curated Hinglish conversation pairs in the following format:
```json
{"prompt":"User: Kaise ho?\nAssistant:","completion":"Main theek hoon, thanks for asking!"}
{"prompt":"User: Sunday ko kya plan hai?\nAssistant:","completion":"Shayad movie dekhne jaun."}
```

### Dataset Selection Rationale
- **Domain**: Everyday conversations and casual interactions
- **Style**: Natural code-switching between Hindi and English
- **Length**: Varied lengths to capture different conversation patterns
- **Examples**: Cover common scenarios like greetings, plans, and casual chat

## Setup

### Option 1: GPT-3.5-turbo Setup (requires API key)
1. Clone this repository
2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Install required packages:
```bash
pip install openai
```

### Option 2: GPT-2 Setup (no API key required)
1. Clone this repository
2. Install required packages:
```bash
pip install torch transformers datasets
```

## Fine-Tuning

### GPT-3.5-turbo Fine-Tuning
The fine-tuning script (`fine_tune.py`) uses the OpenAI API to train the model:

```python
import openai, os

openai.api_key = os.getenv("OPENAI_API_KEY")
resp = openai.FineTune.create(
    training_file="file-ID-you-upload",
    model="gpt-3.5-turbo",
    n_epochs=2
)
```

### GPT-2 Fine-Tuning
The alternative script (`fine_tune__gpt2.py`) uses Microsoft's DialoGPT-medium model:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "microsoft/DialoGPT-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
```

### Model and Hyperparameter Choices
#### GPT-3.5-turbo
- **Base Model**: GPT-3.5-turbo
  - Chosen for its balance of performance and cost
  - Better suited for dialogue tasks compared to davinci
- **Epochs**: 2
  - Sufficient for small dataset while preventing overfitting
- **Learning Rate**: Default
  - OpenAI's default learning rate is well-tuned for most cases

#### GPT-2 (DialoGPT-medium)
- **Base Model**: Microsoft DialoGPT-medium
  - Free to use, no API key required
  - Good performance for dialogue tasks
- **Training Parameters**:
  - Epochs: 10
  - Batch Size: 8
  - Learning Rate: 2e-5
  - Max Sequence Length: 128

## Inference

### GPT-3.5-turbo Inference
You can use either the OpenAI API directly or the provided `inference_gpt3turbo.py` script:

```python
# Using inference_gpt3turbo.py
from inference_gpt3turbo import generate_response

response = generate_response("Mujhe ek chai pilao.")
print(response)

# Or using OpenAI API directly
resp = openai.ChatCompletion.create(
    model="ft-yourmodel",
    messages=[{"role":"user","content":"Mujhe ek chai pilao."}]
)
```

### GPT-2 Inference
```python
from inference import generate_text

prompt = "Hi Aap Kaise Ho"
generated_text = generate_text(prompt)
print(generated_text)
```

### Generation Settings
- **Temperature**: 0.7
  - Balances creativity and coherence
- **Prompt Format**: User-Assistant format
  - Maintains conversation context
  - Clear role separation

## Quality Evaluation

The model's performance is evaluated through:
1. **Human Review**
   - Naturalness of code-switching
   - Contextual appropriateness
   - Grammatical correctness

2. **Automated Metrics**
   - Response coherence
   - Language mixing patterns
   - Response length appropriateness

## Sample Outputs

Example interactions with the fine-tuned model:
```
User: Kaise ho?
Assistant: Main theek hoon, thanks for asking! Aap kaise hain?

User: Weekend pe kya plan hai?
Assistant: Main soch raha hoon ki shopping karne jaun. Kuch new clothes ki zarurat hai.
```
