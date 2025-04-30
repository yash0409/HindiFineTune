from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

# Load the GPT-2 tokenizer and model
model_name = "microsoft/DialoGPT-medium"  # or "distilgpt2" for faster training
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# GPT-2 doesn't have a pad token by default, so we assign it to the EOS token
tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# Load Hinglish dataset
dataset = load_dataset('text', data_files={'train': 'dataset_2.json'})

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Create data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Set advanced training arguments
training_args = TrainingArguments(
    output_dir="./hinglish-model",
    overwrite_output_dir=True,
    num_train_epochs=10,  # Increased epochs
    per_device_train_batch_size=8,  # Increased batch size
    learning_rate=2e-5,  # Adjusted learning rate
    save_steps=500,
    save_total_limit=2,
    logging_steps=100,
    evaluation_strategy="steps",  # Evaluate the model every 500 steps
    push_to_hub=False,
    report_to=[],
)

# Initialize Trainer with adjusted parameters
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
)

# Train the model
trainer.train()

# Save the model and tokenizer
trainer.save_model("./hinglish-model")
tokenizer.save_pretrained("./hinglish-model")
