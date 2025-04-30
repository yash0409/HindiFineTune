import openai, os

openai.api_key = "Enter your API key here"

# Upload the file first
file = openai.files.create(
    file=open("dataset.jsonl", "rb"),
    purpose="fine-tune"
)

# Start fine-tuning
fine_tune_job = openai.fine_tuning.jobs.create(
    training_file=file.id,
    model="gpt-3.5-turbo",
    hyperparameters={"n_epochs": 2}
)

print(fine_tune_job)

