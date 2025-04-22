import os
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from transformers import AutoModelForCausalLM
from datasets import Dataset

# Load the preprocessed dataset
processed_data_path = 'processed_dataset.csv'
df = pd.read_csv(processed_data_path)

# Turn the Pandas df into a Dataset object
# Leave 100 examples out of the training set for testing
train_dataset = Dataset.from_pandas(df[100:])

# Create prompts for GRPO Training
def create_prompt(example):
    example["prompt"] = f"""All letters in the following message have been swapped to encrypt the message. Please determine which pairs of letters were swapped and give me the original unencrypted message. Do not output any explanation or additional text beyond the original message. Here is the message to solve with all of the letters swapped: {example["encrypted_text"]} Original: """
    return example
train_dataset = train_dataset.map(create_prompt)
print(train_dataset)

# Determine device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training will run on: {device}")

# Create a unique checkpoint directory for each run using a timestamp
run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
checkpoint_dir = f'/users/0/brogn002/{run}'
os.makedirs(checkpoint_dir, exist_ok=True)

from trl import GRPOConfig, GRPOTrainer
from rapidfuzz import fuzz

# Define a reward function to use
def reward(completions, **kwargs):
    """
    Computes a similarity score between two strings in the range [0,1].
    """
    original_texts = kwargs["text"]
    rewards = []
    for completion, text in zip(completions, original_texts):
      # Clean up completion to remove any potential formatting
      completion = completion.strip().lower()
      # Do not reward empty strings
      if len(completion) == 0:
            rewards.append(0.0)
            continue
      # Do not reward answers with non-ascii characters
      try:
          completion.encode('ascii')
      except UnicodeEncodeError:
          rewards.append(0.0)
          continue
      # Perfect match gets a full reward
      if completion == text:
          rewards.append(1.0)
          continue
      # Apply RapidFuzz ratio for all cases (handles different lengths well)
      similarity = fuzz.ratio(completion, text) / 100.0
#       # Add additional penalty for length mismatch
#       length_penalty = max(0, 1 - (abs(len(completion) - len(text)) / max(len(text), 1)))
#       # Combined score is a linear combination of similarity and length_penalty
#       final_score = (similarity * 0.5) + (length_penalty * 0.5)
#       rewards.append(final_score)
      rewards.append(similarity)
    return rewards

training_args = GRPOConfig(
    num_train_epochs=2,
    output_dir=checkpoint_dir,
    logging_steps=50,
    per_device_train_batch_size=4,  # Decrease this to lower vram usage
    num_generations=4,  # Decrease this to lower vram usage
    save_strategy="steps",          
    save_steps=500,  
    bf16=True,  # Enable bf16 mixed precision on A100 GPUs
    report_to="wandb",  # Turn on W&B loggin
)

# Path to the model directory
model_dir = '/users/0/brogn002/FOUR_LETTERS_2025-04-13_10-50-32/final_model'
print(os.listdir(model_dir)) # Verify that the folder exists

trainer = GRPOTrainer(
    model=model_dir,
    reward_funcs=reward,
    args=training_args,
    train_dataset=train_dataset,
)

# Train and save the final model
trainer.train()
trainer.save_model(os.path.join(checkpoint_dir, "final_model"))
