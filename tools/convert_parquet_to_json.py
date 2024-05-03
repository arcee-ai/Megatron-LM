"""Convert HF Parquet file to Json file based on Megatron-LLM format."""
from datasets import load_dataset
import pandas as pd
import json

DATASET_NAME = "sec-data-mini"
# Load the dataset from Hugging Face
dataset = load_dataset(f"arcee-ai/{DATASET_NAME}")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset['train'])  # Assuming 'train' split, change accordingly

# Create a new dictionary with the desired key-value format
formatted_data = []
for new_sample in df["text"]:
    formatted_data.append({"text": new_sample})


# Save the dictionary as JSON
with open(f"../data/{DATASET_NAME}.json", "w") as json_file:
    for sample in formatted_data:
        json.dump(sample, json_file)
        json_file.write('\n')
