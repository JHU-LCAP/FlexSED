from openai import OpenAI
import pandas as pd
import itertools
import time
import os
from tqdm import tqdm

api_key = "your_api"
client = OpenAI(api_key=api_key)

# Load the shared labels (sample 20 for testing)
df = pd.read_csv("all_labels.csv")
labels = df["display_name"].tolist()

# Generate all possible label pairs
label_pairs = list(itertools.combinations(labels, 2))

# Output CSV file for saving results
OUTPUT_FILE = "label_relationships.csv"

# Load existing results if the file exists (to avoid reprocessing)
if os.path.exists(OUTPUT_FILE):
    existing_df = pd.read_csv(OUTPUT_FILE)
    # processed_pairs = set(zip(existing_df["Label 1"], existing_df["Label 2"]))
     # results = existing_df.to_dict("records")
    label_pairs = list(zip(existing_df["Label 1"], existing_df["Label 2"]))
    processed_pairs = set()
    results = []
else:
    processed_pairs = set()
    results = []

# Batch size (adjustable: 5 to 10 pairs per request)
BATCH_SIZE = 20


# Function to query GPT-4 for multiple label relationships
def get_relationships(batch_pairs):
    prompt = "For each of the following sound-related label pairs, determine their relationship:\n"
    for i, (label1, label2) in enumerate(batch_pairs, 1):
        prompt += f"{i}. {label1} - {label2}\n"

    prompt += """
    Classify each pair of labels into one of the following four categories. Use the definitions and examples below to guide your decision:

    A. Root-Parent:
        - One label represents a broad category (parent), and the other is a more specific instance or subtype (child).
        - Examples: (Dog, Animal), (Piano, Musical Instruments)

    B. Equivalent Sound Events:
        - The two labels refer to the same or very similar types of sound events.
        - They may differ slightly in timing, perspective, or description, but are essentially variations of the same event.
        - Examples: (Engine Idling, Engine Starting), (Car Door Opening, Car Door Closing)

    C. Different Sound Events in the Same Context:
        - The labels describe distinct sound events, but they typically occur in the same environment or scenario.
        - Examples: (Toothbrush, Tap), (Cooking, Boiling Water)
    
    D. Different Sound Events in Unrelated Contexts:
        - The labels describe different types of sound events that do not usually co-occur and belong to different environments or domains.
        - Examples: (Chainsaw, Violin), (Helicopter, Keyboard Typing)

    For each pair listed below, choose the most appropriate category. Respond in the following format:

    1. <Category>
    2. <Category>
    3. <Category>
    ...
    """

    while True:  # Retry mechanism for API failures
        try:
            response = client.chat.completions.create(
                model="your_llm",
                messages=[{"role": "user", "content": prompt}]
            )
            output = response.choices[0].message.content.strip()
            return output.split("\n")
        except:
            print(f"API Error. Retrying in 5 seconds...")
            time.sleep(5)


# Process each batch and store results
for i in tqdm(range(0, len(label_pairs), BATCH_SIZE)):
    batch = label_pairs[i : i + BATCH_SIZE]
    batch = [(l1, l2) for l1, l2 in batch if (l1, l2) not in processed_pairs]  # Skip processed pairs

    if not batch:  # Skip if all pairs in this batch are already processed
        continue

    classifications = get_relationships(batch)

    for (label1, label2), relationship in zip(batch, classifications):
        results.append({"Label 1": label1, "Label 2": label2, "Relationship": relationship})
        processed_pairs.add((label1, label2))

    # Save results to CSV after each batch
    pd.DataFrame(results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved progress: {len(results)} pairs processed so far.")
