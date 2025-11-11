import pandas as pd
import ast
from collections import defaultdict
from tqdm import tqdm
tqdm.pandas()

# Load your data
rel_df = pd.read_csv('gpt_relationships.csv')  # the first table with Label 1, Label 2, Category
label_df = pd.read_csv('label_to_id.csv')   # the second table with label and id

# Build a label -> id mapping dictionary
label_to_id = dict(zip(label_df['label'], label_df['id']))

# Map Label 1 and Label 2 to their IDs
rel_df['Label 1 id'] = rel_df['Label 1'].map(label_to_id)
rel_df['Label 2 id'] = rel_df['Label 2'].map(label_to_id)

# Optional: reorder columns if you want
rel_df = rel_df[['Label 1', 'Label 1 id', 'Label 2', 'Label 2 id', 'Category']]

# Load your files
file_df = pd.read_csv('train_raw.csv')     # your file_name, pos_ids, neg_ids

# If pos_ids and neg_ids are strings, convert them to lists
file_df['neg_ids_ori'] = file_df['neg_ids']

# Filter only Category A relationships
rel_A = rel_df[rel_df['Category'] == ' A']
print(rel_A)
related_A = defaultdict(set)
for _, row in rel_A.iterrows():
    related_A[row['Label 1 id']].add(row['Label 2 id'])


def process_row(row):
    # Parse pos_ids and neg_ids if they are strings
    pos_ids = ast.literal_eval(row['pos_ids']) if isinstance(row['pos_ids'], str) else row['pos_ids']
    neg_ids = ast.literal_eval(row['neg_ids']) if isinstance(row['neg_ids'], str) else row['neg_ids']

    # Collect all related ids to remove
    to_remove = set()
    for pid in pos_ids:
        to_remove.update(related_A.get(pid, set()))

    # Find ids to be removed and update neg_ids
    removed_ids = list(set(neg_ids) & to_remove)
    new_neg_ids = list(set(neg_ids) - to_remove)

    return pd.Series([new_neg_ids, removed_ids])


# Apply
file_df[['neg_ids', 'removed_ids']] = file_df.progress_apply(process_row, axis=1)

# Save or view
file_df.to_csv('train_df_updated.csv', index=False)
