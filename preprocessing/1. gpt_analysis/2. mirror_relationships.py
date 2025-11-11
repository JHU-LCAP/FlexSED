import pandas as pd
import re
import numpy as np


def extract_category(rel):
    try:
        split = rel.split('.')
        if len(split) != 3:
            print(split)
            return np.nan
        else:
            return split[1]
    except:
        return np.nan


df = pd.read_csv('label_relationships.csv')

# Extract only the category letter (A, B, C, D)
df['Category'] = df['Relationship'].apply(extract_category)

# Drop original relationship text
df = df.drop(columns=["Relationship"])

# Mirror the entries
df_mirror = df.rename(columns={"Label 1": "Label 2", "Label 2": "Label 1"})

# Combine original and mirrored
df_combined = pd.concat([df, df_mirror], ignore_index=True)


# Optional: sort or remove duplicates if needed
df_combined = df_combined.sort_values(by=["Label 1", "Label 2"]).reset_index(drop=True)

df_combined.to_csv('gpt_relationships.csv', index=False)

# Display or export

nan_rows = df_combined['Category'].isna().sum()
print(f"Number of rows containing NaN: {nan_rows}")
