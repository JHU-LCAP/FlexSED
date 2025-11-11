import os
import pandas as pd
import librosa
from tqdm import tqdm

# Paths
csv_path = 'train_df_updated.csv'           # Input CSV with column 'filename'
audio_dir = '../16k/train/'           # Directory containing .wav files

# Load original CSV
df = pd.read_csv(csv_path)
print(df)

# Compute durations
durations = []
for fname in tqdm(df['file_name']):
    path = os.path.join(audio_dir, fname)
    try:
        dur = round(librosa.get_duration(filename=path), 3)
    except Exception as e:
        # print(f"[Error] {fname}: {e}")
        dur = 0.0
    durations.append(dur)

# Append duration column
df['duration'] = durations

print(len(df[df['duration'] == 0]))

# df = df[df['duration'] != 0]

df.to_csv("train_df_duration.csv", index=False)
