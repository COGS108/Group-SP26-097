import pandas as pd
import os
from pathlib import Path

DATASET_PATH = "../../data/00-raw/Kiss_Linnell_Dataset/"

exp_1_path = os.path.join(DATASET_PATH, "Experiment 1")
exp_1_raw = os.path.join(exp_1_path, "Raw data.xlsx")
exp_1_df = pd.read_excel(exp_1_raw) # THIS IS THE DATASET FOR EXP 1

exp_2_path = os.path.join(DATASET_PATH, "Experiment 2")
exp_2_raw = os.path.join(exp_2_path, "Raw data.xlsx")
exp_2_df = pd.read_excel(exp_2_raw) # THIS IS THE DATASET FOR EXP 2


# Add experiment indicator to each before concatenating
exp_1_df['experiment'] = 1
exp_2_df['experiment'] = 2

# Rename columns to match (exp_2 uses slightly different names)
exp_2_df = exp_2_df.rename(columns={
    'reaction time': 'Reaction time',
    'no_go': 'No-go trial',
    'thought_response': 'Thought probe ',
    'arousal': 'Arousal',
    'mood': 'Mood',
    'music-present': 'Music-present'
})

# Concatenate
combined_df = pd.concat([exp_1_df, exp_2_df], ignore_index=True)

# Save cleaned data for later use.
output_path = Path("../../data/01-interim/KL_Dataset_cleaned.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
combined_df.to_csv(output_path, index=False)
print(f"Saved cleaned dataset to {output_path}")