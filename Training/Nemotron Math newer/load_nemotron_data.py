import pandas as pd
from datasets import load_dataset
import itertools
import ast
import json

def process_messages(messages):
    """
    Renames 'tool' role to 'thinking' and checks for existence of 'assistant' role.
    Returns processed list or None if no assistant found.
    """
    # handle case where CSV loads list as string
    if isinstance(messages, str):
        try:
            messages = ast.literal_eval(messages)
        except:
            return None
            
    has_assistant = False
    for msg in messages:
        # Rename 'tool' to 'thinking'
        if msg.get('role') == 'tool':
            msg['role'] = 'thinking'
        
        if msg.get('role') == 'assistant':
            has_assistant = True
            
    return messages if has_assistant else None

# 1. Load the dataset using streaming
print("Started streaming dataset...")
dataset = load_dataset("nvidia/Nemotron-SFT-Math-v3", streaming=True, split="train")

# 2. Extract the first 10,000 rows
print("Extracting first 10,000 rows...")
first_10k_rows = list(itertools.islice(dataset, 10000))

# 3. Convert to Pandas DataFrame
df = pd.DataFrame(first_10k_rows)

# 4. Process messages: rename 'tool' -> 'thinking' and filter out non-assistant rows
print("Processing roles and filtering rows...")
df['messages'] = df['messages'].apply(process_messages)
df = df.dropna(subset=['messages']).reset_index(drop=True)

# 5. Save to CSV
csv_filename = "nemotron_math_10k_processed.csv"
# We use json.dumps for the list of dicts to ensure it's a valid string for pandas to read later
df.to_csv(csv_filename, index=False)
print(f"Saved {len(df)} processed rows to {csv_filename}")

# 6. Load back using pandas
df_reloaded = pd.read_csv(csv_filename)

# Parse stringified lists back to objects for the reloaded DF
df_reloaded['messages'] = df_reloaded['messages'].apply(ast.literal_eval)

# Display verification
print("\nReloaded Data Summary:")
print(f"Total Rows: {len(df_reloaded)}")
if len(df_reloaded) > 0:
    print("\nExample processed message roles:")
    roles = [m['role'] for m in df_reloaded.iloc[0]['messages']]
    print(roles)
    print("\nFirst row snippet:")
    print(df_reloaded.head(1))
