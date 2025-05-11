import numpy as np

# Define the paths to the NPY files relative to the Spark-TTS directory
logits_path = "python_logits_for_2nd_gen_token.npy"

# Token IDs of interest
python_picked_token_id = 158619  # The token ID Python picked as the 2nd new token
csharp_picked_token_id = 161183  # The token ID C# picked as the 2nd new token

try:
    print(f"Attempting to load logits from: {logits_path}")
    python_logits_for_2nd_token = np.load(logits_path)
    print(f"Successfully loaded logits. Shape: {python_logits_for_2nd_token.shape}, Dtype: {python_logits_for_2nd_token.dtype}")

    # Get the specific logits
    # Ensure IDs are within bounds before indexing
    if 0 <= python_picked_token_id < len(python_logits_for_2nd_token):
        logit_for_python_pick = python_logits_for_2nd_token[python_picked_token_id]
        print(f"Logit for Python's actual pick (ID {python_picked_token_id}): {logit_for_python_pick}")
    else:
        print(f"Error: Python's picked token ID {python_picked_token_id} is out of bounds for logits array.")

    if 0 <= csharp_picked_token_id < len(python_logits_for_2nd_token):
        logit_for_csharp_pick = python_logits_for_2nd_token[csharp_picked_token_id]
        print(f"Logit for C#'s actual pick (ID {csharp_picked_token_id}): {logit_for_csharp_pick}")
    else:
        print(f"Error: C#'s picked token ID {csharp_picked_token_id} is out of bounds for logits array.")

    # Get Top-5 from Python's perspective
    top_n = 5
    # Argsort returns indices that would sort the array in ascending order.
    # We take the last top_n indices for descending order.
    top_indices = np.argsort(python_logits_for_2nd_token)[-top_n:][::-1] # [::-1] to reverse to descending
    top_values = python_logits_for_2nd_token[top_indices]

    print(f"\nTop-{top_n} tokens and logits from Python's perspective (for 2nd token decision):")
    for i in range(top_n):
        print(f"  Rank {i+1}: ID = {top_indices[i]}, Logit = {top_values[i]}")

except FileNotFoundError:
    print(f"ERROR: Could not find the NPY file at {logits_path}. Please ensure it was generated correctly.")
except Exception as e:
    print(f"An error occurred: {e}")
