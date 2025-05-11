# small_script_to_read_npy.py
import numpy as np

def read_and_print_npy(file_path, num_values_to_print=100):
    try:
        data = np.load(file_path)
        print(f"Successfully loaded {file_path}")
        print(f"Shape of the array: {data.shape}")
        print(f"Data type: {data.dtype}")
        
        # Assuming the logits are in a shape like (1, 1, vocab_size) or similar
        # We want to print the actual logit values.
        flat_data = data.flatten()
        
        print(f"Total number of values: {flat_data.size}")
        
        print(f"First {min(num_values_to_print, flat_data.size)} values:")
        for i in range(min(num_values_to_print, flat_data.size)):
            print(f"{flat_data[i]:.8f}", end=", ")
            if (i + 1) % 10 == 0:
                print("") # Newline every 10 values
        print("\n")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Specifically load the logits file we are interested in
    file_to_read = "python_logits_for_5th_token.npy" 
    read_and_print_npy(file_to_read, num_values_to_print=100) 