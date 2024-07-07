import os
import tiktoken
import numpy as np

def process_file_in_chunks(input_file_path, train_output_path, val_output_path, chunk_size=500*1024*1024):  # 500 MB chunks
    enc = tiktoken.get_encoding("gpt2")
    file_size = os.path.getsize(input_file_path)
    split_point = int(file_size * 0.9)
    
    with open(input_file_path, 'r', encoding='utf-8') as f, \
         open(train_output_path, 'wb') as train_out, \
         open(val_output_path, 'wb') as val_out:
        
        bytes_read = 0
        train_tokens = 0
        val_tokens = 0
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            chunk_bytes = chunk.encode('utf-8')
            chunk_size = len(chunk_bytes)
            
            if bytes_read + chunk_size <= split_point:
                # This chunk belongs entirely to the training set
                ids = enc.encode_ordinary(chunk)
                train_tokens += len(ids)
                np.array(ids, dtype=np.uint16).tofile(train_out)
            elif bytes_read < split_point < bytes_read + chunk_size:
                # This chunk needs to be split between train and val
                split_index = split_point - bytes_read
                train_part = chunk[:split_index]
                val_part = chunk[split_index:]
                
                train_ids = enc.encode_ordinary(train_part)
                val_ids = enc.encode_ordinary(val_part)
                
                train_tokens += len(train_ids)
                val_tokens += len(val_ids)
                
                np.array(train_ids, dtype=np.uint16).tofile(train_out)
                np.array(val_ids, dtype=np.uint16).tofile(val_out)
            else:
                # This chunk belongs entirely to the validation set
                ids = enc.encode_ordinary(chunk)
                val_tokens += len(ids)
                np.array(ids, dtype=np.uint16).tofile(val_out)
            
            bytes_read += chunk_size
    
    return train_tokens, val_tokens

# Paths
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')
train_output_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_output_path = os.path.join(os.path.dirname(__file__), 'val.bin')

if not os.path.exists(input_file_path):
    print("Training data not found")
    exit()

# Process the file
train_tokens, val_tokens = process_file_in_chunks(input_file_path, train_output_path, val_output_path)

print(f"Train has {train_tokens:,} tokens")
print(f"Val has {val_tokens:,} tokens")

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
