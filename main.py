import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Create the new folder 'weights' in your current directory if it doesn't exist
output_dir = "./weights"
os.makedirs(output_dir, exist_ok=True)

# 2. Download and load the model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# 3. Export each parameter's weights as a flattened text file inside the folder
print("Saving model weights...")
for name, param in model.named_parameters():
    print(name, tuple(param.shape))
    values = param.detach().numpy().flatten()
    
    # Save directly into the ./weights/ directory
    file_path = os.path.join(output_dir, f"{name}.txt")
    with open(file_path, "w") as f:
        f.write(" ".join(str(v) for v in values))

# 4. Save tokenizer files into a dedicated subfolder within weights
print("\nSaving tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

print(f"\nAll files saved successfully inside: {os.path.abspath(output_dir)}")
