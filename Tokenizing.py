from transformers import AutoTokenizer
# Run this file on the terminal


# Load tokenizer from the base model (same as used in training)
tokenizer = AutoTokenizer.from_pretrained("huggingFaceBaseModel") 

# Save it to your fine-tuned model's checkpoint folder
tokenizer.save_pretrained("/path/to/model") # Change this to your location of the fine tuned model to tokenize it.