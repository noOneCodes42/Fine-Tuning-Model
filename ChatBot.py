#!/usr/bin/env python3
import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--message", required=True, help="The message to send to the chatbot.")
    parser.add_argument("--modelPath", required=True, help="Path to the model directory.")
    return parser.parse_args()

def load_model(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        device = "mps" if torch.backends.mps.is_available() else "cpu" # This is for macOS make sure to update this if you are not using MacOS
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        return tokenizer, model, device
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None

def generate_response(tokenizer, model, device, user_message):
    if tokenizer is None or model is None:
        return "Error: Model not loaded."

    try:
        encoded_input = tokenizer(user_message, return_tensors="pt", padding=False, truncation=True)
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)

        output_ids = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    max_length=input_ids.shape[1] + 100,
                                    num_return_sequences=1,
                                    pad_token_id=tokenizer.eos_token_id,
                                    temperature=0.7,
                                    top_k=50,
                                    top_p=0.9,
                                    do_sample=True,
                                    num_beams=5)

        if torch.isnan(output_ids).any() or torch.isinf(output_ids).any():
            return "Error: Invalid logits in the generated output."

        generated_ids = output_ids[0].cpu().numpy().tolist()
        generated_ids = [token for token in generated_ids if token != tokenizer.pad_token_id]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        if "### Response:" in full_output:
            parts = full_output.split("### Response:")
            last_response = parts[-1].strip()
            for marker in ["###", "Instruction:", "Input:"]:
                if marker in last_response:
                    last_response = last_response.split(marker)[0].strip()
            return f"{last_response}"

        return full_output

    except Exception as e:
        return f"Error during response generation: {e}"

def main():
    args = parse_args()
    tokenizer, model, device = load_model(args.modelPath)

    if tokenizer and model:
        response = generate_response(tokenizer, model, device, args.message)
        print(response)

if __name__ == "__main__":
    main()
