import os
import argparse
import json
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    TrainerCallback, TrainingArguments, TrainerState, TrainerControl
)
from datasets import Dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Hugging Face model repo or path")
    parser.add_argument("--data", required=True, help=".jsonl file with instruction, input, output")
    default_output_dir = os.path.join(os.path.expanduser("path"), "to", "storing", "finetuned model") # Change this to the file path you would want to store it in.
    parser.add_argument("--output_dir", default=default_output_dir, help="Where to save the model")
    return parser.parse_args()

def load_jsonl_dataset(jsonl_path):
    print(f"📁 Reading JSONL file: {jsonl_path}")
    with open(jsonl_path, 'r') as f:
        records = [json.loads(line.strip()) for line in f if line.strip()]
    print(f"✅ Loaded {len(records)} records.")
    for rec in records:
        rec['text'] = f"### Instruction:\n{rec['instruction']}\n### Input:\n{rec['input']}\n### Response:\n{rec['output']}"
    return Dataset.from_list(records)

def report_progress(current, total):
    percent = int((current / total) * 100)
    print(f"__PROGRESS__:{percent}")  # For SwiftUI to track
    return percent

class ProgressCallback(TrainerCallback):
    def __init__(self):
        self.current_step = 0
        self.total_steps = None

    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.total_steps = state.max_steps

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        self.current_step = state.global_step
        if self.total_steps is not None:
            report_progress(self.current_step, self.total_steps)

def main():
    args = parse_args()

    print(f"🧠 Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("✅ Tokenizer loaded.")

    # Handle padding token
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            print("Setting pad_token to eos_token.")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            print("Adding a new special token as pad_token: [PAD]")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    print("🧠 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    if tokenizer.pad_token is not None:
        model.resize_token_embeddings(len(tokenizer))
    print("✅ Model loaded.")

    print("📂 Loading dataset...")
    dataset = load_jsonl_dataset(args.data)

    print("🧪 Tokenizing data...")
    dataset = dataset.map(lambda e: tokenizer(e['text'], truncation=True, padding="max_length", max_length=512), batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    print("✅ Data ready.")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2, 
        logging_steps=100,
        save_steps=500,
        save_total_limit=1,
        report_to="tensorboard",
        disable_tqdm=False,
        evaluation_strategy="no",  # No eval dataset required
        logging_dir=os.path.join(os.path.expanduser("~"), "Documents", "logs"),  # ✅ Writable path: This is where the logs are you can change the location I wouldn't reccomend it though.
        dataloader_num_workers=4,
        run_name="fine_tuning_run",
)


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[ProgressCallback()]  # Instantiate and pass the callback
    )

    print("🚀 Starting fine-tuning...")
    trainer.train()
    print("🎉 Fine-tuning complete.")
    print(f"📦 Model saved to: {args.output_dir}")
    print("__PROGRESS__:100")

if __name__ == "__main__":
    main()