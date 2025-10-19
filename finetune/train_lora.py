import os
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # instruction-tuned version
DATA_PATH = "data/mental_health_qa.jsonl"
OUTPUT_DIR = "finetune/adapter"
BATCH_SIZE = 1
LR = 1e-4
EPOCHS = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype="auto")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

raw_data = load_jsonl(DATA_PATH)

dataset = [{"text": f"Question: {d['instruction']}\nAnswer: {d['output']}"} for d in raw_data]

from datasets import Dataset
hf_dataset = Dataset.from_list(dataset)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

tokenized = hf_dataset.map(tokenize, batched=True, remove_columns=["text"])

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none"
)

lora_model = get_peft_model(model, peft_config)
lora_model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LR,
    fp16=False,
    logging_steps=1,
    logging_strategy="steps",
    disable_tqdm=False,
    save_total_limit=1,
    save_strategy="epoch",
    report_to="none",
    logging_dir=os.path.join(OUTPUT_DIR, "logs"),
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=lora_model,
    args=training_args,
    train_dataset=tokenized,
    data_collator=data_collator,
)

for step, batch in enumerate(trainer.get_train_dataloader()):
    if step % 5 == 0:
        print(f"🧠 Step {step}/{len(trainer.get_train_dataloader())}")
    if step > 10:
        break

try:
    trainer.train()
except KeyboardInterrupt:
    print("\n🛑 Training interrupted by user. Saving current LoRA weights...\n")
    lora_model.save_pretrained(OUTPUT_DIR)

print("\n💾 Saving LoRA adapter to:", OUTPUT_DIR)
lora_model.save_pretrained(OUTPUT_DIR)
print("\n✅ Done! Adapter ready for use.\n")