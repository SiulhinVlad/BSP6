import os
import pandas as pd
import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from sklearn.model_selection import train_test_split

split_dir = "D:/BICS/BSP6/Dataset/splits"
os.makedirs(split_dir, exist_ok=True)

train_path = os.path.join(split_dir, "train.csv")
val_path = os.path.join(split_dir, "val.csv")
test_path = os.path.join(split_dir, "test.csv")

def load_dataset(path):
    df = pd.read_csv(path)
    df = df[['findings', 'impression']].dropna()
    df = df[df['findings'].str.strip() != ""]
    df = df[df['impression'].str.strip() != ""]
    df['findings'] = df['findings'].apply(lambda x: "summarize: " + x)  
    return df

def preprocess(batch):
    inputs = tokenizer(batch['findings'], max_length=512, padding="max_length", truncation=True)
    targets = tokenizer(batch['impression'], max_length=128, padding="max_length", truncation=True)
    inputs['labels'] = targets['input_ids']
    return inputs

if all(os.path.exists(p) for p in [train_path, val_path, test_path]):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
else:
    df = load_dataset("D:/BICS/BSP6/Dataset/indiana_reports.csv")
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

model_name = "facebook/bart-base"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir="./bart_openi_model",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available()
)

trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()

trainer.save_model("./bart_openi_model")
tokenizer.save_pretrained("./bart_openi_model")
