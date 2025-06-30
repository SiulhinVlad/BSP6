import torch
from transformers import (
    T5Tokenizer, 
    T5ForConditionalGeneration,
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import pandas as pd
import evaluate
import bert_score
import numpy as np
import json 

model_dir = "./t5_openi_model"
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir).to(
    "cuda" if torch.cuda.is_available() else "cpu"
)

def preprocess(batch):
    inputs = tokenizer(
        batch['findings'], 
        max_length=512, 
        padding="max_length", 
        truncation=True
    )
    targets = tokenizer(
        batch['impression'], 
        max_length=128, 
        padding="max_length", 
        truncation=True
    )
    inputs['labels'] = targets['input_ids']
    return inputs

test_dataset = Dataset.from_pandas(pd.read_csv("D:/BICS/BSP6/Dataset/splits/test.csv")).map(preprocess, batched=True)

rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]  
    
    if preds.ndim == 3:
        preds = np.argmax(preds, axis=-1)
    
    preds_text = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    preds_text = [text.strip() for text in preds_text]
    labels_text = [text.strip() for text in labels_text]
    
    rouge_scores = rouge.compute(
        predictions=preds_text,
        references=labels_text,
        use_stemmer=True
    )
    
    bert_P, bert_R, bert_F1 = bert_score.score(
        preds_text, 
        labels_text,
        lang="en",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    return {
        **rouge_scores,
        "bert_score_precision": bert_P.mean().item(),
        "bert_score_recall": bert_R.mean().item(),
        "bert_score_f1": bert_F1.mean().item(),
        "predictions": preds_text,  
        "references": labels_text    
    }

trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=4,
        do_train=False,
        do_eval=True
    ),
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    compute_metrics=compute_metrics
)

def generate_and_compute(dataset, desc="test"):
    print(f"Generating predictions for {desc} set...")
    predictions = trainer.predict(dataset)
    metrics = compute_metrics((predictions.predictions, predictions.label_ids))
    
    print(f"\n{desc.capitalize()} Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
    
    with open(f"{desc}_results.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved {desc} results to {desc}_results.json")
    
    return metrics

test_metrics = generate_and_compute(test_dataset, "test")