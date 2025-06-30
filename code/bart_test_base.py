import torch
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration
)
from datasets import Dataset
import pandas as pd
import evaluate
import bert_score
import json

model_dir = "./bart_openi_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BartTokenizer.from_pretrained(model_dir)
model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
model.eval()

df = pd.read_csv("D:/BICS/BSP6/Dataset/splits/test.csv")
dataset = Dataset.from_pandas(df)

rouge = evaluate.load("rouge")

def generate_and_compute_manual(dataset, desc="test"):
    preds_text = []
    labels_text = []

    for example in dataset:
        input_ids = tokenizer.encode(
            "summarize: " + example["findings"],  
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)


        label_ids = tokenizer.encode(
            example["impression"],
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_length=128,
                num_beams=4,
                early_stopping=True
            )

        pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        label = tokenizer.decode(label_ids[0], skip_special_tokens=True).strip()

        preds_text.append(pred)
        labels_text.append(label)

    rouge_scores = rouge.compute(
        predictions=preds_text,
        references=labels_text,
        use_stemmer=True
    )

    bert_P, bert_R, bert_F1 = bert_score.score(
        preds_text,
        labels_text,
        lang="en",
        device=device.type
    )

    metrics = {
        **rouge_scores,
        "bert_score_precision": bert_P.mean().item(),
        "bert_score_recall": bert_R.mean().item(),
        "bert_score_f1": bert_F1.mean().item(),
        "predictions": preds_text,
        "references": labels_text
    }

    with open(f"{desc}_bart_results.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n{desc.capitalize()} Results:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")

    print(f"\nSaved {desc} results to {desc}_bart_results.json")
    return metrics

test_metrics = generate_and_compute_manual(dataset, "test")
