import os
import torch
import pandas as pd
import evaluate
import bert_score
import json


from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
)


def evaluate_t5_model(
    model_dir: str,
    test_csv_path: str,
    batch_size: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    tokenizer = T5Tokenizer.from_pretrained(model_dir)

    df_test = pd.read_csv(test_csv_path)
    inputs = ["summarize: " + f for f in df_test["findings"].astype(str)]
    references = df_test["impression"].astype(str).tolist()

    enc = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    preds = []
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_mask[i : i + batch_size].to(device)

            outputs = model.generate(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend([d.strip() for d in decoded])

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=preds, references=references, use_stemmer=True
    )

    P, R, F1 = bert_score.score(
        preds, references, lang="en", device=device
    )

    print(f"\nT5 Federated Model Evaluation on TEST ({len(references)} examples)\n")
    for k, v in rouge_scores.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'BERT-P':10s}: {P.mean().item():.4f}")
    print(f"{'BERT-R':10s}: {R.mean().item():.4f}")
    print(f"{'BERT-F1':10s}: {F1.mean().item():.4f}")

    out = {
        "rouge": rouge_scores,
        "bertscore": {
            "precision": P.cpu().tolist(),
            "recall": R.cpu().tolist(),
            "f1": F1.cpu().tolist(),
        },
        "predictions": preds,
        "references": references,
    }
    save_path = os.path.join(model_dir, "t5_fed_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved T5 evaluation to {save_path}\n")


def evaluate_bart_model(
    model_dir: str,
    test_csv_path: str,
    batch_size: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
    tokenizer = BartTokenizer.from_pretrained(model_dir)

    df_test = pd.read_csv(test_csv_path)
    inputs = ["summarize: " + f for f in df_test["findings"].astype(str)]
    references = df_test["impression"].astype(str).tolist()

    enc = tokenizer(
        inputs,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    preds = []
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        for i in range(0, input_ids.size(0), batch_size):
            batch_ids = input_ids[i : i + batch_size].to(device)
            batch_mask = attention_mask[i : i + batch_size].to(device)

            outputs = model.generate(
                input_ids=batch_ids,
                attention_mask=batch_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            preds.extend([d.strip() for d in decoded])

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(
        predictions=preds, references=references, use_stemmer=True
    )

    P, R, F1 = bert_score.score(
        preds, references, lang="en", device=device
    )

    print(f"\nBART Federated Model Evaluation on TEST ({len(references)} examples)\n")
    for k, v in rouge_scores.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'BERT-P':10s}: {P.mean().item():.4f}")
    print(f"{'BERT-R':10s}: {R.mean().item():.4f}")
    print(f"{'BERT-F1':10s}: {F1.mean().item():.4f}")

    out = {
        "rouge": rouge_scores,
        "bertscore": {
            "precision": P.cpu().tolist(),
            "recall": R.cpu().tolist(),
            "f1": F1.cpu().tolist(),
        },
        "predictions": preds,
        "references": references,
    }
    save_path = os.path.join(model_dir, "bart_fed_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved BART evaluation to {save_path}\n")


if __name__ == "__main__":
    T5_MODEL_DIR  = "./multimodal_fed_model"    
    BART_MODEL_DIR= "./multimodal_bart_model"  
    TEST_CSV_PATH = "D:/BICS/BSP6/Dataset/splits/test.csv"

    evaluate_t5_model(T5_MODEL_DIR, TEST_CSV_PATH, batch_size=8)

    evaluate_bart_model(BART_MODEL_DIR, TEST_CSV_PATH, batch_size=8)
