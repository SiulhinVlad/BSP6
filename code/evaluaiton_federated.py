import os
import json

import torch
import pandas as pd
import evaluate
import bert_score

from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

from transformers import (
    ViTModel,
    ViTFeatureExtractor,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    BartForConditionalGeneration,
    BartTokenizerFast,
)

class MultimodalTestDataset(Dataset):
    def __init__(self, csv_path: Path, img_dir: Path,
                 feature_extractor, tokenizer,
                 max_len_text: int = 512):

        self.img_dir = img_dir
        self.feat_ex = feature_extractor
        self.tok     = tokenizer
        self.max_len_text = max_len_text

        self.df = pd.read_csv(csv_path)
        self.df["findings"]   = self.df["findings"].fillna("")
        self.df["impression"] = self.df["impression"].fillna("")

        print(f"[multimodal test dataset] {csv_path} → {len(self.df)} samples")
        print("  head:", self.df.head(2).to_dict(orient="records"))
        print()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_file = row["filename"]
        img_path = self.img_dir / img_file
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        pixel_values = self.feat_ex(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        findings      = row["findings"]
        impression    = row["impression"]
        prompt        = "summarize: " + findings

        enc = self.tok(
            prompt,
            max_length=self.max_len_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "pixel_values":   pixel_values,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "reference":      impression,
        }


class VisionT5Model(torch.nn.Module):
    def __init__(self,
                 vision_encoder: ViTModel,
                 text_model: T5ForConditionalGeneration):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_model     = text_model
        self.proj = torch.nn.Linear(
            vision_encoder.config.hidden_size,
            text_model.config.d_model
        )

    def forward(self,
                pixel_values=None,
                input_ids=None,
                attention_mask=None,
                labels=None):
        enc_out = self.vision_encoder(pixel_values=pixel_values)
        img_feats = enc_out.last_hidden_state  

        img_feats = self.proj(img_feats) 

        img_token = img_feats.mean(dim=1, keepdim=True)

        img_attn_mask = torch.ones(
            (img_token.shape[0], 1),
            dtype=torch.long,
            device=img_token.device
        )

        if input_ids is not None:
            text_embeds = self.text_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([img_token, text_embeds], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([img_attn_mask, attention_mask], dim=1)
        else:
            inputs_embeds = img_token
            attention_mask = img_attn_mask

        out = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return out

    def generate(self,
                 pixel_values=None,
                 input_ids=None,
                 attention_mask=None,
                 **gen_kwargs):
        enc_out = self.vision_encoder(pixel_values=pixel_values)
        img_feats = enc_out.last_hidden_state
        img_feats = self.proj(img_feats).mean(dim=1, keepdim=True) 
        img_attn_mask = torch.ones((img_feats.shape[0], 1),
                                   dtype=torch.long,
                                   device=img_feats.device)

        if input_ids is not None:
            txt_embeds = self.text_model.get_input_embeddings()(input_ids)
            inputs_embeds = torch.cat([img_feats, txt_embeds], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([img_attn_mask, attention_mask], dim=1)
        else:
            inputs_embeds = img_feats
            attention_mask = img_attn_mask

        return self.text_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )


class VisionBartModel(torch.nn.Module):
    def __init__(self,
                 vision_encoder: ViTModel,
                 bart_model: BartForConditionalGeneration):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.bart_model     = bart_model
        self.proj = torch.nn.Linear(
            vision_encoder.config.hidden_size,
            bart_model.config.d_model
        )

    def forward(self,
                pixel_values=None,
                input_ids=None,
                attention_mask=None,
                labels=None):
        enc_out = self.vision_encoder(pixel_values=pixel_values)
        img_feats = enc_out.last_hidden_state  

        img_feats = self.proj(img_feats)  

        img_token = img_feats.mean(dim=1, keepdim=True)  

        img_attn = torch.ones((img_token.shape[0], 1),
                              dtype=torch.long,
                              device=img_token.device)

        if input_ids is not None:
            txt_embeds = self.bart_model.get_encoder().embed_tokens(input_ids)
            inputs_embeds = torch.cat([img_token, txt_embeds], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([img_attn, attention_mask], dim=1)
        else:
            inputs_embeds = img_token
            attention_mask = img_attn

        out = self.bart_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels
        )
        return out

    def generate(self,
                 pixel_values=None,
                 input_ids=None,
                 attention_mask=None,
                 **gen_kwargs):
        enc_out = self.vision_encoder(pixel_values=pixel_values)
        img_feats = enc_out.last_hidden_state
        img_feats = self.proj(img_feats).mean(dim=1, keepdim=True) 
        img_attn = torch.ones((img_feats.shape[0], 1),
                              dtype=torch.long,
                              device=img_feats.device)

        if input_ids is not None:
            txt_embeds = self.bart_model.get_encoder().embed_tokens(input_ids)
            inputs_embeds = torch.cat([img_feats, txt_embeds], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([img_attn, attention_mask], dim=1)
        else:
            inputs_embeds = img_feats
            attention_mask = img_attn

        return self.bart_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **gen_kwargs
        )


def evaluate_multimodal_t5(
    model_dir: Path,
    test_csv: Path,
    img_dir: Path,
    batch_size: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_encoder = ViTModel.from_pretrained(model_dir / "vision_encoder")
    text_decoder   = T5ForConditionalGeneration.from_pretrained(model_dir / "text_decoder")

    wrapper = VisionT5Model(vision_encoder, text_decoder).to(device)
    sd = torch.load(model_dir / "vision_t5_full_state.pt", map_location="cpu")
    wrapper.load_state_dict(sd["vision_t5_state_dict"])
    wrapper.proj.weight.data.copy_(torch.tensor(sd["proj_weight"]))
    wrapper.proj.bias.data.copy_(torch.tensor(sd["proj_bias"]))
    wrapper.to(device).eval()

    tokenizer = T5TokenizerFast.from_pretrained(model_dir / "tokenizer")
    feat_ex   = ViTFeatureExtractor.from_pretrained(model_dir / "feature_extractor")

    test_ds = MultimodalTestDataset(test_csv, img_dir, feat_ex, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    preds = []
    refs  = []
    with torch.no_grad():
        for batch in test_loader:
            pix = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask= batch["attention_mask"].to(device)

            outs = wrapper.generate(
                pixel_values=pix,
                input_ids=ids,
                attention_mask=mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
            preds.extend([d.strip() for d in dec])
            refs.extend(batch["reference"])

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    P, R, F1 = bert_score.score(preds, refs, lang="en", device=device)

    print(f"\n=== Vision→T5 Federated Evaluation ({len(refs)} examples) ===\n")
    for k, v in rouge_scores.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'BERT-P':10s}: {P.mean().item():.4f}")
    print(f"{'BERT-R':10s}: {R.mean().item():.4f}")
    print(f"{'BERT-F1':10s}: {F1.mean().item():.4f}")

    out = {
        "rouge": rouge_scores,
        "bertscore": {
            "precision": P.cpu().tolist(),
            "recall":    R.cpu().tolist(),
            "f1":        F1.cpu().tolist(),
        },
        "predictions": preds,
        "references":  refs,
    }
    save_path = model_dir / "t5_fed_eval_more_clients.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved Vision→T5 evaluation to {save_path}\n")


def evaluate_multimodal_bart(
    model_dir: Path,
    test_csv: Path,
    img_dir: Path,
    batch_size: int = 8,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vision_encoder = ViTModel.from_pretrained(model_dir / "vision_encoder")
    bart_decoder   = BartForConditionalGeneration.from_pretrained(model_dir / "bart_decoder")

    wrapper = VisionBartModel(vision_encoder, bart_decoder).to(device)
    sd = torch.load(model_dir / "vision_bart_full_state.pt", map_location="cpu")
    wrapper.load_state_dict(sd["vision_bart_state_dict"])
    wrapper.proj.weight.data.copy_(torch.tensor(sd["proj_weight"]))
    wrapper.proj.bias.data.copy_(torch.tensor(sd["proj_bias"]))
    wrapper.to(device).eval()

    tokenizer = BartTokenizerFast.from_pretrained(model_dir / "tokenizer")
    feat_ex   = ViTFeatureExtractor.from_pretrained(model_dir / "feature_extractor")

    test_ds = MultimodalTestDataset(test_csv, img_dir, feat_ex, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    preds = []
    refs  = []
    with torch.no_grad():
        for batch in test_loader:
            pix = batch["pixel_values"].to(device)
            ids = batch["input_ids"].to(device)
            mask= batch["attention_mask"].to(device)

            outs = wrapper.generate(
                pixel_values=pix,
                input_ids=ids,
                attention_mask=mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
            )
            dec = tokenizer.batch_decode(outs, skip_special_tokens=True)
            preds.extend([d.strip() for d in dec])
            refs.extend(batch["reference"])

    rouge = evaluate.load("rouge")
    rouge_scores = rouge.compute(predictions=preds, references=refs, use_stemmer=True)

    P, R, F1 = bert_score.score(preds, refs, lang="en", device=device)

    print(f"\n=== Vision→BART Federated Evaluation ({len(refs)} examples) ===\n")
    for k, v in rouge_scores.items():
        print(f"{k:10s}: {v:.4f}")
    print(f"{'BERT-P':10s}: {P.mean().item():.4f}")
    print(f"{'BERT-R':10s}: {R.mean().item():.4f}")
    print(f"{'BERT-F1':10s}: {F1.mean().item():.4f}")

    out = {
        "rouge": rouge_scores,
        "bertscore": {
            "precision": P.cpu().tolist(),
            "recall":    R.cpu().tolist(),
            "f1":        F1.cpu().tolist(),
        },
        "predictions": preds,
        "references":  refs,
    }
    save_path = model_dir / "bart_fed_eval_more_clients.json"
    with open(save_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ Saved Vision→BART evaluation to {save_path}\n")

if __name__ == "__main__":
    T5_MODEL_DIR   = Path("./output_multimodal_fed/federated_multimodal_t5_final_more_clients")
    BART_MODEL_DIR = Path("./output_multimodal_fed_bart/federated_multimodal_bart_final_more_clients")
    TEST_CSV       = Path("D:/BICS/BSP6/Dataset/new_splits/test.csv")
    IMG_DIR        = Path("D:/BICS/BSP6/Dataset/images/images_normalized")
    BATCH_SIZE     = 8

    evaluate_multimodal_t5(
        model_dir=T5_MODEL_DIR,
        test_csv=TEST_CSV,
        img_dir=IMG_DIR,
        batch_size=BATCH_SIZE
    )

    evaluate_multimodal_bart(
        model_dir=BART_MODEL_DIR,
        test_csv=TEST_CSV,
        img_dir=IMG_DIR,
        batch_size=BATCH_SIZE
    )