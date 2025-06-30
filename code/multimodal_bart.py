import os
import random
from pathlib import Path

import pandas as pd
import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    ViTModel,
    ViTFeatureExtractor,
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

DATA_DIR    = Path("D:/BICS/BSP6/Dataset")
REPORTS_CSV = DATA_DIR / "indiana_reports.csv"
PROJ_CSV    = DATA_DIR / "indiana_projections.csv"
IMG_DIR     = DATA_DIR / "images/images_normalized"
OUT_SPLITS  = DATA_DIR / "new_splits"
OUT_SPLITS.mkdir(exist_ok=True)

TRAIN_CSV = OUT_SPLITS / "train.csv"
VAL_CSV   = OUT_SPLITS / "val.csv"
TEST_CSV  = OUT_SPLITS / "test.csv"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class MultimodalOpenIDataset(Dataset):
    def __init__(self, csv_path, img_dir, feature_extractor, tokenizer,
                 max_len_text: int = 512, max_len_tgt: int = 128):
        self.img_dir      = Path(img_dir)
        self.feat_ex      = feature_extractor
        self.tok          = tokenizer
        self.max_len_text = max_len_text
        self.max_len_tgt  = max_len_tgt
        self.df           = pd.read_csv(csv_path)

        print(f"[dataset] loading {csv_path} → {len(self.df)} samples")
        print("  head:", self.df.head(2).to_dict(orient="records"), "\n")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img_path = self.img_dir / row.filename
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        pix = self.feat_ex(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        findings   = "" if pd.isna(row.findings) else str(row.findings)
        impression = "" if pd.isna(row.impression) else str(row.impression)
        txt = "summarize: " + findings

        enc = self.tok(
            txt,
            max_length=self.max_len_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        dec = self.tok(
            impression,
            max_length=self.max_len_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids      = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        labels         = dec.input_ids.squeeze(0)

        labels[labels == self.tok.pad_token_id] = -100

        if i < 2:
            print(f"[getitem {i}] uid={row.uid}, file={row.filename}")
            print("  text", txt[:60], "…")
            print("  tgt", impression[:60], "…\n")

        return {
            "pixel_values":   pix,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels
        }


VISION_MODEL = "google/vit-base-patch16-224-in21k"
TEXT_MODEL   = "facebook/bart-base"

tokenizer         = BartTokenizer.from_pretrained(TEXT_MODEL)
feature_extractor = ViTFeatureExtractor.from_pretrained(VISION_MODEL)

class VisionBartModel(torch.nn.Module):
    def __init__(self, vision_encoder: ViTModel, bart_model: BartForConditionalGeneration):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.bart_model     = bart_model
        self.proj = torch.nn.Linear(
            vision_encoder.config.hidden_size,
            bart_model.config.d_model
        )

    def forward(self, pixel_values=None, input_ids=None, attention_mask=None, labels=None):
        enc_out = self.vision_encoder(pixel_values=pixel_values)
        img_feats = enc_out.last_hidden_state                
        proj_feats = self.proj(img_feats)                   
        img_mean = proj_feats.mean(dim=1, keepdim=True)    

        img_mask = torch.ones(
            img_mean.size()[:2],
            dtype=torch.long,
            device=img_mean.device
        )

        if input_ids is not None:
            txt_embeds    = self.bart_model.get_input_embeddings()(input_ids)
            merged_embeds = torch.cat([img_mean, txt_embeds], dim=1)  
            merged_mask   = torch.cat([img_mask, attention_mask], dim=1)
        else:
            merged_embeds = img_mean
            merged_mask   = img_mask

        outputs = self.bart_model(
            inputs_embeds=merged_embeds,
            attention_mask=merged_mask,
            labels=labels
        )
        return outputs

vision_encoder = ViTModel.from_pretrained(VISION_MODEL)
bart_decoder   = BartForConditionalGeneration.from_pretrained(TEXT_MODEL)
model          = VisionBartModel(vision_encoder, bart_decoder)

model.config = bart_decoder.config
model.config.decoder_start_token_id = bart_decoder.config.bos_token_id
model.config.eos_token_id           = bart_decoder.config.eos_token_id
model.config.pad_token_id           = bart_decoder.config.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

batch_size = 4
train_ds = MultimodalOpenIDataset(TRAIN_CSV, IMG_DIR, feature_extractor, tokenizer)
val_ds   = MultimodalOpenIDataset(VAL_CSV,   IMG_DIR, feature_extractor, tokenizer)

training_args = Seq2SeqTrainingArguments(
    output_dir                   = "./multimodal_bart_model",
    per_device_train_batch_size  = batch_size,
    per_device_eval_batch_size   = batch_size,
    predict_with_generate        = True,
    num_train_epochs             = 2,
    logging_steps                = 50,
    do_train                     = True,
    do_eval                      = True,
    eval_steps                   = max(1, len(train_ds) // batch_size),
    save_strategy                = "no",            
    learning_rate                = 5e-5,
    fp16                         = torch.cuda.is_available(),
    remove_unused_columns        = False,
    report_to                    = ["none"],
)

trainer = Seq2SeqTrainer(
    model          = model,
    args           = training_args,
    train_dataset  = train_ds,
    eval_dataset   = val_ds,
    data_collator  = default_data_collator,
    tokenizer      = tokenizer      
)

trainer.train()

save_dir = Path("./multimodal_bart_model")
save_dir.mkdir(exist_ok=True)

model_to_save = model.bart_model
model_to_save.save_pretrained(save_dir)

tokenizer.save_pretrained(save_dir)
feature_extractor.save_pretrained(save_dir)

print("\nDone. Model saved to ./multimodal_bart_model")
