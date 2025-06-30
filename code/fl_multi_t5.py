import os
import random
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from transformers import (
    VisionEncoderDecoderModel,
    ViTModel,
    ViTFeatureExtractor,
    T5TokenizerFast,
    T5ForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
)

SEED         = 42
MODEL_NAME   = "t5-base"
VISION_NAME  = "google/vit-base-patch16-224-in21k"

BASE_DIR     = Path("D:/BICS/BSP6/Dataset")
SPLIT_DIR    = BASE_DIR / "new_splits"                  
IMG_DIR      = BASE_DIR / "images/images_normalized" 
OUTPUT_ROOT  = Path("./output_multimodal_fed")      

NUM_CLIENTS  = 7
FED_ROUNDS   = 5
LOCAL_EPOCHS = 1
TRAIN_BS     = 1     
LR           = 5e-5

OUTPUT_ROOT.mkdir(exist_ok=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)

class MultimodalOpenIDataset(Dataset):
    def __init__(self, csv_path, img_dir, feature_extractor, tokenizer,
                 max_len_text=512, max_len_tgt=128):
        self.img_dir      = Path(img_dir)
        self.feat_ex      = feature_extractor
        self.tok          = tokenizer
        self.max_len_text = max_len_text
        self.max_len_tgt  = max_len_tgt

        self.df = pd.read_csv(csv_path)
        print(f"[dataset] loading {csv_path} → {len(self.df)} samples")
        print("  head:", self.df.head(3).to_dict(orient="records"))
        print()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        img_file = row["filename"]
        img_path = self.img_dir / img_file
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = Image.open(img_path).convert("RGB")
        pix = self.feat_ex(images=img, return_tensors="pt")["pixel_values"].squeeze(0)

        findings_txt = "" if pd.isna(row["findings"]) else str(row["findings"])
        impression_txt = "" if pd.isna(row["impression"]) else str(row["impression"])
        txt = "summarize: " + findings_txt
        tgt = impression_txt

        enc = self.tok(
            txt,
            max_length=self.max_len_text,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        dec = self.tok(
            tgt,
            max_length=self.max_len_tgt,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids      = enc.input_ids.squeeze(0)
        attention_mask = enc.attention_mask.squeeze(0)
        labels         = dec.input_ids.squeeze(0)
        labels[labels == self.tok.pad_token_id] = -100

        if i < 2:
            print(f"[getitem {i}] uid={row['uid']}, file={row['filename']}")
            print("  text", txt[:60], "…")
            print("  tgt", tgt[:60], "…\n")

        return {
            "pixel_values":   pix,
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_extractor = ViTFeatureExtractor.from_pretrained(VISION_NAME)
tokenizer         = T5TokenizerFast.from_pretrained(MODEL_NAME)

vision_encoder = ViTModel.from_pretrained(VISION_NAME)
text_decoder   = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

class VisionT5Model(torch.nn.Module):
    def __init__(self, vision_encoder: ViTModel, text_model: T5ForConditionalGeneration):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_model     = text_model
        self.proj = torch.nn.Linear(
            vision_encoder.config.hidden_size,
            text_model.config.d_model
        )

    def forward(self, pixel_values=None, input_ids=None,
                attention_mask=None, labels=None):
        encoder_outputs = self.vision_encoder(pixel_values=pixel_values)
        image_feats     = encoder_outputs.last_hidden_state  

        image_feats = self.proj(image_feats) 

        image_feats = image_feats.mean(dim=1, keepdim=True)

        image_attention_mask = torch.ones(
            (image_feats.shape[0], 1),
            dtype=torch.long,
            device=image_feats.device,
        )

        if input_ids is not None:
            input_embeds = self.text_model.get_input_embeddings()(input_ids)  
            inputs_embeds = torch.cat([image_feats, input_embeds], dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat([image_attention_mask, attention_mask], dim=1)
        else:
            inputs_embeds = image_feats
            attention_mask = image_attention_mask

        outputs = self.text_model(
            inputs_embeds     = inputs_embeds,
            attention_mask   = attention_mask,
            labels           = labels,
        )
        return outputs

global_model = VisionT5Model(vision_encoder, text_decoder).to(device)

global_model.text_model.config.decoder_start_token_id = tokenizer.pad_token_id
global_model.text_model.config.eos_token_id           = tokenizer.eos_token_id
global_model.text_model.config.pad_token_id           = tokenizer.pad_token_id
global_model.text_model.config.vocab_size             = global_model.text_model.config.vocab_size
global_model.text_model.config.max_length             = 128
global_model.text_model.config.no_repeat_ngram_size   = 3
global_model.text_model.config.early_stopping         = True
global_model.text_model.config.num_beams              = 4
global_model.text_model.config.temperature            = 1.0


train_csv = SPLIT_DIR / "train.csv"
val_csv   = SPLIT_DIR / "val.csv"
test_csv  = SPLIT_DIR / "test.csv"

full_train_ds = MultimodalOpenIDataset(
    csv_path = train_csv,
    img_dir  = IMG_DIR,
    feature_extractor = feature_extractor,
    tokenizer         = tokenizer
)

all_indices = list(range(len(full_train_ds)))
random.shuffle(all_indices)
client_indices = np.array_split(all_indices, NUM_CLIENTS)

device = device 

for rnd in range(FED_ROUNDS):
    print(f"  → Round {rnd+1}/{FED_ROUNDS}")
    client_states = []

    for c in range(NUM_CLIENTS):
        client_model = VisionT5Model(
            vision_encoder = ViTModel.from_pretrained(VISION_NAME),
            text_model     = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
        )
        client_model.load_state_dict(global_model.state_dict())
        client_model.to(device)

        local_indices = client_indices[c].tolist()
        local_subset  = torch.utils.data.Subset(full_train_ds, local_indices)

        client_output_dir = OUTPUT_ROOT / f"client{c}_round{rnd}"
        args = Seq2SeqTrainingArguments(
            output_dir             = str(client_output_dir),
            per_device_train_batch_size = TRAIN_BS,
            per_device_eval_batch_size  = TRAIN_BS,
            num_train_epochs       = LOCAL_EPOCHS,
            learning_rate          = LR,
            weight_decay           = 0.01,
            logging_steps          = 100,
            save_strategy          = "no",        
            do_eval                = False,
            remove_unused_columns  = False,
            seed                   = SEED,
            report_to              = ["none"],    
            fp16                   = torch.cuda.is_available(),
        )

        collator = default_data_collator

        trainer = Seq2SeqTrainer(
            model            = client_model,
            args             = args,
            train_dataset    = local_subset,
            tokenizer        = tokenizer,       
            data_collator    = collator,
        )

        trainer.train()

        client_states.append({k: v.cpu() for k, v in client_model.state_dict().items()})

        del trainer, client_model
        torch.cuda.empty_cache()

    new_global_state = {}
    for key in client_states[0].keys():
        stacked = torch.stack([st[key] for st in client_states], dim=0)
        new_global_state[key] = torch.mean(stacked, dim=0)
    global_model.load_state_dict(new_global_state)
    global_model.to(device)

    round_dir = OUTPUT_ROOT / f"round_{rnd}"
    round_dir.mkdir(exist_ok=True)

    vision_encoder_dir = round_dir / "vision_encoder"
    vision_encoder_dir.mkdir(exist_ok=True)
    global_model.vision_encoder.save_pretrained(vision_encoder_dir)

    text_decoder_dir = round_dir / "text_decoder"
    text_decoder_dir.mkdir(exist_ok=True)
    global_model.text_model.save_pretrained(text_decoder_dir)

    torch.save(
        {
            "vision_t5_state_dict": global_model.state_dict(),
            "proj_weight":          global_model.proj.weight.detach().cpu().numpy().tolist(),
            "proj_bias":            global_model.proj.bias.detach().cpu().numpy().tolist(),
        },
        round_dir / "vision_t5_full_state.pt",
    )

    (round_dir / "tokenizer").mkdir(exist_ok=True)
    (round_dir / "feature_extractor").mkdir(exist_ok=True)
    tokenizer.save_pretrained(round_dir / "tokenizer")
    feature_extractor.save_pretrained(round_dir / "feature_extractor")

    print(f"Saved checkpoint after Round {rnd} → {round_dir}")


final_dir = OUTPUT_ROOT / "federated_multimodal_t5_final_more_clients"
final_dir.mkdir(exist_ok=True)

( final_dir / "vision_encoder" ).mkdir(exist_ok=True)
global_model.vision_encoder.save_pretrained(final_dir / "vision_encoder")

( final_dir / "text_decoder" ).mkdir(exist_ok=True)
global_model.text_model.save_pretrained(final_dir / "text_decoder")

torch.save(
    {
        "vision_t5_state_dict": global_model.state_dict(),
        "proj_weight":          global_model.proj.weight.detach().cpu().numpy().tolist(),
        "proj_bias":            global_model.proj.bias.detach().cpu().numpy().tolist(),
    },
    final_dir / "vision_t5_full_state.pt",
)

( final_dir / "tokenizer" ).mkdir(exist_ok=True)
( final_dir / "feature_extractor" ).mkdir(exist_ok=True)
tokenizer.save_pretrained(final_dir / "tokenizer")
feature_extractor.save_pretrained(final_dir / "feature_extractor")

print("Federated multimodal‐T5 saved under:", final_dir.resolve())
