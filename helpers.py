# System Libraries
import os
import sys
import glob
import copy
import math
import random
import warnings
warnings.filterwarnings("ignore")
import subprocess
from pathlib import Path
from tqdm.auto import tqdm
from IPython.display import Audio, display

# Pytorch Libraries
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Hugging Face Transformers Libraries
from transformers import AutoProcessor, MusicgenForConditionalGeneration, MusicgenProcessor

# ML Libraries
from sklearn.model_selection import train_test_split
import scipy
import pandas as pd
import numpy as np

# Visualization Libraries
import matplotlib.pyplot as plt

# Audio Libraries
import librosa

# AudioCraft Libraries
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from audiocraft.modules.conditioners import ConditioningAttributes

# Custom Libraries
from classes import *


########################################################
# Helper Functions - General
########################################################

## Load Device
def load_device(device_name):
    if device_name == "mps":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = "cpu"
    return device

## Load MusicGen model
def load_musicgen_model(model_name):
    model = MusicGen.get_pretrained(model_name)
    return model

## Load Compression model
def load_compression_model(model):
    compression_model = model.compression_model
    return compression_model

## Save the audio clip
def save_the_audio(file_path,wav,m):
    audio_write(file_path, wav[0].cpu(), m.sample_rate, strategy="loudness")
    print(f"Audio saved as: {file_path}.wav")

## Display the audio clip
def display_the_audio(wav,sr):
    Audio(wav[0].cpu().squeeze().numpy(), rate=sr)

########################################################
# Helper Functions - EnCodec Confirmation
########################################################
## Load Sample Audio for EnCodec confirmation and prepare a segment to feed in to EnCodec encoder
def load_sample_audio_segment_for_encodec_confirmation(dataset_directory, duration_seconds=10, target_sample_rate=32000):
    """
    Loads a sample audio segment from the dataset directory,
    converts it to mono, resamples it to 32kHz (SR),
    crops it to the desired duration starting at an offset of 30 seconds,
    and returns the audio segment.
    """

    DATASET_DIR = dataset_directory
    wav_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True))
 #    print(f"\nFound {len(wav_files)} wav files.")
 #    print("Example:", wav_files[0] if wav_files else "None")

    #Pick one file
    wav_path = wav_files[3]
 #    print("Using:", wav_path)

    #Load the audio file
    wav, sr = torchaudio.load(wav_path)  # [C, T] (Channels, Time)
 #    print("Loaded shape:", wav.shape, "sr:", sr)

    #Convert to mono channel if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
 #        print("Converted to mono, new shape:", wav.shape)
    else:
 #        print("Already mono.")
        pass

    #Resample to 32kHz if needed
    TARGET_SR = target_sample_rate
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
        sr = TARGET_SR
 #        print("After resampling sr:", sr)
    else:
 #        print("Already at target sample rate 32kHz.")
        pass
    
    #Crop the audio to the desired duration starting at an offset
    DUR_SEC = duration_seconds
    num_samples = DUR_SEC * sr

    #If the file is shorter than 10 seconds, pad with zeros (rare but safe)
    if wav.shape[1] < num_samples:
        pad = num_samples - wav.shape[1]
        wav_10s = torch.nn.functional.pad(wav, (0, pad))
        
    else:
        start_sec = 30  # change this to crop from later, e.g., 30 for 30s in
        start = start_sec * sr
        end = start + num_samples
        # Ensure we don't go past end
        if end > wav.shape[1]:
            start = max(0, wav.shape[1] - num_samples)
            end = start + num_samples
        wav_10s = wav[:, start:end]

 #    print(f"Trimmed the audio to {DUR_SEC} seconds, starting from 30 seconds")
 #    print("Cropped shape:", wav_10s.shape, "sr:", sr)
    print(f"Loaded {wav_path} and trimmed the audio to {DUR_SEC} seconds, starting from 30 seconds")
    
    return wav_10s

########################################################
# Helper Functions - Dataset Loading, Cleaning & Segmentation
########################################################

## Set Configuration for Dataset Loading, Cleaning & Segmentation
def set_configuration_for_dataset_loading_cleaning_segmentation():
    """
    Sets the configuration for dataset loading, cleaning, and segmentation.
    """
    # Input and output folders
    DATASET_DIR = "dataset_wav"
    OUTPUT_DIR = "dataset_segments_10s"

    #************ Audio settings ************

    # These define the audio format required by MusicGen / EnCodec.
    # MusicGen and EnCodec were trained using 32,000 samples per second    
    TARGET_SR = 32000

    # MusicGen is trained on fixed-length audio chunks.
    # Typical durations: 5 sec	quick testing, 10 sec	common training, 30 sec	larger models
    # For Apple M2 machine, 10 sec is ideal.

    SEGMENT_SEC = 10

    # Multiplying SEGMENT_SEC by TARGET_SR converts seconds → samples.
    # EnCodec encoder will therefore receive: [B, C, T] = [1, 1, 320000]

    SEGMENT_SAMPLES = TARGET_SR * SEGMENT_SEC

    ########################################################
    #************ Silence trimming settings ************
    ########################################################
    # These remove long silent parts of recordings.
    # This is important because Carnatic concerts often contain:
    # •	audience noise
    # •	silence before music starts
    # •	pauses between songs


    # Audio amplitude range: -1.0 → +1.0
    # If audio amplitude is below 0.01, we treat it as silence.
    # Example waveform: 0.7 is loud, 0.3 is medium, 0.05 is quiet, 0.001 is silence
    # |x| < 0.01 → silence

    SILENCE_THRESHOLD = 0.01   # amplitude threshold

    # Silence shorter than 0.5 sec is ignored, because Music naturally contains tiny pauses between notes.

    MIN_SILENCE_SEC = 0.5      # minimum duration to consider silence

    # MIN_SILENCE_SAMPLES converts seconds → samples:
    # 0.5 sec × 32000 = 16000 samples
    # So silence must last 16000 samples to be treated as actual silence.

    MIN_SILENCE_SAMPLES = int(MIN_SILENCE_SEC * TARGET_SR)

    ########################################################
    #************ Heuristic filtering ************
    ########################################################
    # These detect bad or unusable audio segments.

    # MIN_RMS
    # RMS = Root Mean Square amplitude
    # RMS measures average loudness.
    # Example: RMS Value of 0.0005 is silence, 0.002 is quiet background, 0.05 is music, 0.2 is loud music
    # RMS < 0.005 → too quiet
    # Meaning the segment likely contains: silence, noise, microphone hiss

    MIN_RMS = 0.005            # too quiet => likely silence / unusable

    # MAX_ZERO_RATIO
    # This measures percentage of near-zero samples
    # Example: In a segment of 320000 samples, if 100000 samples are near-zero, then:
    # zero_ratio = 100000 / 320000 ≈ 0.31
    # This means 31% of the segment is silence.
    # max allowed = 30%
    # So that segment would be rejected.

    MAX_ZERO_RATIO = 0.30      # too many near-zero samples => likely silence-heavy

    # CLIP_AMPLITUDE
    # This checks if audio is clipped.
    # Clipping occurs when audio exceeds recording limits:
    # max amplitude = 1.0
    # Clipped waveform looks flat instead of smooth curves.
    # Clipped audio is bad for training.
    # This threshold simply flags suspicious signals.

    CLIP_AMPLITUDE = 0.99      # sanity check for clipped audio

    ########################################################
    #************ Segment acceptance ************
    ########################################################

    # At least 60% of the segment must contain real sound

    # Example: In a segment of 10 seconds, if only 3 seconds contain music, then non_silent_ratio = 3/10 =  0.3
    # That segment gets rejected.
    # But if 7 seconds contain music, then ratio = 0.7 → accepted

    MIN_NON_SILENT_RATIO = 0.60   # segment should have enough active audio

    return DATASET_DIR, OUTPUT_DIR, TARGET_SR, SEGMENT_SEC, SEGMENT_SAMPLES, SILENCE_THRESHOLD, MIN_SILENCE_SEC, MIN_SILENCE_SAMPLES, MIN_RMS, MAX_ZERO_RATIO, CLIP_AMPLITUDE, MIN_NON_SILENT_RATIO

## Create output directory
def create_output_directory(OUTPUT_DIR):
    """
    Creates the output directory if it doesn't exist.
    """
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        # print("Output directory created:", OUTPUT_DIR)
    else:
        # print("Output directory already exists:", OUTPUT_DIR)
        pass

# Load + Resample + Mono
def load_audio(filepath, target_sr=32000):
    wav, sr = torchaudio.load(filepath)  # [C, T]

    # Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr

# Trim Leading and trailing silence
# This removes silence at the beginning and end
def trim_silence(wav, threshold=0.01):
    """
    wav: [1, T]
    Returns trimmed waveform [1, T_trimmed]
    """
    x = wav.abs().squeeze(0)  # [T]
    non_silent = torch.where(x > threshold)[0]

    if len(non_silent) == 0:
        return None

    start = non_silent[0].item()
    end = non_silent[-1].item() + 1

    return wav[:, start:end]

# Copmute simple Quality features
def compute_audio_stats(wav):
    """
    wav: [1, T]
    """
    x = wav.squeeze(0)

    rms = torch.sqrt(torch.mean(x ** 2)).item()
    zero_ratio = (x.abs() < 1e-4).float().mean().item()
    max_amp = x.abs().max().item()

    return {
        "rms": rms,
        "zero_ratio": zero_ratio,
        "max_amp": max_amp
    }

# Estimate non-silent ratio in a segment
def non_silent_ratio(wav, threshold=0.01):
    x = wav.abs().squeeze(0)
    ratio = (x > threshold).float().mean().item()
    return ratio

# Split into 10-second chunks
# We will use non-overlapping chunks first.
def segment_audio(wav, segment_samples):
    """
    wav: [1, T]
    Returns list of [1, segment_samples]
    """
    total_samples = wav.shape[1]
    num_segments = total_samples // segment_samples

    segments = []
    for i in range(num_segments):
        start = i * segment_samples
        end = start + segment_samples
        seg = wav[:, start:end]
        segments.append((i, seg))
    return segments

# Simple Segment Filter
# This keeps reasonably active music segments and removes silence-heavy pieces.
def is_valid_segment(seg, SILENCE_THRESHOLD, MIN_RMS, MAX_ZERO_RATIO, MIN_NON_SILENT_RATIO):
    stats = compute_audio_stats(seg)
    ns_ratio = non_silent_ratio(seg, threshold=SILENCE_THRESHOLD)

    if stats["rms"] < MIN_RMS:
        return False, {**stats, "non_silent_ratio": ns_ratio, "reason": "low_rms"}

    if stats["zero_ratio"] > MAX_ZERO_RATIO:
        return False, {**stats, "non_silent_ratio": ns_ratio, "reason": "high_zero_ratio"}

    if ns_ratio < MIN_NON_SILENT_RATIO:
        return False, {**stats, "non_silent_ratio": ns_ratio, "reason": "too_much_silence"}

    return True, {**stats, "non_silent_ratio": ns_ratio, "reason": "accepted"}

########################################################
# Helper Functions - Train, Test, Val Split
########################################################

# Assigns the split to each segment.
def assign_split(source_file, train_set, val_set, test_set):
    """
    Assigns the split to each segment.
    """
    if source_file in train_set:
        return "train"
    elif source_file in val_set:
        return "val"
    elif source_file in test_set:
        return "test"
    else:
        return "unknown"

########################################################
# Helper Functions - Tokenizer Loading + Token Caching
########################################################

#Set Configuration for Tokenizer Loading + Token Caching
def set_configuration_for_tokenization():
    """
    Sets the configuration for tokenizer loading and token caching.
    """
    #Input CSVs from your split step
    SPLIT_DIR = "splits"
    TRAIN_CSV = os.path.join(SPLIT_DIR, "train_segments.csv")
    VAL_CSV   = os.path.join(SPLIT_DIR, "val_segments.csv")
    TEST_CSV  = os.path.join(SPLIT_DIR, "test_segments.csv")
    #Output token directory
    TOKEN_DIR = "dataset_tokens_10s"
    c
    os.makedirs(TOKEN_DIR, exist_ok=True)
    #Audio settings
    TARGET_SR = 32000

    return TRAIN_CSV, VAL_CSV, TEST_CSV, TOKEN_DIR, TARGET_SR

#load segment safely
def load_segment(filepath, target_sr=32000):
    wav, sr = torchaudio.load(filepath)   # [C, T]

    #Convert to mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    #Resample
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    return wav, sr

#encode one segment and save tokens
def encode_and_save_segment(segment_path, output_path, compression_model, TARGET_SR, device="cpu"):
    wav, sr = load_segment(segment_path, TARGET_SR)

    #Add batch dimension: [1, C, T]
    wav_in = wav.unsqueeze(0).to(device)

    with torch.no_grad():
        codes, scale = compression_model.encode(wav_in)

    # Remove batch dimension for storage
    codes_to_save = codes[0].cpu()   # [K, T']
    
    if scale is not None:
        # scale may be [B] or [B, ...] depending on version
        scale_to_save = scale[0].cpu() if hasattr(scale, "__len__") and len(scale) > 0 else scale
    else:
        scale_to_save = None

    payload = {
                "codes": codes_to_save,
                "scale": scale_to_save,
                "sample_rate": sr,
                "segment_path": segment_path
            }

    torch.save(payload, output_path)


 #Process one split
 #This function reads each segment file, tokenizes it, saves .pt, returns updated metadata

#Caches the tokens for a given split.
def cache_tokens_for_split(df_split, split_name, compression_model, token_root, TARGET_SR, device):
    split_token_dir = os.path.join(token_root, split_name)
    os.makedirs(split_token_dir, exist_ok=True)

    records = []

    for _, row in tqdm(df_split.iterrows(), total=len(df_split), desc=f"Caching {split_name}"):
        segment_path = row["segment_file"]
        source_file = row["source_file"]

        base_name = Path(segment_path).stem
        token_path = os.path.join(split_token_dir, f"{base_name}.pt")

        try:
            encode_and_save_segment(segment_path, token_path, compression_model, TARGET_SR, device=device)
            status = "saved"

        except Exception as e:
            token_path = None
            status = f"error: {str(e)}"

        records.append({
                            "source_file": source_file,
                            "segment_file": segment_path,
                            "token_file": token_path,
                            "split": split_name,
                            "status": status
                        })

    return pd.DataFrame(records)

########################################################
# Helper Functions - LoRA Finetuning
########################################################

#Sets the configuration for LoRA finetuning.
def set_lora_configuration():
    LORA_RANK = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    return LORA_RANK, LORA_ALPHA, LORA_DROPOUT

#Sets the configuration for LoRA training.
def set_lora_training_configuration():
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8 # simulate batch size 8
    NUM_EPOCHS = 10
    LR = 1e-4
    WEIGHT_DECAY = 1e-4 #regularization #Adds a penalty to large weights during training #prevents overfitting
    MAX_GRAD_NORM = 1.0 #stabilize updates #Gradient clipping
    PATIENCE = 3  # early stopping
    return BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LR, WEIGHT_DECAY, MAX_GRAD_NORM, PATIENCE

#Sets the configuration for LoRA debugging.
def set_lora_debug_configuration():
    USE_DEBUG_SUBSET = False # If True, only train on a small subset for debugging.
    DEBUG_TRAIN_N = 512 # Number of training samples to use for debugging.
    DEBUG_VAL_N = 64 # Number of validation samples to use for debugging.
    return USE_DEBUG_SUBSET, DEBUG_TRAIN_N, DEBUG_VAL_N

#Sets the configuration for LoRA token configuration.
def set_lora_token_configuration():
    
    TOKEN_META_DIR = "dataset_tokens_10s/metadata"
    TRAIN_TOKENS_CSV = os.path.join(TOKEN_META_DIR, "train_tokens.csv")
    VAL_TOKENS_CSV   = os.path.join(TOKEN_META_DIR, "val_tokens.csv")

    return TOKEN_META_DIR, TRAIN_TOKENS_CSV, VAL_TOKENS_CSV

#Creates the checkpoint directory for LoRA finetuning.
def create_lora_checkpoint_directory():
    CHECKPOINT_DIR = "checkpoints_lora_musicgen_small"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return CHECKPOINT_DIR

#Collate the tokens for a given batch.
def collate_rvq(batch):
    codes = torch.stack([item["codes"] for item in batch], dim=0)  # [B, K, T]
    return {
                "codes": codes,
                "token_file": [item["token_file"] for item in batch],
                "segment_file": [item["segment_file"] for item in batch],
                "source_file": [item["source_file"] for item in batch],
            }

#Builds the train and val datasets.
def build_train_val_datasets(TRAIN_TOKENS_CSV, VAL_TOKENS_CSV, USE_DEBUG_SUBSET, DEBUG_TRAIN_N, DEBUG_VAL_N, BATCH_SIZE):
    train_dataset = CachedRVQDataset(
                                    TRAIN_TOKENS_CSV,
                                    debug_n=DEBUG_TRAIN_N if USE_DEBUG_SUBSET else None
                                      )
    print("Train dataset Created")

    val_dataset = CachedRVQDataset(
                                    VAL_TOKENS_CSV,
                                    debug_n=DEBUG_VAL_N if USE_DEBUG_SUBSET else None
                                    )
    print("Val dataset Created")

    train_loader = DataLoader(
                                train_dataset,
                                batch_size=BATCH_SIZE,
                                shuffle=True,
                                num_workers=0,
                                collate_fn=collate_rvq
                                )
    print("Train dataset loaded into DataLoader")

    val_loader = DataLoader(
                            val_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_rvq
                        )

    print("Val dataset loaded into DataLoader")

    return train_loader, val_loader

#Freezes the base LM.
def freeze_base_lm(lm):
    for p in lm.parameters():
        p.requires_grad = False

    print("MusicGen small model loaded and all base LM parameters frozen.")
    
#Checks if the module is a linear layer and if it is, it returns True.
#We target linear layers in the Transformer LM, but skip the condition provider because you are training unconditional continuation.
def should_lora_wrap(module_name, module):
    if not isinstance(module, nn.Linear):
        return False

    # Skip condition provider / text conditioners
    # Since the focus is on unconditional training, there is no need to adapt text-conditioning modules.
    if "condition_provider" in module_name:
        return False

    return True


# The below function walks through the model recursively and replaces eligible linear layers with LoRALinear.
def apply_lora_recursively(module, prefix="", rank=8, alpha=16, dropout=0.0):
    replaced = 0

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if should_lora_wrap(full_name, child):
            # Replaces original child module with LoRA-wrapped version.
            setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout))
            replaced += 1

        else:
            replaced += apply_lora_recursively(
                child,
                prefix=full_name,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )

    return replaced

#Tries to enable gradient checkpointing.
#Gradient checkpointing is a technique to reduce the memory usage of the model by only computing the gradients of the required parts of the model.
#This is useful for large models that do not fit in memory.
def try_enable_gradient_checkpointing(lm_model):
    enabled = False

    # common style
    if hasattr(lm_model, "gradient_checkpointing_enable"):
        lm_model.gradient_checkpointing_enable()
        enabled = True

    # transformer submodule
    if hasattr(lm_model, "transformer") and hasattr(lm_model.transformer, "gradient_checkpointing_enable"):
        lm_model.transformer.gradient_checkpointing_enable()
        enabled = True

    # custom flag style
    if hasattr(lm_model, "transformer") and hasattr(lm_model.transformer, "gradient_checkpointing"):
        lm_model.transformer.gradient_checkpointing = True
        enabled = True

    print("Gradient checkpointing enabled?" , enabled)


#Initializes the optimizer.
def initialize_optimizer(lm, LR, WEIGHT_DECAY):
    optimizer = torch.optim.AdamW(
                                [p for p in lm.parameters() if p.requires_grad],
                                lr=LR,
                                weight_decay=WEIGHT_DECAY
                            )
    return optimizer



# Creates empty conditioning objects because you are training unconditional continuation.
# def build_empty_conditions(batch_size):
#     return [ConditioningAttributes() for _ in range(batch_size)]
def build_empty_conditions(batch_size):
    conds = []
    for _ in range(batch_size):
        c = ConditioningAttributes()
        c.text = {"description": ""}
        conds.append(c)
    return conds

#Moves the condition tensors to the device.
def move_condition_tensors_to_device(obj, device):
    """
    Recursively move nested condition tensors to target device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_condition_tensors_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_condition_tensors_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_condition_tensors_to_device(v, device) for v in obj)
    else:
        return obj


#Builds the condition tensors on CPU.
@torch.no_grad()
def build_condition_tensors_on_cpu(lm_model, batch_size, target_device):
    """
    Build empty-text condition tensors on CPU using the condition provider,
    then move the resulting tensors to target_device.
    """
    conds = build_empty_conditions(batch_size)

    # Move condition provider temporarily to CPU
    lm_model.condition_provider = lm_model.condition_provider.to("cpu")

    tokenized = lm_model.condition_provider.tokenize(conds)
    condition_tensors = lm_model.condition_provider(tokenized)

    # Move resulting tensors to the LM device
    condition_tensors = move_condition_tensors_to_device(condition_tensors, target_device)

    return condition_tensors

#Computes the cross-entropy loss.
def compute_ce_loss(lm_model, codes):
    """
    codes: [B, K, T], dtype long

    Build empty-text condition tensors on CPU, move them to the LM device,
    and bypass the live T5 path during the LM forward pass.
    """
    B = codes.shape[0]

    # Build condition tensors externally on CPU, then move to device
    condition_tensors = build_condition_tensors_on_cpu(
                                                        lm_model,
                                                        batch_size=B,
                                                        target_device=codes.device
                                                    )

    outputs = lm_model.compute_predictions(
                                            codes=codes,
                                            conditions=[],                    # do NOT trigger tokenize path inside LM.forward
                                            condition_tensors=condition_tensors
                                        )

    logits = outputs.logits              # [B, K, T, card]
    mask = outputs.mask.bool()           # [B, K, T]

    B_, K, T, card = logits.shape
    assert codes.shape == (B_, K, T)

    logits_flat = logits.reshape(-1, card)
    targets_flat = codes.reshape(-1)
    mask_flat = mask.reshape(-1)

    valid_logits = logits_flat[mask_flat]
    valid_targets = targets_flat[mask_flat]

    loss = F.cross_entropy(valid_logits, valid_targets)
    return loss

#This function evaluates the loss of the model on the validation set.
#It is used to evaluate the performance of the model on the validation set.
@torch.no_grad()
def evaluate_loss(lm_model, dataloader, device="cpu", max_batches=None):
    lm_model.eval()
    losses = []

    for i, batch in enumerate(tqdm(dataloader, desc="Validation", leave=False)):
        if max_batches is not None and i >= max_batches:
            break

        codes = batch["codes"].to(device)
        loss = compute_ce_loss(lm_model, codes)
        losses.append(loss.item())

    return sum(losses) / max(1, len(losses))

#Extracts the LoRA state dictionary from the model.
def extract_lora_state_dict(model):
    sd = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            sd[f"{name}.lora_A"] = module.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = module.lora_B.detach().cpu()
    return sd

#Saves the LoRA checkpoint.
def save_lora_checkpoint(model, path, extra=None):
    payload = {
        "lora_state_dict": extract_lora_state_dict(model),
        "extra": extra or {}
    }
    torch.save(payload, path)

#Loads the LoRA checkpoint.
def load_lora_checkpoint(model, path, device="cpu"):
    payload = torch.load(path, map_location=device)
    lora_sd = payload["lora_state_dict"]

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            key_A = f"{name}.lora_A"
            key_B = f"{name}.lora_B"
            if key_A in lora_sd:
                module.lora_A.data.copy_(lora_sd[key_A].to(module.lora_A.device))
            if key_B in lora_sd:
                module.lora_B.data.copy_(lora_sd[key_B].to(module.lora_B.device))

    return payload.get("extra", {})

#Runs the training loop.
def run_training_loop(lm, optimizer, train_loader, val_loader, NUM_EPOCHS, GRAD_ACCUM_STEPS, MAX_GRAD_NORM, CHECKPOINT_DIR, PATIENCE, device):
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    global_step = 0

    history = []

    for epoch in range(NUM_EPOCHS):
        lm.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")

        for step, batch in enumerate(pbar):
            codes = batch["codes"].to(device)

            loss = compute_ce_loss(lm, codes)
            loss_for_backward = loss / GRAD_ACCUM_STEPS
            loss_for_backward.backward()

            running_loss += loss.item()
            global_step += 1

            if (step + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in lm.parameters() if p.requires_grad],
                    MAX_GRAD_NORM
                )
                optimizer.step()
                optimizer.zero_grad()

            avg_train_loss = running_loss / (step + 1)
            pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}")

        # final optimizer step if loop ended mid-accumulation
        if len(train_loader) % GRAD_ACCUM_STEPS != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in lm.parameters() if p.requires_grad],
                MAX_GRAD_NORM
            )
            optimizer.step()
            optimizer.zero_grad()

        val_loss = evaluate_loss(lm, val_loader, device=device)

        print(f"\nEpoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": val_loss
        })

        # save last
        last_ckpt = os.path.join(CHECKPOINT_DIR, "lora_last.pt")
        save_lora_checkpoint(lm, last_ckpt, extra={"epoch": epoch + 1, "val_loss": val_loss})

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            best_ckpt = os.path.join(CHECKPOINT_DIR, "lora_best.pt")
            save_lora_checkpoint(lm, best_ckpt, extra={"epoch": epoch + 1, "val_loss": val_loss})
            print(f"Saved new best checkpoint: {best_ckpt}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")

            if epochs_without_improvement >= PATIENCE:
                print("Early stopping triggered.")
                break

    return history

#Saves the training history.
def save_training_history(CHECKPOINT_DIR, history):
    history_df = pd.DataFrame(history)
    history_csv = os.path.join(CHECKPOINT_DIR, "training_history.csv")
    history_df.to_csv(history_csv, index=False)
    print(f"Training history saved to: {history_csv}")
    print(history_df)


########################################################
# Helper Functions - LoRA Post Finetuning Evaluation
########################################################

#Reusing LoRALinear, apply_lora_recursively and load_lora_checkpoint from LoRA Finetuning

#Sets the configuration for LoRA post finetuning evaluation.
def set_post_finetuning_evaluation_configuration():
    
    TEST_TOKENS_CSV = "dataset_tokens_10s/metadata/test_tokens.csv"
    LORA_BEST_CKPT = "checkpoints_lora_musicgen_small/lora_best.pt"

    EVAL_DIR = "evaluation_results"
    os.makedirs(EVAL_DIR, exist_ok=True)

    # Fixed seeds for reproducibility
    UNCOND_SEEDS = [11, 22, 33]
    # UNCOND_SEEDS = [42]
    
    NUM_TEST_CLIPS = 50   # adjust as needed
    CONT_TOTAL_DURATION = 20  # 10s prompt + 10s continuation 
    
    return TEST_TOKENS_CSV, LORA_BEST_CKPT, EVAL_DIR, UNCOND_SEEDS, NUM_TEST_CLIPS, CONT_TOTAL_DURATION


#Helper to load baseline model
def load_baseline_model(device="cpu"):
    m = MusicGen.get_pretrained("facebook/musicgen-small")
    m.lm = m.lm.to(device)
    m.compression_model = m.compression_model.to(device)
    m.lm.condition_provider = m.lm.condition_provider.to(device)
    m.device = torch.device(device)
    return m

#Helper to load fine-tuned LoRA model
def load_finetuned_lora_model(LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BEST_CKPT, device="cpu"):
    m = MusicGen.get_pretrained("facebook/musicgen-small")
    lm = m.lm

    # Freeze base params
    for p in lm.parameters():
        p.requires_grad = False

    # Inject LoRA
    _ = apply_lora_recursively(
                                lm,
                                rank=LORA_RANK,
                                alpha=LORA_ALPHA,
                                dropout=LORA_DROPOUT
                            )

    # Move to device
    lm = lm.to(device)

    # Load LoRA checkpoint
    _ = load_lora_checkpoint(lm, LORA_BEST_CKPT, device=device)

    # Reattach into wrapper
    m.lm = lm
    m.compression_model = m.compression_model.to(device)
    m.lm.condition_provider = m.lm.condition_provider.to(device)
    m.device = torch.device(device)

    return m

#Helper to load and normalize a prompt clip
def load_prompt_wav(segment_path, target_sr):
    wav, sr = torchaudio.load(segment_path)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
        sr = target_sr

    wav = wav.unsqueeze(0)   # [1, C, T]
    return wav, sr

#Helper to save audio
def save_audio_tensor(base_path, wav_tensor, sr):
    """
    wav_tensor: [C, T] or [1, C, T]
    """
    if wav_tensor.dim() == 3:
        wav_tensor = wav_tensor[0]
    audio_write(base_path, wav_tensor.cpu(), sr, strategy="loudness")

#Generates unconditional audio for baseline and finetuned models.
def generate_unconditional_audio(baseline_model, finetuned_model, EVAL_DIR, UNCOND_SEEDS):
    uncond_dir = os.path.join(EVAL_DIR, "unconditional")
    os.makedirs(uncond_dir, exist_ok=True)

    for seed in UNCOND_SEEDS:
        torch.manual_seed(seed)

        #Evaluation for baseline model
        baseline_model.set_generation_params(duration=10)
        wav_base = baseline_model.generate_unconditional(num_samples=1)

        torch.manual_seed(seed)
        #Evaluation for finetuned model
        finetuned_model.set_generation_params(duration=10)
        wav_ft = finetuned_model.generate_unconditional(num_samples=1)


        base_path = os.path.join(uncond_dir, f"seed_{seed}_baseline")
        ft_path   = os.path.join(uncond_dir, f"seed_{seed}_finetuned")

        save_audio_tensor(base_path, wav_base, baseline_model.sample_rate)
        save_audio_tensor(ft_path, wav_ft, finetuned_model.sample_rate)

        print(f"Saved unconditional generation for seed {seed}")


# Generate audio continuation for baseline and finetuned models.
def generate_audio_continuation(baseline_model, finetuned_model, df_eval, EVAL_DIR, CONT_TOTAL_DURATION, device):
    cont_dir = os.path.join(EVAL_DIR, "continuation")
    os.makedirs(cont_dir, exist_ok=True)

    evaluation_rows = []

    for i, row in df_eval.iterrows():
        segment_path = row["segment_file"]
        source_file = row["source_file"]

        clip_name = Path(segment_path).stem
        clip_dir = os.path.join(cont_dir, clip_name)
        os.makedirs(clip_dir, exist_ok=True)

        wav_ref, sr_ref = load_prompt_wav(segment_path, baseline_model.sample_rate)

        # Save prompt
        save_audio_tensor(os.path.join(clip_dir, "prompt_10s"), wav_ref, sr_ref)

        # Baseline continuation
        baseline_model.set_generation_params(duration=CONT_TOTAL_DURATION)
        wav_base = baseline_model.generate_continuation(
                                                            wav_ref.to(device),
                                                            sr_ref,
                                                            descriptions=None
                                                        )

        save_audio_tensor(os.path.join(clip_dir, "baseline_20s"), wav_base, sr_ref)

        # Fine-tuned continuation
        finetuned_model.set_generation_params(duration=CONT_TOTAL_DURATION)
        wav_ft = finetuned_model.generate_continuation(
                                                        wav_ref.to(device),
                                                        sr_ref,
                                                        descriptions=None
                                                    )

        save_audio_tensor(os.path.join(clip_dir, "finetuned_20s"), wav_ft, sr_ref)

        # Save continuation-only portions too
        prompt_len_samples = wav_ref.shape[-1]

        wav_base_cont_only = wav_base[:, :, prompt_len_samples:]
        wav_ft_cont_only = wav_ft[:, :, prompt_len_samples:]

        save_audio_tensor(os.path.join(clip_dir, "baseline_cont_only_10s"), wav_base_cont_only, sr_ref)
        save_audio_tensor(os.path.join(clip_dir, "finetuned_cont_only_10s"), wav_ft_cont_only, sr_ref)

        evaluation_rows.append({
                                    "clip_name": clip_name,
                                    "source_file": source_file,
                                    "segment_file": segment_path,
                                    "prompt_file": os.path.join(clip_dir, "prompt_10s.wav"),
                                    "baseline_file": os.path.join(clip_dir, "baseline_20s.wav"),
                                    "finetuned_file": os.path.join(clip_dir, "finetuned_20s.wav"),
                                    "baseline_cont_only_file": os.path.join(clip_dir, "baseline_cont_only_10s.wav"),
                                    "finetuned_cont_only_file": os.path.join(clip_dir, "finetuned_cont_only_10s.wav"),
                                })

        print(f"Saved continuation comparison for {clip_name}")

        evaluation_df_metadata = pd.DataFrame(evaluation_rows)
        evaluation_df_metadata.to_csv(os.path.join(cont_dir, "evaluation_continuation_metadata.csv"), index=False)


########################################################

########################################################
# Helper Functions - Quantitative Evaluation - Fine-tuning CE Loss
########################################################

#Loads the finetuning metrics data.
def load_finetuning_metrics_data(finetuning_metrics_csv_path):
    
    # Path to your training history CSV
    history_csv = "checkpoints_lora_musicgen_small/2/training_history.csv"

    # Load CSV
    df_hist = pd.read_csv(history_csv)

    # Compute validation perplexity from validation CE loss
    df_hist["val_perplexity"] = df_hist["val_loss"].apply(lambda x: math.exp(x))

    # Optional: round for neat display
    df_display = df_hist.copy()
    df_display["train_loss"] = df_display["train_loss"].round(4)
    df_display["val_loss"] = df_display["val_loss"].round(4)
    df_display["val_perplexity"] = df_display["val_perplexity"].round(4)

    # Keep only the columns we want in the final table
    metrics_table = df_display[["epoch", "train_loss", "val_loss", "val_perplexity"]].copy()


    return metrics_table

#Plots the finetuning CE loss metrics.
def plot_fintuning_ce_loss_visualization(metrics_table,best_row):
    plot_dir = "evaluation_results/training_metrics"
    os.makedirs(plot_dir, exist_ok=True)

    plot_path = os.path.join(plot_dir, "training_validation_ce_loss.png")

    plt.figure(figsize=(8, 5))
    plt.plot(metrics_table["epoch"], metrics_table["train_loss"], marker="o", label="Training CE Loss")
    plt.plot(metrics_table["epoch"], metrics_table["val_loss"], marker="o", label="Validation CE Loss")
    plt.axvline(best_row["epoch"], linestyle="--", linewidth=1, label=f"Best Val Epoch = {int(best_row['epoch'])}")

    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation CE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved plot to:", plot_path)


########################################################
# Helper Functions - Quantitative Evaluation - Boundary Continuation Loss
########################################################

def set_boundary_continuation_evaluation_configuration():
    """
    Sets the boundary continuation evaluation configuration.
    Returns the configuration.
    """

    # Root folder created during post-finetune evaluation
    CONT_ROOT = "evaluation_results/continuation"

    # Output
    BOUNDARY_METRIC_DIR = "evaluation_results/boundary_metrics"
    os.makedirs(BOUNDARY_METRIC_DIR, exist_ok=True)

    # Audio config
    SR = 32000
    BOUNDARY_SEC = 1
    BOUNDARY_SAMPLES = SR * BOUNDARY_SEC

    # Feature config
    N_FFT = 2048
    HOP_LENGTH = 512
    N_MELS = 128
    N_MFCC = 20

    return CONT_ROOT, BOUNDARY_METRIC_DIR, SR, BOUNDARY_SEC, BOUNDARY_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, N_MFCC

# Load audio as mono, fixed sample rate
def load_audio_mono(path, sr=32000):
    y, sr = librosa.load(path, sr=sr, mono=True)
    return y, sr

# Extract 3-second boundary windows for prompt
def get_prompt_boundary(y, boundary_samples):
    """
    Last 3 seconds of prompt
    """
    if len(y) < boundary_samples:
        pad = boundary_samples - len(y)
        y = np.pad(y, (pad, 0))
    return y[-boundary_samples:]

# Extract 3-second boundary windows for continuation
def get_cont_boundary(y, boundary_samples):
    """
    First 3 seconds of continuation
    """
    if len(y) < boundary_samples:
        pad = boundary_samples - len(y)
        y = np.pad(y, (0, pad))
    return y[:boundary_samples]

# Calculate cosine distance between two vectors
def cosine_distance(a, b, eps=1e-8):
    a = np.asarray(a).flatten()
    b = np.asarray(b).flatten()

    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + eps
    sim = np.dot(a, b) / denom
    return 1.0 - sim

#############Feature extraction functions#############

# Mel feature vector
# We compute mel spectrogram, convert to dB, then average over time.

def extract_mel_vector(y, sr, N_FFT, HOP_LENGTH, N_MELS):
    mel = librosa.feature.melspectrogram(
                                            y=y,
                                            sr=sr,
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH,
                                            n_mels=N_MELS
                                        )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_vec = mel_db.mean(axis=1)   # shape: [n_mels]
    return mel_vec

# MFCC feature vector
#We compute MFCCs and average over time.

def extract_mfcc_vector(y, sr, N_MFCC, N_FFT, HOP_LENGTH):
    mfcc = librosa.feature.mfcc(
                                    y=y,
                                    sr=sr,
                                    n_mfcc=N_MFCC,
                                    n_fft=N_FFT,
                                    hop_length=HOP_LENGTH
                                )
    mfcc_vec = mfcc.mean(axis=1)    # shape: [n_mfcc]
    return mfcc_vec

# Chroma feature vector
# We compute chroma and average over time.

def extract_chroma_vector(y, sr, N_FFT, HOP_LENGTH):
    chroma = librosa.feature.chroma_stft(
                                            y=y,
                                            sr=sr,
                                            n_fft=N_FFT,
                                            hop_length=HOP_LENGTH
                                        )
    chroma_vec = chroma.mean(axis=1)   # shape: [12]
    return chroma_vec


# Onset-envelope vector
# Here, unlike the others, we keep the full onset envelope sequence across the 3-second region.

def extract_onset_vector(y, sr, HOP_LENGTH):
    onset_env = librosa.onset.onset_strength(
                                                y=y,
                                                sr=sr,
                                                hop_length=HOP_LENGTH
                                            )
    return onset_env


#############COMPUTE DISTANCE SAMPLES#############
#  Compute all 4 distances for one sample

def compute_boundary_metrics_for_sample(prompt_path, cont_path, boundary_samples, N_FFT, HOP_LENGTH, N_MELS, N_MFCC, sr=32000):
    # Load audio
    y_prompt, _ = load_audio_mono(prompt_path, sr=sr)
    y_cont, _ = load_audio_mono(cont_path, sr=sr)

    # Extract 3-second boundaries
    y_prompt_b = get_prompt_boundary(y_prompt, boundary_samples)
    y_cont_b = get_cont_boundary(y_cont, boundary_samples)

    # Mel
    mel_prompt = extract_mel_vector(y_prompt_b, sr, N_FFT, HOP_LENGTH, N_MELS)
    mel_cont = extract_mel_vector(y_cont_b, sr, N_FFT, HOP_LENGTH, N_MELS)
    mel_dist = cosine_distance(mel_prompt, mel_cont)

    # MFCC
    mfcc_prompt = extract_mfcc_vector(y_prompt_b, sr, N_MFCC, N_FFT, HOP_LENGTH)
    mfcc_cont = extract_mfcc_vector(y_cont_b, sr, N_MFCC, N_FFT, HOP_LENGTH)
    mfcc_dist = cosine_distance(mfcc_prompt, mfcc_cont)

    # Chroma
    chroma_prompt = extract_chroma_vector(y_prompt_b, sr, N_FFT, HOP_LENGTH)
    chroma_cont = extract_chroma_vector(y_cont_b, sr, N_FFT, HOP_LENGTH)
    chroma_dist = cosine_distance(chroma_prompt, chroma_cont)

    # Onset
    onset_prompt = extract_onset_vector(y_prompt_b, sr, HOP_LENGTH)
    onset_cont = extract_onset_vector(y_cont_b, sr, HOP_LENGTH)

    # Match onset vector length safely
    min_len = min(len(onset_prompt), len(onset_cont))
    onset_prompt = onset_prompt[:min_len]
    onset_cont = onset_cont[:min_len]

    onset_dist = cosine_distance(onset_prompt, onset_cont)

    return {
            "mel_boundary_distance": mel_dist,
            "mfcc_boundary_distance": mfcc_dist,
            "chroma_boundary_distance": chroma_dist,
            "onset_boundary_distance": onset_dist
        }


# Load generated audio directories.
def load_generated_audio_directories(CONT_ROOT):
    """
    Loads the generated audio directories.
    """
    sample_dirs = sorted([
                            d for d in glob.glob(os.path.join(CONT_ROOT, "*"))
                            if os.path.isdir(d)
                        ])
    print("Number of evaluation audio folders:", len(sample_dirs))
    
    return sample_dirs


def check_generated_audio_sr_mono(CONT_ROOT):
    rows = []

    wav_files = glob.glob(os.path.join(CONT_ROOT, "**/*.wav"), recursive=True)

    print("Total audio files found:", len(wav_files))

    for wav_path in tqdm(wav_files, desc="Checking audio properties"):
        try:
            wav, sr = torchaudio.load(wav_path)

            num_channels = wav.shape[0]
            num_samples = wav.shape[1]
            duration_sec = num_samples / sr

            rows.append({
                            "file": wav_path,
                            "sample_rate": sr,
                            "num_channels": num_channels,
                            "duration_sec": duration_sec
                        })

        except Exception as e:
            rows.append({
                            "file": wav_path,
                            "sample_rate": "ERROR",
                            "num_channels": "ERROR",
                            "duration_sec": "ERROR",
                            "error": str(e)
                        })

    df_audio_check = pd.DataFrame(rows)

    bad_sr = df_audio_check[df_audio_check["sample_rate"] != 32000]
    bad_channels = df_audio_check[df_audio_check["num_channels"] != 1]

    return int(bad_sr.empty and bad_channels.empty), bad_sr, bad_channels, df_audio_check

def compute_boundary_metrics(sample_dirs, SR, BOUNDARY_SAMPLES,N_FFT, HOP_LENGTH, N_MELS, N_MFCC,BOUNDARY_METRIC_DIR):

    rows = []

    # sample_dirs = sample_dirs[:10] # For testing
    # print(len(sample_dirs))

    for sample_dir in tqdm(sample_dirs, desc="Computing boundary metrics"):
        clip_name = Path(sample_dir).name

        prompt_path = os.path.join(sample_dir, "prompt_10s.wav")
        baseline_path = os.path.join(sample_dir, "baseline_cont_only_10s.wav")
        finetuned_path = os.path.join(sample_dir, "finetuned_cont_only_10s.wav")

        # Skip incomplete folders
        if not (os.path.exists(prompt_path) and os.path.exists(baseline_path) and os.path.exists(finetuned_path)):
            print(f"Skipping incomplete folder: {sample_dir}")
            continue

        # Baseline metrics
        baseline_metrics = compute_boundary_metrics_for_sample(
                                                                prompt_path=prompt_path,
                                                                cont_path=baseline_path,
                                                                boundary_samples=BOUNDARY_SAMPLES,
                                                                N_FFT=N_FFT, 
                                                                HOP_LENGTH=HOP_LENGTH, 
                                                                N_MELS=N_MELS,
                                                                N_MFCC=N_MFCC,
                                                                sr=SR,
                                                            )

        # Fine-tuned metrics
        finetuned_metrics = compute_boundary_metrics_for_sample(
                                                                    prompt_path=prompt_path,
                                                                    cont_path=finetuned_path,
                                                                    boundary_samples=BOUNDARY_SAMPLES,
                                                                    N_FFT=N_FFT, 
                                                                    HOP_LENGTH=HOP_LENGTH, 
                                                                    N_MELS=N_MELS,
                                                                    N_MFCC=N_MFCC,
                                                                    sr=SR,
                                                                )

        row = {
                "clip_name": clip_name,
                "prompt_file": prompt_path,
                "baseline_file": baseline_path,
                "finetuned_file": finetuned_path,

                "baseline_mel_boundary_distance": baseline_metrics["mel_boundary_distance"],
                "baseline_mfcc_boundary_distance": baseline_metrics["mfcc_boundary_distance"],
                "baseline_chroma_boundary_distance": baseline_metrics["chroma_boundary_distance"],
                "baseline_onset_boundary_distance": baseline_metrics["onset_boundary_distance"],

                "finetuned_mel_boundary_distance": finetuned_metrics["mel_boundary_distance"],
                "finetuned_mfcc_boundary_distance": finetuned_metrics["mfcc_boundary_distance"],
                "finetuned_chroma_boundary_distance": finetuned_metrics["chroma_boundary_distance"],
                "finetuned_onset_boundary_distance": finetuned_metrics["onset_boundary_distance"],
            }

        rows.append(row)

    df_boundary = pd.DataFrame(rows)
    print("Finished computing per-sample boundary metrics.")
    # df_boundary.head()

    per_sample_csv = os.path.join(BOUNDARY_METRIC_DIR, "boundary_metrics_per_sample.csv")
    df_boundary.to_csv(per_sample_csv, index=False)

    print("Saved per-sample metrics to:", per_sample_csv)


