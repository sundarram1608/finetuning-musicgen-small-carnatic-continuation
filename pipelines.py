from helpers import *

########################################################
# Hugging Face Examples for Text to Music
########################################################

def run_huggingface_examples_text_to_music():
    """
    This function runs the Hugging Face examples for text to music.
    This function is used to generate music from text using the Hugging Face MusicGen model.
    """
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        inputs = processor(
                            text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
                            padding=True,
                            return_tensors="pt",
                        )
        print("Before generating audio")
        audio_values = model.generate(**inputs, max_new_tokens=256)
        print("After generating audio")
        sampling_rate = model.config.audio_encoder.sampling_rate
        
        #Playing the generated audio and saving it as .wav file
        print("Before saving audio as .wav file")
        scipy.io.wavfile.write("examples/text-to-music/hf_eg_audio_1.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
        print("Audio saved as hf_eg_audio_1.wav")

        scipy.io.wavfile.write("examples/text-to-music/hf_eg_audio_2.wav", rate=sampling_rate, data=audio_values[1, 0].numpy())
        print("Audio saved as hf_eg_audio_2.wav")


        print("Before playing the generated audio")
        Audio(audio_values[0].numpy(), rate=sampling_rate)
        Audio(audio_values[1].numpy(), rate=sampling_rate)    

    except Exception as e:
        print(f"Error running Hugging Face examples for text to music: {e}")

def run_huggingface_examples_text_to_music_carnatic():
    """
    This function runs the Hugging Face examples for text to music for carnatic music.
    This function is used to generate music from text for carnatic music using the Hugging Face MusicGen model.
    """
    try:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")

        inputs = processor(
                            text=["Carnatic music with tabla and flute", "Traditional Carnatic composition with veena and mridangam"],
                            padding=True,
                            return_tensors="pt",
                        )

        audio_values = model.generate(**inputs, max_new_tokens=256)

        sampling_rate = model.config.audio_encoder.sampling_rate
        
        #Playing the generated audio and saving it as .wav file
        scipy.io.wavfile.write("examples/text-to-music/hf_eg_carnatic_audio_1.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
        print("Audio saved as hf_eg_carnatic_audio_1.wav")

        scipy.io.wavfile.write("examples/text-to-music/hf_eg_carnatic_audio_2.wav", rate=sampling_rate, data=audio_values[1, 0].numpy())
        print("Audio saved as hf_eg_carnatic_audio_2.wav")
        
        #Playing the generated audio
        Audio(audio_values[0].numpy(), rate=sampling_rate)
        Audio(audio_values[1].numpy(), rate=sampling_rate)    

    except Exception as e:
        print(f"Error running Hugging Face examples for text to music for carnatic music: {e}")

########################################################
# Reproducibility Check for AudioCraft
########################################################

def run_reproducibility_check_audiocraft():
    """
    Checks installed package versions and verifies the AudioCraft MusicGen model
    can be loaded, along with its sample rate and compression model type.
    """
    try:
        print("=== Package versions ===")
        subprocess.run(
                        [sys.executable, "-m", "pip", "show", "audiocraft", "torch", "torchaudio", "torchvision"],
                        check=False
                    )

        print("\n\n=== Loading MusicGen model ===")
        m = MusicGen.get_pretrained("small")
        print("\nsample rate:", m.sample_rate)

        cm = m.compression_model
        print("compression model:", type(cm))

        print("\nReproducibility check completed successfully.")

    except Exception as e:
        print(f"Error running reproducibility check: {e}")

########################################################
# Validate Unconditional Generation and EnCodec Confirmation
########################################################

## Confirmation to run musicgen-small inference locally and generate unconditional audio
def run_audiocraft_unconditional_generation_confirmation():
    """
    Runs unconditional MusicGen generation using the MusicGen small model for 10 seconds and saves the output audio.
    Loads the device, MusicGen model, and sets the generation parameters.
    Generates the audio, defines the output directory, creates the directory if it doesn't exist, joins the directory path and the filename, saves the audio, and displays the audio.
    """

    try:
        #Load device
        device = load_device("mps")
        print("device:", device)

        #Load MusicGen model
 #       print("Loading MusicGen model")
        model_name = "facebook/musicgen-small"
        m = load_musicgen_model(model_name)
        print("MusicGen small model loaded successfully")
        #Set generation parameters
 #       print("Setting generation parameters")
        m.set_generation_params(duration=10)
 #       print("Setting generation parameters completed successfully")
        #Generate audio
 #       print("Generating audio")
        wav = m.generate_unconditional(num_samples=1)
        print("Audio generated successfully")
        #Define output directory
 #       print("Defining output directory")
        output_dir = "examples/feasibility_check"
 #       print("Output directory defined successfully")
        #Create the directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        #Join the directory path and the filename
        file_path = os.path.join(output_dir, "unconditional_generation_example")
        #Save the audio
        save_the_audio(file_path,wav,m)

        #Display the audio
        display_the_audio(wav,sr = m.sample_rate)
        print("Unconditional generation audio confirmation completed successfully.")

    except Exception as e:
        print(f"Error running AudioCraft unconditional generation: {e}")

## Confirmation to load facebook/encodec_32khz and encode/decode a short clip
def run_audiocraft_encodec_confirmation():
    """
    Loads a sample audio segment from the dataset directory,
    converts it to mono, resamples it to 32kHz (SR),
    crops it to the desired duration starting at an offset of 30 seconds,
    and passes it to the EnCodec encoder. Decodes the output and saves the audio.
    """
    wav_10s = load_sample_audio_segment_for_encodec_confirmation(dataset_directory = "dataset_wav", duration_seconds=10, target_sample_rate=32000)
    #Load device
    device = load_device("mps")
    print("device:", device)

    #Load pretrained MusicGen and get its compression model (Encodec)
    model_name = "facebook/musicgen-small"
    m = load_musicgen_model(model_name)
    m.lm = m.lm.to(device)
    print("MusicGen small model loaded successfully")
    cm = load_compression_model(model = m)
    cm = cm.to(device)
    print("Loaded compression model:", type(cm))
    wav_in = wav_10s.unsqueeze(0).to(device)

    #Passing in to EnCodec encoder
    with torch.no_grad():
        enc_out = cm.encode(wav_in)
        codes, scale = enc_out
        wav_rec = cm.decode(codes, scale)

    #Define output directory
    output_dir = "examples/feasibility_check"

    #Create the folder if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Join the directory path and the filename
    file_path = os.path.join(output_dir, "encodec_encoded_decoded")
    #Save the audio
    save_the_audio(file_path,wav_rec,m)

    #Display the audio
    display_the_audio(wav_rec,sr = m.sample_rate)
    # Audio(wav_rec[0].cpu().squeeze().numpy(), rate=m.sample_rate)

    print("EnCodec encoded/decoded audio confirmation completed successfully.")

########################################################
# Dataset Loading, Cleaning & Segmentation
########################################################

def run_dataset_loading_cleaning_segmentation():
    """
    Loads the dataset, cleans the dataset, and segments the dataset.
    Loads the configuration for the dataset loading, cleaning & segmentation.
    Creates the output directory if it doesn't exist.
    Loads the wav files from the dataset directory.
    Initializes the records, saved_count, and rejected_count.
    Processes the wav files, trims the silence, segments the audio, and saves the segments.
    Creates the metadata CSV and prints the status of the segments.
    """
    DATASET_DIR, OUTPUT_DIR, TARGET_SR, SEGMENT_SEC, SEGMENT_SAMPLES, SILENCE_THRESHOLD, MIN_SILENCE_SEC, MIN_SILENCE_SAMPLES, MIN_RMS, MAX_ZERO_RATIO, CLIP_AMPLITUDE, MIN_NON_SILENT_RATIO = set_configuration_for_dataset_loading_cleaning_segmentation()
    
    create_output_directory(OUTPUT_DIR)

    wav_files = sorted(glob.glob(os.path.join(DATASET_DIR, "**", "*.wav"), recursive=True))

    records = []
    saved_count = 0
    rejected_count = 0

    # pbar = tqdm(total=len(wav_files), desc="Processing files")
 
    # for wav_path in tqdm(wav_files[0:3], desc="Processing files",dynamic_ncols=True,leave=True):
    for wav_path in tqdm(wav_files,desc="Processing files",dynamic_ncols=True,leave=True):
        try:
            wav, sr = load_audio(wav_path, TARGET_SR)

            # Trim silence from start/end
            wav_trimmed = trim_silence(wav, threshold=SILENCE_THRESHOLD)

            if wav_trimmed is None:
                rejected_count += 1
                records.append({
                    "source_file": wav_path,
                    "segment_index": None,
                    "segment_file": None,
                    "status": "rejected_full_silence"
                })
                continue

            # Split into fixed 10s crops
            segments = segment_audio(wav_trimmed, SEGMENT_SAMPLES)

            if len(segments) == 0:
                rejected_count += 1
                records.append({
                    "source_file": wav_path,
                    "segment_index": None,
                    "segment_file": None,
                    "status": "rejected_too_short"
                })
                continue

            base_name = Path(wav_path).stem

            for seg_idx, seg in segments:
                keep, stats = is_valid_segment(seg, SILENCE_THRESHOLD, MIN_RMS, MAX_ZERO_RATIO, MIN_NON_SILENT_RATIO)

                seg_filename = f"{base_name}_seg{seg_idx:04d}.wav"
                seg_filepath = os.path.join(OUTPUT_DIR, seg_filename)

                if keep:
                    torchaudio.save(seg_filepath, seg.cpu(), TARGET_SR)
                    saved_count += 1
                    status = "saved"
                else:
                    rejected_count += 1
                    status = f"rejected_{stats['reason']}"

                records.append({
                    "source_file": wav_path,
                    "segment_index": seg_idx,
                    "segment_file": seg_filepath if keep else None,
                    "status": status,
                    "rms": stats["rms"],
                    "zero_ratio": stats["zero_ratio"],
                    "max_amp": stats["max_amp"],
                    "non_silent_ratio": stats["non_silent_ratio"],
                    "duration_sec": SEGMENT_SEC
                })

        except Exception as e:
            rejected_count += 1
            records.append({
                            "source_file": wav_path,
                            "segment_index": None,
                            "segment_file": None,
                            "status": f"error: {str(e)}"
                        })
        # pbar.update(1)

    print("Saved segments:", saved_count)
    print("Rejected items:", rejected_count)
    # pbar.close()  

    #Create Metadata CSV
    df_segments = pd.DataFrame(records)
    metadata_csv = os.path.join(OUTPUT_DIR, "segments_metadata.csv")
    df_segments.to_csv(metadata_csv, index=False)

    print("Metadata saved to:", metadata_csv)
    print(df_segments["status"].value_counts())

    print("Dataset loading, cleaning & segmentation completed successfully.")


########################################################
# Train Test Split
########################################################

def run_train_test_split():
    """
    Loads the metadata CSV, creates the saved dataframe, prints the total number of audio segments, the total number of clean saved segments, and the unique recordings.
    Splits the recordings into train, val, and test sets in the ratio of 85:5:10.
    Assigns the split to each segment and prints the split segments count.
    Creates separate dataframes for train, val, and test and saves them to CSV.
    """
    #Load the metadata CSV
    
    metadata_csv = "dataset_segments_10s/segments_metadata.csv"
    df = pd.read_csv(metadata_csv)
    df_saved = df[df["status"] == "saved"].copy()
    print("Total Audio segments: ", len(df))
    print("Total Clean Saved segments: ", len(df_saved))
    print("Unique recordings:", df_saved["source_file"].nunique())

    unique_recordings = sorted(df_saved["source_file"].unique())

    ## Split recordings: Train 85%, Val 5%, Test 10%
    RANDOM_SEED = 42
    # First split: 90% train+val, 10% test
    train_val_recs, test_recs = train_test_split(
                                                    unique_recordings,
                                                    test_size=0.10,
                                                    random_state=RANDOM_SEED,
                                                    shuffle=True
                                                )

    # Second split: from remaining 90%, take 5/90 = 0.055555... as val
    val_ratio_relative = 0.05 / 0.90

    train_recs, val_recs = train_test_split(
                                                train_val_recs,
                                                test_size=val_ratio_relative,
                                                random_state=RANDOM_SEED,
                                                shuffle=True
                                            )

    print("\nTrain recordings:", len(train_recs))
    print("Val recordings:", len(val_recs))
    print("Test recordings:", len(test_recs))

    ## Assign split to each segment

    train_set = set(train_recs)
    val_set = set(val_recs)
    test_set = set(test_recs)

    df_saved["split"] = df_saved["source_file"].apply(lambda x: assign_split(x, train_set, val_set, test_set))

    print("Split Segments Count:\n",  df_saved["split"].value_counts())

    ## Create Seperate Dataframes for Train, Test, Val
    df_train = df_saved[df_saved["split"] == "train"].copy()
    df_val = df_saved[df_saved["split"] == "val"].copy()
    df_test = df_saved[df_saved["split"] == "test"].copy()

    ## Save the dataframes to CSV
    
    split_dir = "splits"
    os.makedirs(split_dir, exist_ok=True)

    train_csv = os.path.join(split_dir, "train_segments.csv")
    val_csv = os.path.join(split_dir, "val_segments.csv")
    test_csv = os.path.join(split_dir, "test_segments.csv")
    all_csv = os.path.join(split_dir, "all_segments_with_split.csv")

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    df_saved.to_csv(all_csv, index=False)
    print("Saved:")

    print("train test val split completed successfully.")

########################################################
# Tokenizer Loading + Token Caching
########################################################

def run_tokenizer_loading_token_caching():
    """
    Loads the tokenizer, tokenizes the audio segments, and saves the tokens to a CSV.
    """
    #Setting up the configuration for tokenizer loading and token caching
    TRAIN_CSV, VAL_CSV, TEST_CSV, TOKEN_DIR, TARGET_SR = set_configuration_for_tokenization()
    
    #Loading the device
    device = load_device("mps")

    #Loading the CSV files for train, val, and test
    df_train = pd.read_csv(TRAIN_CSV)
    df_val   = pd.read_csv(VAL_CSV)
    df_test  = pd.read_csv(TEST_CSV)

    # df_train = df_train.head().copy()
    # df_val   = df_val.head().copy()
    # df_test  = df_test.head().copy()

    #Loading the Encodec Tokenizer from MusicGen
    m = load_musicgen_model("facebook/musicgen-small")
    cm = load_compression_model(model = m)
    cm = cm.to(device)
    print(f"Loaded the {type(cm)} Compression model and is in the device {next(cm.parameters()).device}")

    # Cache train/ val/ test tokens
    df_train_tokens = cache_tokens_for_split(df_train, "train", cm, TOKEN_DIR, TARGET_SR, device)
    df_val_tokens   = cache_tokens_for_split(df_val, "val", cm, TOKEN_DIR, TARGET_SR, device)
    df_test_tokens  = cache_tokens_for_split(df_test, "test", cm, TOKEN_DIR, TARGET_SR, device)

    #Saving token metadata CSVs
    token_meta_dir = os.path.join(TOKEN_DIR, "metadata")
    os.makedirs(token_meta_dir, exist_ok=True)

    train_token_csv = os.path.join(token_meta_dir, "train_tokens.csv")
    val_token_csv   = os.path.join(token_meta_dir, "val_tokens.csv")
    test_token_csv  = os.path.join(token_meta_dir, "test_tokens.csv")
    all_token_csv   = os.path.join(token_meta_dir, "all_tokens.csv")

    df_train_tokens.to_csv(train_token_csv, index=False)
    df_val_tokens.to_csv(val_token_csv, index=False)
    df_test_tokens.to_csv(test_token_csv, index=False)

    df_all_tokens = pd.concat([df_train_tokens, df_val_tokens, df_test_tokens], ignore_index=True)
    df_all_tokens.to_csv(all_token_csv, index=False)

    print("Saved token metadata CSVs")

    print("Tokenizer loading & token caching completed successfully.")


########################################################
# Pretrained model loading & Baseline evaluation for Carnatic continuation
#######################################################

def run_baseline_evaluation():
    """
    Loads the pretrained model, evaluates the baseline performance.
    Loads the device, pretrained model, compression model, and the baseline list.
    Loads the token metadata CSV, filters the saved tokens, and evaluates the baseline performance.
    Loads the token file, decodes the original 10-second segment from cached tokens, and generates the continuation audio.
    Saves the continuation audio as .wav file and displays the audio.
    """


    device = load_device("cpu")

    m = load_musicgen_model("facebook/musicgen-small")
    cm = load_compression_model(model = m)
    cm = cm.to(device)
    m.lm = m.lm.to(device)
    m.compression_model = m.compression_model.to(device)
    m.lm.condition_provider = m.lm.condition_provider.to(device)
    m.device = torch.device(device)


    #Loading the token token files from metadata CSV for the baseline list
 
    baseline_list = ['dataset_segments_10s/Aarenduraghavanai_seg0000.wav',
                 'dataset_segments_10s/Aarenduraghavanai_seg0001.wav',
                 'dataset_segments_10s/Aarenduraghavanai_seg0002.wav',
                 'dataset_segments_10s/Aarenduraghavanai_seg0003.wav',
                 'dataset_segments_10s/Aarenduraghavanai_seg0004.wav']
 
    TOKEN_META_CSV = "dataset_tokens_10s/metadata/train_tokens.csv"
    df_train_tokens = pd.read_csv(TOKEN_META_CSV)

    df_train_tokens = df_train_tokens[df_train_tokens["status"] == "saved"].copy()
    print("Available cached train token files:", len(df_train_tokens))

    for i in baseline_list:
        # print ("*************************\n")
        # sample_token_file = df_train_tokens[df_train_tokens["segment_file"]==i]["token_file"]
        # print("Using token file:", sample_token_file)

        matches = df_train_tokens[df_train_tokens["segment_file"] == i]
        if len(matches) == 0:
            print(f"No token file found for: {i}")
            continue

        sample_token_file = matches["token_file"].iloc[0]
        print("Using token file:", sample_token_file)

        payload = torch.load(sample_token_file)
        # print(payload.keys())

        codes = payload["codes"]
        scale = payload["scale"]
        segment_path = payload["segment_path"]

        # print("codes shape:", codes.shape)
        # print("scale:", scale)
        # print("segment path:", segment_path)

    #Decoding the original 10-second segment from cached tokens
        with torch.no_grad():
            codes_batched = codes.unsqueeze(0).to(device)   # [1, K, T]

            if scale is not None:
                if torch.is_tensor(scale):
                    scale_batched = scale.unsqueeze(0).to(device) if scale.ndim == 0 else scale.to(device)
                else:
                    scale_batched = None
            else:
                scale_batched = None

            wav_orig = cm.decode(codes_batched, scale_batched)

        # print("Decoded original shape:", wav_orig.shape)
        # display(Audio(wav_orig[0, 0].detach().cpu().numpy(), rate=m.sample_rate))

    #High-level continuation API in CPU
        device = load_device("cpu")
        # print("device:", device)
        m = load_musicgen_model("facebook/musicgen-small")
        lm = m.lm.to(device)
        cm = load_compression_model(model = m)
        cm = cm.to(device)
        m.lm.condition_provider = m.lm.condition_provider.to(device)
        m.device = torch.device(device)

        wav_ref, sr_ref = torchaudio.load(segment_path)

        if wav_ref.shape[0] > 1:
            wav_ref = wav_ref.mean(dim=0, keepdim=True)

        if sr_ref != m.sample_rate:
            wav_ref = torchaudio.functional.resample(wav_ref, sr_ref, m.sample_rate)
            sr_ref = m.sample_rate

        wav_ref = wav_ref.unsqueeze(0).to(device)

        # print("wav_ref device:", wav_ref.device)
        # print("wrapper device:", m.device)
        # print("lm device:", next(m.lm.parameters()).device)
        # print("compression model device:", next(m.compression_model.parameters()).device)

        m.set_generation_params(duration=20)

        wav_cont = m.generate_continuation(
                                            wav_ref,
                                            sr_ref,
                                            descriptions=None
                                        )

        # print("Continuation output shape:", wav_cont.shape)

        # display(Audio(wav_cont[0, 0].detach().cpu().numpy(), rate=sr_ref))
        
 
        out_dir = "examples/baseline_continuation"
        os.makedirs(out_dir, exist_ok=True)
        name = Path(i).stem  # removes folder + .wav
    
        save_the_audio(os.path.join(out_dir, f"{name}_original_10s"),wav_ref,m)
        save_the_audio(os.path.join(out_dir, f"{name}_continued_20s"),wav_cont,m)

    print("Pretrained model loading & Baseline evaluation for Carnatic continuation completed successfully.")

########################################################
# LoRA finetuning for Carnatic continuation on Cached RVQ Tokens
#######################################################

def run_lora_finetuning():
    """
    Loads the configuration for LoRA finetuning, 
    builds the train and val datasets, 
    loads the pretrained model, 
    freezes the base LM, 
    applies LoRA to the linear layers, 
    initializes the optimizer, 
    runs the training loop, 
    saves the training history,
    and prints the completion message.
    """

    LORA_RANK, LORA_ALPHA, LORA_DROPOUT = set_lora_configuration()
    BATCH_SIZE, GRAD_ACCUM_STEPS, NUM_EPOCHS, LR, WEIGHT_DECAY, MAX_GRAD_NORM, PATIENCE = set_lora_training_configuration()
    USE_DEBUG_SUBSET, DEBUG_TRAIN_N, DEBUG_VAL_N = set_lora_debug_configuration()
    TOKEN_META_DIR, TRAIN_TOKENS_CSV, VAL_TOKENS_CSV = set_lora_token_configuration()

    CHECKPOINT_DIR = create_lora_checkpoint_directory()

    train_loader, val_loader = build_train_val_datasets(TRAIN_TOKENS_CSV, VAL_TOKENS_CSV, USE_DEBUG_SUBSET, DEBUG_TRAIN_N, DEBUG_VAL_N, BATCH_SIZE)

    m = load_musicgen_model("facebook/musicgen-small")
    lm = m.lm

    freeze_base_lm(lm)

    
    num_replaced = apply_lora_recursively(
                                            lm,
                                            rank=LORA_RANK,
                                            alpha=LORA_ALPHA,
                                            dropout=LORA_DROPOUT
                                        )

    print("Linear layers are wrapped with LoRA:", num_replaced)

    device = load_device("mps")
    lm = lm.to(device)

    try_enable_gradient_checkpointing(lm)

    optimizer = initialize_optimizer(lm, LR, WEIGHT_DECAY)

    history = run_training_loop(lm, optimizer, train_loader, val_loader, NUM_EPOCHS, GRAD_ACCUM_STEPS, MAX_GRAD_NORM, CHECKPOINT_DIR, PATIENCE, device)

    save_training_history(CHECKPOINT_DIR, history)


    print("Fine-tuning using LoRA completed successfully.")

########################################################
# Post finetuning Evaluation
#######################################################

def run_post_finetuning_evaluation():
    device = load_device("cpu")

    LORA_RANK, LORA_ALPHA, LORA_DROPOUT = set_lora_configuration()
    TEST_TOKENS_CSV, LORA_BEST_CKPT, EVAL_DIR, UNCOND_SEEDS, NUM_TEST_CLIPS, CONT_TOTAL_DURATION = set_post_finetuning_evaluation_configuration()

    #Load test metadata and choose fixed clips
    df_test = pd.read_csv(TEST_TOKENS_CSV)
    df_test = df_test[df_test["status"] == "saved"].copy().reset_index(drop=True)

    df_test = df_test.sample(50, random_state=42)
    # print("Available test clips:", len(df_test))

    # fixed sample for reproducibility
    random.seed(42)
    test_indices = random.sample(range(len(df_test)), k=min(NUM_TEST_CLIPS, len(df_test)))
    df_eval = df_test.iloc[test_indices].copy().reset_index(drop=True)

    # df_eval[["segment_file", "source_file"]].head()

    #Load Models
    baseline_model = load_baseline_model(device=device)
    finetuned_model = load_finetuned_lora_model(LORA_RANK, LORA_ALPHA, LORA_DROPOUT, LORA_BEST_CKPT, device=device)

    print("Both baseline and fine-tuned models loaded.")

    #Unconditional generation for baseline and finetuned models
    generate_unconditional_audio(baseline_model, finetuned_model, EVAL_DIR, UNCOND_SEEDS)

    #Audio continuation for baseline and finetuned models
    generate_audio_continuation(baseline_model, finetuned_model, df_eval, EVAL_DIR, CONT_TOTAL_DURATION, device)

    print("Post finetuning generation completed successfully.")



########################################################
# Quantitative Evaluation
#######################################################

def run_finetuning_ce_loss_visualization():
    """
    Runs the finetuning validation CE loss metrics visualization.
    Finds the best validation epoch and prints the metrics.
    Plots the training and validation CE loss metrics.
    Saves the plot as a PNG file.
    Displays the plot.
    """

    #Path to your finetuning metrics CSV
    finetuning_metrics_csv_path = "checkpoints_lora_musicgen_small/2/training_history.csv"

    #Load the finetuning metrics data
    metrics_table = load_finetuning_metrics_data(finetuning_metrics_csv_path)

    best_idx = metrics_table["val_loss"].idxmin() # Index of the best validation loss
    best_row = metrics_table.loc[best_idx] # Best validation loss row

    print("Best validation epoch:")
    print(f"Epoch: {int(best_row['epoch'])}")
    print(f"Train CE Loss: {best_row['train_loss']:.4f}")
    print(f"Validation CE Loss: {best_row['val_loss']:.4f}")

    plot_fintuning_ce_loss_visualization(metrics_table,best_row)


def run_boundary_continuation_evaluation():

    """
    """

    CONT_ROOT, BOUNDARY_METRIC_DIR, SR, BOUNDARY_SEC, BOUNDARY_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, N_MFCC = set_boundary_continuation_evaluation_configuration()

    check_value, bad_sr, bad_channels, df_audio_check = check_generated_audio_sr_mono(CONT_ROOT)
    if check_value == 1:
        print("All audio files are in the correct format.")
        sample_dirs = load_generated_audio_directories(CONT_ROOT)
        compute_boundary_metrics(sample_dirs, SR, BOUNDARY_SAMPLES, N_FFT, HOP_LENGTH, N_MELS, N_MFCC,BOUNDARY_METRIC_DIR)
    else:
        print("Some audio files are not in the correct format.")
        print("Bad sample rates:", bad_sr)
        print("Bad channels:", bad_channels)
        print("DF audio check:", df_audio_check)


def run_boundary_metrics_analysis():
    """
    Runs the boundary metrics analysis.
    """
    # Load per-sample boundary metrics
    boundary_csv = "evaluation_results/boundary_metrics/boundary_metrics_per_sample.csv"
    df_boundary = pd.read_csv(boundary_csv)

    # Build and save a comparison summary table
    summary = pd.DataFrame({
                            "metric": [
                                            "Mel boundary distance",
                                            "MFCC boundary distance",
                                            "Chroma boundary distance",
                                            "Onset boundary distance"
                                        ],
                            "baseline_mean": [
                                            df_boundary["baseline_mel_boundary_distance"].mean(),
                                            df_boundary["baseline_mfcc_boundary_distance"].mean(),
                                            df_boundary["baseline_chroma_boundary_distance"].mean(),
                                            df_boundary["baseline_onset_boundary_distance"].mean()
                                        ],
                            "finetuned_mean": [
                                            df_boundary["finetuned_mel_boundary_distance"].mean(),
                                            df_boundary["finetuned_mfcc_boundary_distance"].mean(),
                                            df_boundary["finetuned_chroma_boundary_distance"].mean(),
                                            df_boundary["finetuned_onset_boundary_distance"].mean()
                                        ]
                        })

    # Lower is better, so positive improvement means fine-tuned improved
    summary["absolute_improvement"] = summary["baseline_mean"] - summary["finetuned_mean"]
    summary["percent_improvement"] = (summary["absolute_improvement"] / summary["baseline_mean"]) * 100

    summary = summary.round(6)

    comparison_dir = "evaluation_results/boundary_metrics/comparison"
    os.makedirs(comparison_dir, exist_ok=True)

    summary_csv = os.path.join(comparison_dir, "baseline_vs_finetuned_summary.csv")
    summary.to_csv(summary_csv, index=False)

    # Create and per-sample improvement columns
    # This is useful to inspect whether the fine-tuned model helps consistently.
    df_compare = df_boundary.copy()

    df_compare["mel_improvement"] = (
                                        df_compare["baseline_mel_boundary_distance"] - df_compare["finetuned_mel_boundary_distance"]
                                    )

    df_compare["mfcc_improvement"] = (
                                        df_compare["baseline_mfcc_boundary_distance"] - df_compare["finetuned_mfcc_boundary_distance"]
                                    )

    df_compare["chroma_improvement"] = (
                                        df_compare["baseline_chroma_boundary_distance"] - df_compare["finetuned_chroma_boundary_distance"]
                                    )

    df_compare["onset_improvement"] = (
                                        df_compare["baseline_onset_boundary_distance"] - df_compare["finetuned_onset_boundary_distance"]
                                    )

    # df_compare.head()

    compare_csv = os.path.join(comparison_dir, "baseline_vs_finetuned_per_sample.csv")
    df_compare.to_csv(compare_csv, index=False)

    # print("Saved per-sample comparison to:", compare_csv)

    # Count how often fine-tuned beats baseline
    # Since lower is better, fine-tuned wins when: finetuned_distance < baseline_distance

    win_counts = pd.DataFrame({
                                "metric": [
                                            "Mel boundary distance",
                                            "MFCC boundary distance",
                                            "Chroma boundary distance",
                                            "Onset boundary distance"
                                        ],
                                "finetuned_better_count": [
                                                            (df_compare["finetuned_mel_boundary_distance"] < df_compare["baseline_mel_boundary_distance"]).sum(),
                                                            (df_compare["finetuned_mfcc_boundary_distance"] < df_compare["baseline_mfcc_boundary_distance"]).sum(),
                                                            (df_compare["finetuned_chroma_boundary_distance"] < df_compare["baseline_chroma_boundary_distance"]).sum(),
                                                            (df_compare["finetuned_onset_boundary_distance"] < df_compare["baseline_onset_boundary_distance"]).sum(),
                                                        ],
                                "total_samples": [len(df_compare)] * 4
                            })

    win_counts["finetuned_better_percent"] = (
                                                win_counts["finetuned_better_count"] / win_counts["total_samples"] * 100
                                            ).round(2)

    # win_counts

    win_csv = os.path.join(comparison_dir, "baseline_vs_finetuned_win_counts.csv")
    win_counts.to_csv(win_csv, index=False)
    
    print(win_counts)
    # print("Saved win-count table to:", win_csv)

    # Bar chart: mean baseline vs fine-tuned distances and save it as a PNG file
    barplot_path = os.path.join(comparison_dir, "baseline_vs_finetuned_barplot.png")

    plt.figure(figsize=(9, 5))
    x = np.arange(len(summary))
    width = 0.35

    plt.bar(x - width/2, summary["baseline_mean"], width, label="Baseline")
    plt.bar(x + width/2, summary["finetuned_mean"], width, label="Fine-tuned")

    plt.xticks(x, summary["metric"], rotation=20)
    plt.ylabel("Boundary Distance (lower is better)")
    plt.title("Baseline vs Fine-tuned Boundary Continuity Metrics")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(barplot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved bar chart to:", barplot_path)


    # plot and save per-sample : Positive values mean fine-tuned is better.
    improvement_cols = [
                        "mel_improvement",
                        "mfcc_improvement",
                        "chroma_improvement",
                        "onset_improvement"
                    ]

    improvement_plot_path = os.path.join(comparison_dir, "per_sample_improvement_plot.png")

    plt.figure(figsize=(10, 6))
    for col in improvement_cols:
        plt.plot(df_compare[col].values, marker="o", label=col)

    plt.axhline(0, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Test Sample Index")
    plt.ylabel("Improvement (Baseline - Fine-tuned)")
    plt.title("Per-sample Improvement in Boundary Metrics")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(improvement_plot_path, dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved per-sample improvement plot to:", improvement_plot_path)

    # Report Table
    report_table = summary.copy()
    report_table["baseline_mean"] = report_table["baseline_mean"].round(4)
    report_table["finetuned_mean"] = report_table["finetuned_mean"].round(4)
    report_table["absolute_improvement"] = report_table["absolute_improvement"].round(4)
    report_table["percent_improvement"] = report_table["percent_improvement"].round(2)

    report_table

def run_quantitative_evaluation():
    """
    Runs the finetuning validation CE loss metrics evaluation
    and then runs the boundary continuation evaluation.
    """

    # run_finetuning_ce_loss_visualization()

    # run_boundary_continuation_evaluation()

    run_boundary_metrics_analysis()