# FINE TUNING FOUNDATIONAL MUSIC MODEL #[MusicGen](https://dl.acm.org/doi/10.5555/3666122.3668188) FOR CARNATIC MUSIC CONTIUATION

## Course Work "INFO 697 Capstone Research"

**Project Motivation & Description**<br>
Foundation music models like MusicGen are trained predominantly on Western music, leaving culturally rich traditions such as Carnatic music underrepresented. This project fine-tunes MusicGen on ~10 hours of licensed Carnatic Music from the Indian Art Music Raga Recognition Dataset, using an audio-to-audio continuation framework, bypassing the need for scarce prompt-labeled Carnatic corpora. The goal is to generate stylistically coherent Carnatic continuations from short audio excerpts while preserving melodic and ornamental authenticity. Evaluation follows a hybrid approach, combining objective boundary distances (mel-spectrogram, chroma, MFCC, etc.) with human listening studies via a [Streamlit interface](https://capstone-user-evaluation-survey.streamlit.app).


## Dataset:
The audio dataset is currently not included in this repository due to its heavy file size.<br> 
Link to dataset:  <br>
[Indian Art Music Raga Recognition datasets](https://compmusic.upf.edu/datasets)<br>

Link to the sample dataset: <br>
[Sample dataset](https://drive.google.com/drive/folders/1hdnKrlbUHkPReDeCAvJrAnbq7juKEMda?usp=share_link)<br>

The Sample dataset consists of 5 Carnatic music recordings converted to .wav format from the actual dataset.


## Methodology:<br>
An overview of the methodology followed for this project is as follows.<br>
![Process Overview](images/project_overview.jpg)
<br>

For this project, I used Python in the Cursor IDE for coding. GPU-accelerated machine learning training and inference on macOS made available through Metal Performance Shaders (mps) of Apple silicon M4 chip was leveraged for finetuning. Listening studies were conducted through a Streamlit interface, and responses were recorded in Google Spreadsheets using the Google Drive API services.<br>


## Details on files & folders<br>
**This code repository is organised into the following key components:**<br>
| File / Folder | Description |
|---------------|-------------|
| `README.md` | The current file you are reading; gives an overview of the project. |
| `initial_environment_setup.md` | Step-by-step guide to fork AudioCraft for Apple Silicon (disabling xFormers) and set up the Python 3.9 virtual environment with all MusicGen dependencies. |
| `dataset_relocation_and_conversion.ipynb` | Converts the raw RagaDataset audio files into 32 kHz `.wav` format under `dataset_wav/` and reports total duration statistics. |
| `classes.py` | Defines the custom `CachedRVQDataset` (loads cached EnCodec RVQ tokens from CSV) and `LoRALinear` (low-rank adapter that wraps and freezes a base `nn.Linear` layer). |
| `helpers.py` | Central utility module containing all helper functions — device/model loading, audio cleaning & segmentation, train/val/test splitting, EnCodec token caching, LoRA injection/training/checkpointing, and boundary-continuation evaluation metrics (Mel, MFCC, Chroma, Onset). |
| `pipelines.py` | High-level orchestration layer that chains helper functions into runnable end-to-end stages (reproducibility check, EnCodec confirmation, dataset prep, tokenization, baseline eval, LoRA fine-tuning, post-finetuning eval, quantitative eval). |
| `fine-tuning.ipynb` | Main execution notebook that runs the full MusicGen-Small LoRA fine-tuning pipeline for Carnatic music continuation by sequentially invoking the stage functions from `pipelines.py` |
| `Reports/` | This section will be updated post completion of this project. |
| `images/` | This folder contains the image files used in `Readme.md`. |


> **Note:** Folders such as `dataset_wav/`, `dataset_segments_10s/`, `splits/`, `dataset_tokens_10s/`, `checkpoints_lora_musicgen_small/`, and `evaluation_results/` are generated automatically when the notebooks/pipelines are executed, and they are intentionally excluded from this repository via `.gitignore`. The raw `RagaDataset/` must be downloaded separately and placed at the project root before running `dataset_relocation_and_conversion.ipynb`.

## How to use this repository? <br>
Follow the steps below to reproduce the MusicGen-Small LoRA fine-tuning pipeline for Carnatic music continuation.

### 1. Clone the Repository

```bash
git clone https://github.com/sundarram1608/finetuning-musicgen-small-carnatic-continuation.git
cd finetuning-musicgen-small-carnatic-continuation
```

### 2. Set Up the Environment

Follow the detailed instructions in [`initial_environment_setup.md`](./initial_environment_setup.md) to:
- Fork and patch the AudioCraft repository for Apple Silicon compatibility.
- Install Python 3.9 via Homebrew.
- Create and activate the `musicgen3.9.2` virtual environment.
- Install all required dependencies (AudioCraft, torchcodec, ffmpeg, ipykernel, etc.).
- Register the Jupyter kernel for use in Cursor/ VS Code.

### 3. Download the Raw Dataset

Place the raw `RagaDataset/` folder at the project root. This folder is **not** included in the repository and must be sourced by obtaining access separately from [Indian Art Music Raga Recognition datasets](https://compmusic.upf.edu/datasets).

### 4. Convert the Raw Dataset to WAV

Open and run [`dataset_relocation_and_conversion.ipynb`](./dataset_relocation_and_conversion.ipynb) end-to-end. This will:
- Convert the raw audio files into 32 kHz `.wav` format.
- Save them under a newly created `dataset_wav/` folder.

### 5. Run the Fine-Tuning Pipeline

Open [`fine-tuning.ipynb`](./fine-tuning.ipynb) and select the `Python (venv_musicgen3.9.2)` kernel. Execute the cells sequentially, as each section corresponds to a stage in the pipeline:

> **Note:** Step 0 can be ignored as it's just a starter to ensure we can access the baseline MusicGen models

| Step | Stage | Description |
|------|-------|-------------|
| 0 | Hugging Face Examples | Sanity-check text-to-music generation using the HF MusicGen wrapper. |
| 1 | Reproducibility Check | Verify installed package versions and load MusicGen-Small. |
| 2 | Unconditional Generation & EnCodec Confirmation | Confirm local inference and EnCodec encode/decode work correctly. |
| 3 | Dataset Loading, Cleaning & Segmentation | Resample, trim silence, and split audio into 10-second segments. |
| 4 | Train / Val / Test Split | Generate split CSVs under `splits/`. |
| 5 | Tokenizer Loading & Token Caching | Encode segments into RVQ tokens and cache them under `dataset_tokens_10s/`. |
| 6 | Baseline Evaluation | Generate baseline continuations from pretrained MusicGen-Small. |
| 7 | LoRA Fine-Tuning | Train LoRA adapters on cached RVQ tokens; checkpoints saved under `checkpoints_lora_musicgen_small/`. |
| 8 | Post Fine-Tuning Evaluation | Generate continuations from the fine-tuned model and compare against baseline. |
| 9 | Quantitative Evaluation | Compute CE loss, perplexity, and boundary-continuation metrics (Mel, MFCC, Chroma, Onset). |

### 6. Review the Outputs

After execution, the following folders will be populated automatically:
- `dataset_wav/`, `dataset_segments_10s/`, `splits/`, `dataset_tokens_10s/` — processed data.
- `checkpoints_lora_musicgen_small/` — LoRA checkpoints and training history.
- `evaluation_results/` — generated audio continuations and quantitative metric CSVs.

### Notes
- The pipeline is built and tested on **macOS with Apple Silicon (M1/M2)**. Other platforms may work, but have not been verified.
- All stages are modular. So, you can re-run individual cells in `fine-tuning.ipynb` without restarting the entire pipeline, as long as upstream artifacts already exist.
- For debugging, set `USE_DEBUG_SUBSET = True` inside `set_lora_debug_configuration()` in `helpers.py` to train on a small subset.

### 7. Qualitative Evaluation (Listening Study)

In addition to the quantitative metrics computed, a qualitative evaluation is carried out through a human listening study. Participants compare baseline and fine-tuned continuations side-by-side and rate them on perceptual musicality, continuity, and authenticity.
The study is hosted via a Streamlit web interface, available here: [Listening Study Link](https://capstone-user-evaluation-survey.streamlit.app)

Results from this study will be summarized in the `Reports/` folder upon project completion.

## Credits:
I thank my mentor, Dr. Xiao Hu, for guidance and Meta for open-sourcing its MusicGen model.
