# LLM Training with LoRA — Part 1: Continued Pre-training

Adapting a pre-trained language model to the Sherlock Holmes corpus using LoRA/QLoRA.

## Project Structure

```
assignment1/
├── notebooks/
│   ├── 00_report.ipynb               # Report (GPU info, assumptions, mistakes, AI usage)
│   ├── 01_memory_estimation.ipynb    # GPU memory estimation table (no GPU needed)
│   ├── 02_data_preprocessing.ipynb   # Download, clean, tokenize Holmes corpus
│   ├── 03_training.ipynb             # Continued pre-training with LoRA
│   ├── 04_evaluation.ipynb           # Perplexity evaluation (base vs adapted)
│   ├── 05_experiments.ipynb          # Ablation experiments (rank, LR, modules, scheduler)
│   └── *.png                         # All plots and figures
├── requirements.txt                  # Python dependencies
├── data/
│   ├── raw/                          # Downloaded Gutenberg texts
│   └── processed/                    # Tokenized & chunked training/validation data
└── outputs/
    ├── baseline/                     # Baseline training outputs & adapter
    ├── experiments/                  # Ablation experiment outputs
    └── evaluation_results.json       # Final evaluation metrics
```

## Execution Order

0. **00_report.ipynb** — The report. Contains GPU info, assumptions, mistakes log, and AI usage. Export to PDF for submission.
1. **01_memory_estimation.ipynb** — Run locally (no GPU needed). Produces the memory analysis table for the report.
2. **02_data_preprocessing.ipynb** — Run locally. Downloads and preprocesses the Sherlock Holmes corpus.
3. **03_training.ipynb** — Run on GPU (cluster or Colab). Trains the baseline LoRA adapter.
4. **04_evaluation.ipynb** — Run on GPU. Computes perplexity for base and adapted models.
5. **05_experiments.ipynb** — Run on GPU. Runs ablation experiments (LoRA rank, LR, target modules, scheduler).

## Setup

```bash
pip install -r requirements.txt
```

## GPU Used

All training and evaluation was performed locally on an **NVIDIA RTX A1000 Laptop GPU (6 GB VRAM)** with CUDA 12.8 and PyTorch 2.11.0+cu128. QLoRA + gradient checkpointing kept peak memory at ~3.9 GB.

## Model

Default: `Qwen/Qwen2.5-1.5B` (configurable in each notebook)
