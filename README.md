# Fine-Tuning Gemma-2-2B-IT with GRPO for Enhanced Reasoning

This project provides a comprehensive pipeline for fine-tuning the `Google/gemma-2-2b-it` model using Generative Reward Post-Optimization (GRPO). The goal is to improve the model's mathematical reasoning capabilities, with a specific focus on the GSM8K benchmark. The project also includes a preliminary Supervised Fine-Tuning (SFT) step on the LIMO dataset to guide the model towards a desired response format.

## Project Overview

The core of this project is to leverage a two-stage fine-tuning process to enhance the reasoning abilities of the Gemma-2-2B-IT model:

1.  **Supervised Fine-Tuning (SFT) on LIMO:** The model is first fine-tuned on the LIMO dataset. This step teaches the model to generate responses in a specific XML-like format, with clear separation between the reasoning process and the final answer.

2.  **Generative Reward Post-Optimization (GRPO) on GSM8K:** The SFT-tuned model is then further trained using GRPO on the GSM8K dataset. This reinforcement learning step optimizes the model to produce not only well-formatted but also correct answers to mathematical word problems.

The project is structured to be modular and configurable, with clear separation of concerns between configuration, training, and evaluation.

## Key Features

- **Two-Stage Fine-Tuning:** Combines SFT and GRPO for robust and effective model optimization.
- **Parameter-Efficient Fine-Tuning (PEFT):** Utilizes LoRA for efficient training, reducing computational and memory requirements.
- **vLLM Integration:** Leverages the vLLM library for high-throughput inference during GRPO, significantly speeding up the training process.
- **Comprehensive Evaluation:** Includes a dedicated script to evaluate the fine-tuned model on the GSM8K test set, providing metrics on format correctness and answer accuracy.
- **Clear and Configurable:** All important parameters and configurations are centralized in a single `config.py` file for easy modification.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- An NVIDIA GPU with CUDA support (bfloat16 is recommended)
- A Hugging Face account and an access token with permissions for Gemma models

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Gemma-2-2B-IT-GRPO.git
    cd Gemma-2-2B-IT-GRPO
    ```

2.  **Set up the environment and install dependencies:**
    The project uses `uv` for environment and package management. The `install.sh` script automates the setup process.

    ```bash
    bash install.sh
    ```

3.  **Set your Hugging Face Token:**
    You need to set your Hugging Face token as an environment variable.

    ```bash
    export HF_TOKEN="your-hugging-face-token"
    ```

## Usage

The project provides three main scripts, which can be executed using the shortcuts defined in `pyproject.toml`.

### 1. SFT and GRPO Training (`gemma-instruct-grpo.py`)

This script runs the full two-stage fine-tuning pipeline: SFT on LIMO followed by GRPO on GSM8K.

```bash
pdm run instruct
```

This will:
- Load the base `Google/gemma-2-2b-it` model.
- Perform SFT on the LIMO dataset.
- Perform GRPO on the GSM8K dataset.
- Save the final fine-tuned model to the `gemma-2-2b-it-grpo` directory.

### 2. GRPO-Only Training (`gemma-grpo.py`)

If you want to run only the GRPO training on the base model (without the initial SFT step), you can use this script.

```bash
pdm run start
```

This will:
- Load the base `Google/gemma-2-2b-it` model.
- Perform GRPO training on the GSM8K dataset.
- Save the final fine-tuned model to the `gemma-2-2b-it-grpo` directory.

### 3. Evaluation (`gsm8k-eval.py`)

After fine-tuning, you can evaluate the model's performance on the GSM8K test set.

```bash
pdm run eval
```

This will:
- Load the fine-tuned model from the `gemma-2-2b-it-grpo` directory.
- Run inference on the GSM8K test set.
- Print a report with metrics on format correctness and answer accuracy.
- Save a detailed log of the evaluation in `records.json`.

## Configuration

All key parameters can be modified in the `config.py` file. This includes:

- `MODEL_NAME`: The base model to be used.
- `OUTPUT_MODEL`: The directory where the fine-tuned model will be saved.
- `max_prompt_length` and `max_completion_length`: The maximum lengths for prompts and completions.
- System prompts and response formats.

The training arguments for SFT and GRPO can be found in `gemma-instruct-grpo.py` and `gemma-grpo.py` respectively, within the `SFTConfig` and `GRPOConfig` objects.

## File Descriptions

- **`config.py`**: Central configuration file.
- **`gemma-grpo.py`**: Script for GRPO-only training.
- **`gemma-instruct-grpo.py`**: Script for the full SFT + GRPO training pipeline.
- **`gsm8k-eval.py`**: Script for evaluating the fine-tuned model.
- **`install.sh`**: Installation script for setting up the environment and dependencies.
- **`pyproject.toml`**: Project configuration and script definitions.
- **`README.md`**: This file.