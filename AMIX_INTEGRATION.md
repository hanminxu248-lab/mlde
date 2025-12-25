# AMix Integration Guide

This document describes how to use the AMix language model with the MLDE (Machine Learning Directed Evolution) framework.

## Overview

The AMix model has been integrated as an alternative to ESM2 for:
1. **Mutation model** - Predicting mutations at masked positions
2. **Fitness predictor encoder** - Encoding sequences for fitness prediction
3. **Oracle/Landscape model** - Evaluating fitness in simulation

## Requirements

Additional dependencies required for AMix:
- `hydra-core`
- `omegaconf`

These are already included in `requirements.txt`.

## Usage

### 1. Running Directed Evolution with AMix

To use AMix for directed evolution, add the `--use_amix` flag and provide the checkpoint path:

```bash
python scripts/run_discrete_de.py \
    --use_amix \
    --amix_ckpt_path /path/to/AMix-1-main/checkpoints/model.ckpt \
    --amix_oracle_ckpt_path /path/to/AMix-1-main/checkpoints/model.ckpt \
    --task AAV \
    --wt "PROTEIN_SEQUENCE" \
    --wt_fitness 0.5 \
    --predictor_ckpt_path /path/to/predictor.ckpt \
    --n_steps 100 \
    --population 128 \
    --save_name results.csv
```

**Arguments:**
- `--use_amix`: Enable AMix model (default: False, uses ESM2)
- `--amix_ckpt_path`: Path to AMix checkpoint file (required when `--use_amix` is set)
- `--amix_oracle_ckpt_path`: Path to AMix checkpoint for oracle model (optional, uses ESM1b if not provided)

### 2. Training Decoder with AMix Encoder

To train a fitness predictor decoder with AMix as the encoder:

```bash
python scripts/train_decoder.py \
    --encoder_type amix \
    --amix_ckpt_path /path/to/AMix-1-main/checkpoints/model.ckpt \
    --data_file /path/to/training_data.csv \
    --dataset_name my_dataset \
    --dec_hidden_dim 512 \
    --batch_size 128 \
    --num_epochs 30 \
    --output_dir ./exps
```

**Arguments:**
- `--encoder_type`: Choose encoder type (`esm2` or `amix`, default: `esm2`)
- `--amix_ckpt_path`: Path to AMix checkpoint file (required when `--encoder_type amix`)

### 3. Backward Compatibility

All existing ESM2 functionality remains intact. To use ESM2 (default behavior), simply omit the AMix-specific arguments:

```bash
# Uses ESM2 by default
python scripts/run_discrete_de.py \
    --task AAV \
    --wt "PROTEIN_SEQUENCE" \
    --wt_fitness 0.5 \
    --predictor_ckpt_path /path/to/predictor.ckpt \
    --pretrained_mutation_name facebook/esm2_t12_35M_UR50D \
    --n_steps 100 \
    --population 128 \
    --save_name results.csv
```

## AMix Model Structure

The AMix model is located in `AMix-1-main/` and consists of:
- **ProfileBFN**: Main Bayesian Flow Network wrapper
- **EsmForBFN**: ESM-based architecture adapted for BFN
- **FlashEsmModel**: Flash attention optimized components

### Checkpoint Structure

AMix expects checkpoints in the following structure:
```
/path/to/checkpoint/
├── .hydra/
│   └── config.yaml        # Hydra configuration
└── checkpoints/
    └── model.ckpt         # Model weights
```

## Implementation Details

### Utility Functions
A shared utility module (`de/samplers/models/amix_utils.py`) provides common functions for loading AMix models:
- `load_amix_config()`: Load configuration from checkpoint
- `load_amix_model()`: Load and instantiate AMix model

This reduces code duplication and improves maintainability across all AMix components.

### Tokenizer
AMix uses the same tokenizer as ESM2 (`facebook/esm2_t30_150M_UR50D`), which simplifies integration and ensures compatibility.

### Model Forward Pass
The AMix model expects:
- Input: One-hot encoded token embeddings
- Timestep: Set to 1.0 for inference (fully denoised)
- Output: Logits and hidden states compatible with ESM2 interface

### Classes Added

1. **`de/samplers/models/amix_utils.py`** (New utility module)
   - `load_amix_config()`: Shared config loading
   - `load_amix_model()`: Shared model loading

2. **`de/samplers/models/amix.py`**
   - `AMix`: Main wrapper for mutation model
   - `ModelOutput`: Named tuple for model outputs

3. **`de/predictors/attention/module.py`**
   - `AMix_Attention`: Fitness predictor with AMix encoder
   - `AMixDecoderModule`: Training module for AMix-based predictor

4. **`de/predictors/oracle.py`**
   - `AMix_Attention1d`: AMix encoder for oracle
   - `AMix_Landscape`: Fitness landscape simulation with AMix

## Notes

- The decoder architecture (Attention1d + dense layers) remains the same for both ESM2 and AMix
- Device placement is handled automatically
- AMix model requires a valid Hydra config file in the checkpoint directory
- For oracle/landscape models, pre-trained decoder weights are still required in `./landscape_params/amix_landscape/<task>/decoder.pt`

## Troubleshooting

### Config File Not Found
If you get an error about missing config file:
```
FileNotFoundError: Config file not found at /path/to/.hydra/config.yaml
```

Ensure your checkpoint path points to a directory with the following structure:
- The checkpoint file should be in a `checkpoints/` subdirectory
- The `.hydra/config.yaml` should be two levels up from the checkpoint

### Missing Dependencies
If you encounter import errors for `hydra` or `omegaconf`:
```bash
pip install hydra-core omegaconf
```
