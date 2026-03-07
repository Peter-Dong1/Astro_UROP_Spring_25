# Archived Exploratory Code

This directory contains code from early experimental phases (January-February 2026) that are **not part of the final results**.

---

## Why This Code Was Archived

During Phase 1 of the project, deep learning approaches (RNNs, Transformers, LSTM Autoencoders) were explored for analyzing eROSITA light curves. These experiments were ultimately not used in the final analysis for several reasons:

1. **Data characteristics**: Light curves have only 10-20 time points, insufficient for deep learning
2. **Interpretability**: Latent space representations difficult to interpret astronomically
3. **Technical issues**: Posterior collapse in VAEs, limited GPU availability
4. **Better alternative**: Statistical feature extraction proved more effective and interpretable

**The project pivoted to statistical approaches in March 2026**, which form the basis of the final results in `ORGANIZED_RESULTS/`.

---

## Directory Structure

### deep_learning_experiments/

**RNN-based Variational Autoencoder (VAE)**:
- `RNN_9_model.py` - GRU encoder/decoder model definition
- `train_rnn.py` - Training script with Weights & Biases integration
- `RNN_train.py` - Alternative training implementation
- `test_rnn.py` - Evaluation script
- `test_model_9.py` - Model testing utilities
- `cont_9.py` - Continuation/fine-tuning script

**Transformer-based VAE**:
- `trans_model.py` - Transformer encoder with multi-head attention
- `test_trans.py` - Training and evaluation

**Visualization & Utilities**:
- `plotmodel.py` - Model output plotting
- `plotmodelerror.py` - Error analysis plots

**SLURM Job Scripts** (for cluster execution):
- `rnn.slurm` - RNN VAE training job
- `trans.slurm` - Transformer VAE training job
- `feature.slurm` - Early feature extraction job
- `helper.slurm` - Helper job script
- `plot_rnn.slurm` - Plot generation job

**Key Architecture Details**:
- Input: 9 features per timestep (3 energy bands × 3 values: RATE, ERRM, ERRP)
- Encoder: GRU/Transformer → latent space (typically 8-16 dimensions)
- Decoder: GRU/ResNet blocks → reconstruct light curves
- Loss: ELBO (Evidence Lower Bound) or Poisson NLL
- Outlier detection: Isolation Forest on latent space

### notebooks/

Contains Jupyter notebook checkpoints from interactive analysis sessions:

**.ipynb_checkpoints/**
- `LSTM_AutoEncoder-checkpoint.ipynb` - LSTM-based autoencoder experiments
- `Raw Data Clustering-checkpoint.ipynb` - Clustering on raw light curve data
- `Statistcal Clustering-checkpoint.ipynb` - Statistical approach experiments (precursor to Phase 2)
- `ML Approach Standard-checkpoint.ipynb` - Standard ML methods
- `light_curves-checkpoint.ipynb` - Light curve visualization and exploration
- `AddErrors-checkpoint.ipynb` - Error propagation experiments
- Various plots and utility scripts

---

## Technical Details

### RNN VAE Architecture

```
Light Curve (N×9) → GRU Encoder → μ, log(σ²) → Sample z ~ N(μ,σ²)
                                                      ↓
                                              GRU Decoder → Reconstructed LC
```

**Training**:
- Optimizer: Adam
- Loss: Reconstruction + KL divergence
- Posterior collapse mitigation: KL annealing

### Transformer VAE Architecture

```
Light Curve (N×9) → Positional Encoding → Multi-Head Attention → μ, log(σ²)
                                                                        ↓
                                                                Sample z ~ N(μ,σ²)
                                                                        ↓
                                                                ResNet Decoder → Reconstructed LC
```

**Key Features**:
- Self-attention across time points
- Positional encoding for temporal information
- ResNet decoder blocks

---

## Why These Approaches Didn't Work

### Challenge 1: Sparse Data
- Only 10-20 time points per light curve
- Deep learning typically requires hundreds+ of time points
- Too little data to learn complex temporal patterns

### Challenge 2: Variable Length Sequences
- Light curves have different lengths
- Required padding/masking, complicated training
- Statistical features handle this naturally

### Challenge 3: Interpretability
- Latent space dimensions hard to interpret
- Astronomers need physically meaningful features
- Statistical features (bexvar, lag1_autocorr, etc.) have clear meanings

### Challenge 4: Posterior Collapse
- VAEs often collapsed to prior distribution
- Reconstruction ignored, model learned nothing useful
- Difficult to tune β-VAE tradeoff

### Challenge 5: Limited Computational Resources
- GPU availability on cluster limited
- Long training times (hours to days)
- Statistical approach much faster (minutes)

---

## What Worked Instead

**Phase 2: Statistical Feature Extraction** (March 2026 onwards)

The project pivoted to:
1. Extract 10 statistical features per light curve
2. Use HDBSCAN clustering in feature space
3. Compute cosine similarity for ranking sources

**Results**: All final deliverables in `ORGANIZED_RESULTS/`

See:
- `../ORGANIZED_RESULTS/` - Final organized results
- `../PROJECT_STATUS.md` - Complete project history
- `../ORGANIZED_RESULTS/6_documentation/RUN_HISTORY.md` - Detailed timeline

---

## Could This Code Be Useful?

### Maybe, For:
1. **Different datasets**: With longer light curves (100+ points), deep learning might work
2. **Transfer learning**: Pre-train on simulated data, fine-tune on real data
3. **Hybrid approaches**: Use VAE latent space as additional features for clustering
4. **Anomaly detection**: VAE reconstruction error as anomaly score

### Probably Not For:
- eRASS1 data with current sparsity
- Primary analysis (statistical approach is better)
- Quick exploratory analysis (too slow to train)

---

## Running Archived Code (Not Recommended)

If you really want to run these experiments:

### Setup:
```bash
conda create -n dl_env python=3.9
conda activate dl_env
pip install torch torchvision  # PyTorch
pip install wandb  # Weights & Biases for tracking
pip install numpy pandas astropy matplotlib
```

### Train RNN VAE:
```bash
# On cluster
sbatch rnn.slurm

# Or locally (slow without GPU)
python train_rnn.py --epochs 100 --latent-dim 16
```

### Train Transformer VAE:
```bash
python test_trans.py --epochs 50 --batch-size 64
```

**Note**: You'll need access to the FITS files at `/pool001/rarcodia/eROSITA_public/data/eRASS1_lc_rebinned` on the cluster.

---

## Historical Context

These experiments were valuable learning experiences that informed the final approach:

- **Learned**: Light curve sparsity is a fundamental limitation
- **Learned**: Interpretability matters for astronomical applications
- **Learned**: Sometimes simpler (statistical) methods are better

The statistical approach in Phase 2 directly addressed the limitations discovered in Phase 1.

---

## For Future Reference

If you're working on similar time series analysis problems:

**Use deep learning when**:
- You have long sequences (100+ time points)
- Complex temporal patterns exist
- Large training dataset available
- End-to-end learning is valuable

**Use statistical features when**:
- Sequences are short (<50 points)
- Interpretability is important
- Computational resources limited
- Domain knowledge suggests relevant features

For eROSITA light curves, statistical features were the right choice.

---

**Bottom line**: This code documents the experimental process but is not part of the final results. The working pipeline is in `ORGANIZED_RESULTS/5_code/`.
