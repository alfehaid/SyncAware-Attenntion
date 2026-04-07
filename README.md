# SyncAware Attention

An attention mechanism for Digital Twin synchronization in cybersecurity applications.

This repository contains the source code, data generation scripts, and reproduction
materials for the paper:

> **SyncAware attention: an attention mechanism for digital twin synchronization in cybersecurity applications.**
> Haitham Saleh M. Alfehaid, Iftikhar Ahmad, Madini O. Alassafi.
> Department of Information Systems, Faculty of Computing and Information Technology,
> King Abdulaziz University, Jeddah, Saudi Arabia.

---

## Description

SyncAware Attention is a novel attention mechanism that extends the standard
scaled dot-product attention with a synchronization-quality term `G(S)`. The term
`G(S)` modulates the attention scores based on real-time synchronization metrics
(accuracy, normalized delay, reliability, and temporal consistency) so that the
model can down-weight features that come from out-of-sync or unreliable Digital
Twin (DT) sources.

The mechanism is integrated into a hybrid CNN + LSTM + Transformer model
(`DT-HybridNet`) and is evaluated on two intrusion-detection datasets: an
enhanced version of NSL-KDD and a synthetic Digital Twin dataset.

### Why this project is useful

Conventional attention mechanisms assume that all input tokens are equally
fresh and equally reliable. In Digital Twin environments this assumption breaks
down: sensor readings can be delayed by hundreds of milliseconds, communication
links can drop, and the virtual replica can drift from the physical asset.
SyncAware Attention is the first attention mechanism designed to make this
synchronization quality a first-class signal inside the model. On the
benchmarks reported in the paper it raises intrusion-detection accuracy from
89.23% (standard attention) to 93.45%, keeps accuracy above 82% even when
network delay exceeds one second, and adds less than 8% computational overhead
compared to standard scaled dot-product attention. This makes the mechanism a
practical drop-in upgrade for any DT-based cybersecurity model that has access
to synchronization metadata.

---

## Repository Structure

```
SyncAware-Attenntion/
|-- data/
|   |-- dataset_collector.py        # Generates raw DT-aware training samples
|   |-- generate_large_synthetic.py # Large-scale synthetic DT dataset generator
|   |-- validate_generated_data.py  # Sanity checks for generated batches
|   `-- dataset_summary.md          # Per-dataset feature counts and statistics
|-- models/
|   |-- dt_hybrid_net.py            # Full DT-HybridNet model + SyncAwareAttention
|   |-- dt_hybrid_simplified.py     # Lightweight version used by the trainer
|   |-- train_and_evaluate.py       # End-to-end training and evaluation pipeline
|   |-- quick_test.py               # Smoke test on a small subset
|   `-- GoogleColab/
|       |-- DT_HybridNet_Colab.ipynb  # Self-contained Colab reproduction notebook
|       `-- README.md                 # Colab quick-start guide
|-- preprocessing/
|   |-- dt_aware_preprocessor.py    # DT-aware feature engineering and scaling
|   `-- test_preprocessor.py        # Unit tests for the preprocessor
`-- README.md                       # This file
```

---

## Dataset Information

Two datasets are used in the paper. Both are released as a single archive
(`datasets.tar.gz`, ~47 MB) under the GitHub release tag `v1.0`:
<https://github.com/alfehaid/SyncAware-Attenntion/releases/tag/v1.0>

### 1. Enhanced NSL-KDD

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Original source | NSL-KDD, Canadian Institute for Cybersecurity, University of New Brunswick |
| Original URL  | <https://www.unb.ca/cic/datasets/nsl.html>                     |
| Reference     | Tavallaee, M., Bagheri, E., Lu, W., Ghorbani, A. (2009)        |
| Samples       | 125,973                                                        |
| Features      | 67 (41 original network features + 20 DT features + 6 metadata)|
| Attack rate   | 46.5% (DoS 36.5%, Probe 9.3%, R2L 0.8%, U2R 0.04%)             |
| License       | Public research dataset (UNB CIC terms of use)                 |

The 20 added Digital Twin features cover synchronization quality, model health,
and system performance ratios. Augmentation is performed by the
`preprocessing/dt_aware_preprocessor.py` module.

### 2. Synthetic Digital Twin Dataset

| Property      | Value                                                          |
|---------------|----------------------------------------------------------------|
| Source        | Custom generator (`data/generate_large_synthetic.py`)          |
| Samples       | 125,000                                                        |
| Features      | 95+ (41 network + 20 DT + 8 temporal + 6 categorical + metadata)|
| Attack rate   | 46.5%                                                          |
| Attack types  | Traditional (DoS, DDoS, Probe, R2L, U2R, Brute Force) plus DT-specific (Twin Desynchronization, Model Poisoning, Twin Spoofing, Sync Storm, Twin Inference, Cascade Failure) |
| Generation    | Procedural; numpy random distributions parameterised to match NSL-KDD marginal statistics |
| License       | Released with this repository under the same license as the source code |

The synthetic dataset is fully reproducible from the seed used in
`generate_large_synthetic.py`. No third-party data is incorporated into the
synthetic samples.

---

## Code Information

| File                                  | Role                                                       |
|---------------------------------------|------------------------------------------------------------|
| `models/dt_hybrid_net.py`             | Reference implementation of `DT-HybridNet` and `SyncAwareAttention` |
| `models/dt_hybrid_simplified.py`      | Lightweight variant used by the training pipeline          |
| `models/train_and_evaluate.py`        | Loads a dataset, trains the model, reports metrics         |
| `models/quick_test.py`                | Minimal end-to-end smoke test                              |
| `preprocessing/dt_aware_preprocessor.py` | DT-aware imputation, scaling, and feature engineering   |
| `data/dataset_collector.py`           | Collects raw network samples with synthetic DT signals     |
| `data/generate_large_synthetic.py`    | Generates the large synthetic dataset in batches           |
| `data/validate_generated_data.py`     | Validates batch integrity and class balance                |
| `models/GoogleColab/DT_HybridNet_Colab.ipynb` | Self-contained reproduction notebook for Google Colab |

---

## Requirements

- Python 3.9 or newer
- PyTorch 1.12 or newer (CUDA 11.6 recommended for GPU training)
- NumPy
- pandas
- scikit-learn
- (Optional) matplotlib and seaborn for the visualisation cells in the Colab notebook

Install the dependencies with:

```bash
pip install torch numpy pandas scikit-learn
```

For GPU training, follow the PyTorch installation guide for your CUDA version:
<https://pytorch.org/get-started/locally/>

Hardware used in the paper experiments: NVIDIA RTX 3080 GPU, Intel Xeon Gold
6248R CPU, 64 GB RAM.

---

## Usage Instructions

### Step 1. Get the datasets

Download `datasets.tar.gz` from the v1.0 release and extract it inside the
repository root:

```bash
wget https://github.com/alfehaid/SyncAware-Attenntion/releases/download/v1.0/datasets.tar.gz
tar -xzf datasets.tar.gz
```

This creates the `data/real_datasets/` and `data/synthetic_large/` directories
expected by the training scripts.

### Step 2. Quick smoke test

```bash
cd models
python quick_test.py
```

This trains the simplified model on a small subset and prints accuracy and loss.

### Step 3. Full training and evaluation

```bash
cd models
python train_and_evaluate.py
```

By default the script loads the enhanced NSL-KDD dataset, trains
`DT-HybridSimplified` for 100 epochs with early stopping (patience 15), and
writes results to `results/`.

### Step 4. Reproduce on Google Colab

Open `models/GoogleColab/DT_HybridNet_Colab.ipynb` in Google Colab and run all
cells. The notebook installs its own dependencies and generates a small
self-contained dataset, so no external download is required.

---

## Methodology

The full mathematical formulation of `SyncAware Attention` and the experimental
protocol are given in the paper. A short summary:

1. **Synchronization quality function `G(S)`** combines accuracy, normalized
   delay, reliability, and temporal consistency into a single score in `[0, 1]`.
2. **SyncAware attention** modulates the standard attention scores element-wise
   by `G(S)` before the softmax:
   `softmax((Q K^T (.) G(S)) / sqrt(d_k)) V`.
3. **DT-HybridNet** combines a CNN branch (spatial features), an LSTM branch
   (temporal patterns), and a Transformer fusion block. Each Transformer head
   uses SyncAware attention with head-specific synchronization features.
4. **Training**: 5-fold cross-validation, Adam optimizer (`lr = 1e-4`,
   `weight_decay = 1e-5`), batch size 256, early stopping (patience 15).
5. **Evaluation**: accuracy, precision, recall, F1, AUC-ROC, AUC-PR, and
   inference time per sample.

---

## Citation

If you use this code or the released datasets in your research, please cite the
paper:

```bibtex
@article{alfehaid2025syncaware,
  title   = {SyncAware attention: an attention mechanism for digital twin synchronization in cybersecurity applications},
  author  = {Alfehaid, Haitham Saleh M. and Ahmad, Iftikhar and Alassafi, Madini O.},
  journal = {PeerJ Computer Science},
  year    = {2025},
  note    = {Manuscript under review}
}
```

The DOI and full bibliographic details will be added once the paper is published.

When using the enhanced NSL-KDD dataset, please also cite the original source:

```bibtex
@misc{nslkdd2009,
  title  = {NSL-KDD dataset},
  author = {Tavallaee, Mahbod and Bagheri, Ebrahim and Lu, Wei and Ghorbani, Ali A.},
  year   = {2009},
  url    = {https://www.unb.ca/cic/datasets/nsl.html},
  note   = {Network intrusion detection dataset based on KDD Cup 1999}
}
```

---

## Contributing

Contributions, bug reports, and reproducibility checks are welcome.

- **Bug reports and questions**: please open an issue on the
  [GitHub issue tracker](https://github.com/alfehaid/SyncAware-Attenntion/issues)
  and include the Python version, the dataset you used, and the full error
  traceback when relevant.
- **Pull requests**: please base your work on the `main` branch and describe
  the motivation for the change in the PR description. For non-trivial changes
  please open an issue first to discuss the design.
- **Reproducibility**: if you cannot reproduce a number reported in the paper,
  please open an issue with the exact command, the random seed, and the
  hardware you used. We will respond and update the documentation.

---

## Maintainers and Contact

This repository is maintained by the authors of the paper at the Department
of Information Systems, Faculty of Computing and Information Technology,
King Abdulaziz University.

For questions about the code or the datasets, please contact the corresponding
author:

- Haitham Saleh M. Alfehaid - <halfehaid0003@stu.kau.edu.sa>

GitHub repository:
<https://github.com/alfehaid/SyncAware-Attenntion>
