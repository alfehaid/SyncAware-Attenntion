# Dataset Summary

## 1. Enhanced NSL-KDD Dataset

| Property | Value |
|----------|-------|
| Source | University of New Brunswick (Real Network Traffic) |
| Samples | 125,973 |
| Features | 67 (41 original + 20 DT + 6 metadata) |
| Attack Rate | 46.5% |
| Attack Types | Normal (53.5%), DoS (36.5%), Probe (9.3%), R2L (0.8%), U2R (0.04%) |
| Size | ~25 MB |

DT features added: synchronization metrics, model health indicators, system performance ratios.

## 2. Synthetic Digital Twin Dataset

| Property | Value |
|----------|-------|
| Source | Custom DT-IDS Generator (generate_large_synthetic.py) |
| Samples | 125,000 |
| Features | 95+ (41 network + 20 DT + 8 temporal + 6 categorical + metadata) |
| Attack Rate | 46.5% |
| Size | ~850 MB (25 batches) |

Attack distribution:
- Normal: 66,875 (53.5%)
- Traditional attacks: 31,979 (25.6%) — DoS, DDoS, Probe, R2L, U2R, Brute Force
- DT-specific attacks: 26,146 (20.9%) — Twin Desynchronization, Model Poisoning, Twin Spoofing, Sync Storm, Twin Inference, Cascade Failure

## Feature Categories

| Category | Count | Description |
|----------|-------|-------------|
| Network Flow | 41 | Duration, bytes, protocols, services |
| DT Synchronization | 5 | Delays, accuracy, status |
| DT Model Health | 4 | Confidence, drift, entropy |
| DT System Metrics | 6 | Load, latency, health |
| Temporal | 8 | Timestamps, sequences, trends |
| Contextual | 6 | Topology, devices, connections |
| Attack Metadata | 6 | Categories, severity, confidence |
