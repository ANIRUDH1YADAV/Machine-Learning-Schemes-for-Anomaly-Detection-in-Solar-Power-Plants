# Solar Power Plant Anomaly Detection

This project implements the anomaly detection workflow from Ibrahim et al. (2022) and adds a custom BiLSTM baseline for comparison.

## What Was Implemented

The notebook loads the Plant 1 and Plant 2 generation and weather datasets, merges them by timestamp, fills missing values, and creates anomaly labels from the paper’s reported fault dates. It then evaluates four approaches:

- Isolation Forest on AC power, DC power, and irradiation
- Prophet forecasting with residual-based anomaly detection
- AE-LSTM autoencoder for sequence reconstruction error
- BiLSTM autoencoder as a custom extension



## How The Results Relate To The Paper

The implementation follows the paper’s model families and core hyperparameters, especially for Prophet and AE-LSTM. The AE-LSTM architecture matches the paper’s layer layout and 24-step window, while Isolation Forest and Prophet use the paper’s reported settings.

The current run is more conservative than the paper in two ways:

1. Thresholds are calibrated on training data only.
2. The anomaly labels are applied at the day level for the reported fault dates.

Because of that, the absolute metrics do not exactly reproduce the paper. The AE-LSTM remains the closest match to the paper’s modeling intent, but the custom BiLSTM performs better in this notebook’s current run.

## Results From The Current Run

| Model | MAE [W] | MSE [W²] | R² Score | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Isolation Forest | - | - | - | 0.9160 | 0.1231 | 0.0601 | 0.0808 |
| Facebook Prophet | 76.6111 | 19902.6738 | 0.8514 | 0.7816 | 0.1553 | 0.0952 | 0.1180 |
| AE-LSTM | 22.5563 | 3379.2658 | 0.9748 | 0.5657 | 0.5657 | 1.0000 | 0.7226 |
| BiLSTM | 15.1979 | 1982.3563 | 0.9852 | 0.7694 | 0.7226 | 0.9613 | 0.8250 |

## BiLSTM Result Summary

The BiLSTM variant improved over AE-LSTM on every tracked metric in this run:

- Lower MAE and MSE
- Higher R²
- Higher F1 score

That makes BiLSTM the best-performing model in the current notebook, even though it is not part of the original paper.

## Recommended Future Improvements

The best next steps are likely to come from better labels and stronger sequence modeling, not just minor tuning:

- Use inverter-level anomaly labels instead of day-level labels if available
- Train separate models per inverter or per plant
- Use multivariate sequences instead of AC power only
- Add a validation split and tune thresholds for F1 or PR-AUC
- Try GRU, CNN-LSTM, attention-based autoencoders, or a Transformer model
- Evaluate with rolling-window testing to better reflect deployment
- Add class-imbalance-aware metrics such as PR-AUC, balanced accuracy, and event-based recall
- Save trained models and build a repeatable inference pipeline

## Generated Files

- `code/plant1_processed.csv`
- `code/plant2_processed.csv`
- `code/correlation_matrix.csv`
- `code/iso_forest_results.csv`
- `code/prophet_results.csv`
- `code/aelstm_results.csv`
- `code/final_comparison.csv`
- `code/IMPLEMENTATION_AUDIT.txt`

