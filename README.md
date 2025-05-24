# 🔁 Transformer-Parameter-Prediction

This project explores the use of Transformer-based architectures for predicting parameters of periodic signals in a regression setting.  
The model aims to infer both the amplitude **A** and frequency **ω** of a signal defined by:

\[
y_i(A_i, ω_i; t) = A_i \cdot \sin(ω_i t)
\]

The input data is a vector representation of the signal, and the task is to estimate the underlying parameters using deep learning — specifically, Transformers.

## 🎯 Objective

- Predict amplitude **A** and frequency **ω** of sine signals using Transformers.
- Investigate the effect of Fourier transform on prediction quality.
- Compare three Transformer architectures:
  - Encoder-Only
  - Decoder-Only
  - Encoder-Decoder
- Perform hyperparameter optimization using Optuna.

## 🧪 Data Description

- **Synthetic input**: 100 time-point vectors representing sampled sine signals.
- **Target variables**: amplitude (A) and frequency (ω), sampled in range [0.1, 10].
- Signals generated over interval \([0, 2π]\).
- Data normalized using `MinMaxScaler` (range [0, 1]).
- Optional preprocessing: **Fourier Transform** using `torch.fft.fft`.

## 🛠️ Methods

- Built custom Transformer models using PyTorch.
- Optimized with:
  - `d_model = 128`
  - 4 layers, 4 attention heads
  - FF layer dim = 256
  - Dropout = 0.2, Batch size = 32
  - LR = 0.0001
- Activation functions tested: `relu`, `leaky_relu`, `gelu`, `silu`, `tanh`, `elu`. Best results with **silu**.

### Transformer Types Evaluated

| Model           | MSE (A) | MSE (ω) | MAE (A) | MAE (ω) | R²       |
|----------------|---------|---------|---------|---------|----------|
| Encoder         | 0.0769  | 0.2655  | 0.2078  | 0.3379  | 0.9791   |
| Decoder         | 0.0281  | 0.0794  | 0.1093  | 0.1406  | 0.9934   |
| Encoder-Decoder | 0.0452  | 0.1101  | 0.1387  | 0.1907  | 0.9905   |

> The **Decoder-Only** model consistently outperformed others.

## ⚡ Fourier Transform Impact

Fourier Transform improved frequency predictions but slightly worsened amplitude predictions. Best performance achieved with **Log Fourier Transform**:

| Method              | MSE (A) | MSE (ω) | MAE (A) | MAE (ω) | R²     |
|---------------------|---------|---------|---------|---------|--------|
| Without Fourier     | 0.0741  | 0.0830  | 0.1973  | 0.1865  | 0.9908 |
| Log Fourier         | 0.0633  | 0.0581  | 0.1737  | 0.1783  | 0.9929 |

> Log Fourier provided a more stable frequency representation for high-frequency signals.

## 📈 Final Results

With 10,000 samples and Fourier preprocessing:

| Model           | MSE (A) | MSE (ω) | MAE (A) | MAE (ω) | R²     |
|----------------|---------|---------|---------|---------|--------|
| Encoder         | 0.0610  | 0.0580  | 0.1558  | 0.2020  | 0.9928 |
| Decoder         | 0.0417  | 0.0099  | 0.1260  | 0.0777  | 0.9969 |
| Encoder-Decoder | 0.0586  | 0.0164  | 0.1477  | 0.0880  | 0.9954 |

## 📄 Report

The full report with all tables, charts, architecture details and references is available.

## 📚 Dependencies
* torch
* numpy
* scipy
* matplotlib
* optuna

## 🧠 Author
Lidiia Pylyp — 2024

