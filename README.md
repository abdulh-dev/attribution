# attribution
# Ephemeris: Multi-Model Subsystem Attribution Framework

## Overview

Ephemeris is a subsystem-level anomaly attribution framework designed to identify **which spacecraft subsystem is responsible for an anomaly** after an anomaly has already been detected.

Rather than relying on a single neural network, Ephemeris combines multiple machine learning paradigms—statistical, temporal, causal, and ensemble learning—to produce an interpretable subsystem diagnosis.

The philosophy behind the project is that no single model captures every aspect of system behavior:

- **Autoencoder** detects *that* something is wrong.
- **GMM** determines how statistically unusual the behavior is.
- **LSTM** captures temporal degradation patterns.
- **PCMCI** models causal relationships between sensor channels.
- **XGBoost** combines all evidence to predict the failing subsystem.

---

# Project Pipeline

```
                     Raw Sensor Telemetry
                              │
                              ▼
                      Window Generation
                              │
                              ▼
                    Autoencoder Encoder
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
   Latent Representation                 Reconstruction Error
          │                                       │
          └──────────────┬────────────────────────┘
                         │
              Feature Extraction Stage
                         │
        ┌────────────────┼────────────────┐
        │                │                │
        ▼                ▼                ▼
      GMM             LSTM             PCMCI
 Statistical      Temporal Model    Causal Discovery
   Features          Features          Features
        │                │                │
        └────────────────┼────────────────┘
                         ▼
                 Feature Fusion Vector
                         │
                         ▼
                   XGBoost Classifier
                         │
                         ▼
              Predicted Subsystem Label
```


---

# Model Components

---

## 1. Autoencoder

### Purpose

The autoencoder learns normal spacecraft behavior.

It is **not** responsible for identifying the subsystem.

Instead, it answers:

> "Does this telemetry window look abnormal?"

---

### Inputs

```
[B,T,C]

B = batch size
T = timesteps
C = sensor channels
```

---

### Outputs

```
latent

[B,32]

per_channel_error

[B,C]
```

---

### Saved Model

```
saved_models/autoencoder.pt
```

---

# 2. Gaussian Mixture Model (GMM)

### Purpose

Model the statistical distribution of normal telemetry.

The GMM estimates how likely a new sample belongs to the learned normal operating modes.

---

### Input

```
per_channel_error
```

or

```
latent vectors
```

---

### Output Features

Example:

```
log likelihood

distance to nearest Gaussian

cluster probabilities

responsibility vector
```

Example feature vector:

```
[
-14.2,
0.83,
0.11,
0.06,
...
]
```

---

### Saved Model

```
saved_models/gmm.pkl
```

---

# 3. LSTM

### Purpose

Capture temporal behavior.

The LSTM sees sequences rather than individual telemetry windows.

Instead of asking

> "Is this point abnormal?"

it asks

> "How did we get here?"

---

### Input

```
history

[B,T,C]
```

---

### Output

Final hidden state

```
[B,64]
```

This becomes the temporal embedding.

---

### Saved Model

```
saved_models/lstm.pt
```

---

# 4. PCMCI

### Purpose

Discover causal relationships between sensor channels.

Unlike the LSTM, PCMCI focuses on:

```
Which variables influence others?
```

rather than

```
Which variables change together?
```

---

### Input

Historical telemetry

---

### Output

Causal graph

Example

```
Voltage

↓

Current

↓

Temperature
```

Derived numerical features:

```
incoming strength

outgoing strength

graph degree

subsystem influence

causal entropy
```

Result:

```
[B,20]
```

---

### Saved Model

```
saved_models/pcmci_graph.pkl
```

---

# 5. Feature Fusion

Outputs from every model are concatenated.

Example dimensions

| Feature | Dimensions |
|----------|-----------:|
| Reconstruction Error | 64 |
| Latent | 32 |
| GMM | 8 |
| LSTM | 64 |
| PCMCI | 20 |
| Total | 188 |

Result

```
[B,188]
```

This is the final representation of one telemetry window.

---

# 6. XGBoost

### Purpose

Fuse every learned representation into a final subsystem prediction.

Unlike the previous models, XGBoost does **not** operate on raw telemetry.

It only receives the fused feature vector.

---

### Input

```
[B,188]
```

---

### Output

```
[B,6]
```

Example

```
NONE            0.01

Thermal         0.72

Power           0.06

Mechanical      0.12

Communication   0.05

Control         0.04
```

---

### Saved Model

```
saved_models/xgb_subsystem_model.pkl
```

---

# Training Pipeline

Every model is trained independently.

## Step 1

Train Autoencoder

```
train_autoencoder.py
```

Produces

```
autoencoder.pt
```

---

## Step 2

Train GMM

```
train_gmm.py
```

Produces

```
gmm.pkl
```

---

## Step 3

Train LSTM

```
train_lstm.py
```

Produces

```
lstm.pt
```

---

## Step 4

Run PCMCI

```
train_pcmci.py
```

Produces

```
pcmci_graph.pkl
```

---

## Step 5

Generate fused dataset

```
generate_fused_dataset.py
```

For every telemetry window

```
Autoencoder

↓

latent

↓

reconstruction error

↓

GMM features

↓

LSTM embedding

↓

PCMCI features

↓

188-dimensional feature vector
```

Output

```
X_train.npy

y_train.npy
```

---

## Step 6

Train XGBoost

```
train_xgboost.py
```

Produces

```
xgb_subsystem_model.pkl
```

---

# Inference Pipeline

During deployment

```
Telemetry Window

↓

Autoencoder

↓

per_channel_error

latent

↓

GMM

↓

LSTM

↓

PCMCI

↓

Feature Fusion

↓

XGBoost

↓

Subsystem Probabilities
```