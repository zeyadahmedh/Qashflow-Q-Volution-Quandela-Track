# Qashflow Q-Volution: Quandela Track

This repository contains the project for the Q-Volution Hackathon by Girls In Quantum, supported by Quandela, focused on forecasting swaption volatility surfaces using hybrid quantum-classical machine learning models. The primary goal is to predict future volatility surfaces based on historical data.

Several models were developed and tested, including a classical LSTM baseline and various hybrid architectures that integrate photonic quantum circuits. The standout model, **Hybrid Photonic-Quantum Reservoir Computing (HP-QRC)**, demonstrates a novel approach by combining quantum and classical reservoir computing techniques.

## Models Explored

This project explores a range of time-series forecasting models to tackle the swaption volatility prediction problem:

*   **Classical LSTM:** A purely classical Long Short-Term Memory network to establish a performance baseline.
*   **Hybrid Quantum LSTM:** An LSTM network enhanced with a variational quantum circuit (VQC) layer to process compressed feature representations.
*   **Hybrid Quantum GRU:** A Gated Recurrent Unit (GRU) network integrated with a VQC layer.
*   **Hybrid Quantum TCN:** A Temporal Convolutional Network (TCN) augmented with a VQC layer, tested with 2, 3, 4, and 5-photon input states.
*   **Hybrid Quantum Transformer:** A Transformer Encoder architecture where a VQC acts as a feature extractor before the self-attention mechanism.
*   **Hybrid Photonic-Quantum Reservoir Computing (HP-QRC):** A unique model combining a Perceval-based quantum reservoir, a classically simulated photonic reservoir, and a classical Ridge regression readout layer.

## Repository Structure

```
.
├── Models/              # Saved model weights and scalers for various architectures
├── Notebooks/           # Jupyter notebooks for each model's implementation and training
└── hpqrc_results/       # Detailed results, predictions, and plots for the HP-QRC model
```

## Getting Started

### Prerequisites

*   Python 3.9+
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/zeyadahmedh/Qashflow-Q-Volution-Quandela-Track.git
    cd Qashflow-Q-Volution-Quandela-Track
    ```

2.  **Install the `merlin-quantum` library:**
    The notebooks use a specific version of the `merlin` library. Clone and install it from the source:
    ```bash
    git clone https://github.com/merlinquantum/merlin.git
    cd merlin
    pip install -e ".[dev]"
    cd ..
    ```

3.  **Install other required Python packages:**
    ```bash
    pip install perceval-quandela torch scikit-learn polars pandas numpy matplotlib seaborn aqora-cli
    ```

## Usage

### 1. Data Access

The project uses the `quandela/challenge-swpations` dataset from Aqora. The notebooks contain the necessary code to download this data automatically using the `aqora_cli` library.

```python
import aqora_cli
import polars as pl
from aqora_cli.pyarrow import dataset

df_raw = pl.scan_pyarrow_dataset(dataset("quandela/challenge-swpations", "v0.0.0")).collect()
```

### 2. Running the Models

All model implementations, training loops, and evaluations are contained within the Jupyter notebooks in the `Notebooks/` directory. You can run these notebooks using Jupyter Lab, Jupyter Notebook, or Google Colab to train the models and reproduce the results.

- **`HP-QR.ipynb`**: Contains the complete pipeline for the Hybrid Photonic-Quantum Reservoir Computing model, which was the primary submission.
- **`Hybrid-Quantum LSTM.ipynb`**: Implementation of the hybrid LSTM model.
- **`Hybrid_Quantum_GRU.ipynb`**: Implementation of the hybrid GRU model.
- **`Hybrid_Quantum_TCN.ipynb`**: Implementation of the hybrid TCN model (2-photon version). Variants for 3, 4, and 5 photons are also available.
- **`Hybrid_Quantum_TransformEncoder.ipynb`**: Implementation of the hybrid Transformer model.
- **`Classical LSTM.ipynb`**: The classical baseline model.

## Highlight: Hybrid Photonic-Quantum Reservoir Computing (HP-QRC)

The most developed model in this project is the HP-QRC, which follows a unique architecture for feature extraction.

**Pipeline:**
1.  **Preprocessing:** Raw price data (224 features) is standardized using `StandardScaler` and then reduced to 5 principal components using `PCA`.
2.  **Windowing:** The time-series data is structured into overlapping windows of 5 time steps.
3.  **Encoding:** The windowed PCA features are scaled to a `[0, π]` range using `MinMaxScaler` for phase encoding in the quantum circuit.
4.  **Feature Extraction:**
    *   **Quantum Reservoir:** An ensemble of 5 Perceval-based photonic circuits acts as quantum reservoirs, processing the encoded inputs to generate quantum features.
    *   **Photonic Reservoir:** A classical simulation of a time-delay photonic reservoir generates a separate set of features.
5.  **Readout:** The features from the quantum reservoir, the photonic reservoir, and the raw PCA data are concatenated. A `Ridge` regressor is then trained on this combined feature vector to predict the next day's PCA components.
6.  **Post-processing:** The predicted PCA components are transformed back to the original price space using the inverse PCA and inverse scaler transformations.

Saved artifacts for this model, including the trained model (`model.pkl`), the quantum ensemble weights (`quantum_ensemble.pt`), and prediction CSVs, are located in the `hpqrc_results/` directory.

The performance of other hybrid models can be found within their respective notebooks. For example, the Hybrid Quantum Transformer achieved a test Mean Squared Error of `0.1548`.
