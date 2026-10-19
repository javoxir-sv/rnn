# Recurrent Neural Networks for Name Modeling

This project is an experimental PyTorch implementation of recurrent neural
networks for modeling and analyzing baby names. It compares hand-written
vanilla RNN, GRU, and LSTM cells on character-level name generation and gender
classification tasks, and includes experiments with perplexity, multitask
training, vanishing gradients, and learned character embeddings.

The main experiment is implemented in [`main.py`](./main.py). The original
notebook, [`main.ipynb`](./main.ipynb), contains the exploratory narrative and
plots. Training utilities shared by the script are in
[`ttloop.py`](./ttloop.py).

## What the project does

The script runs an end-to-end set of experiments:

1. Loads and explores the baby-name dataset, including name lengths, gender
   balance, and character frequencies.
2. Adds start/end tokens, pads names, and creates reproducible training,
   validation, and test splits.
3. Defines vanilla RNN, GRU, and LSTM cells and models with character-generation
   and gender-classification outputs.
4. Trains and evaluates models with character-only, gender-only, and combined
   objectives; plots losses and generates sample names.
5. Calculates character prediction perplexity, collects gradients to
   demonstrate vanishing-gradient behavior, and visualizes learned character
   embeddings with PCA and t-SNE.
6. Saves the final LSTM model to `data/lstm.pkl`.

The recurrent cells are implemented directly with PyTorch tensor operations
rather than using the built-in `torch.nn.RNN`, `torch.nn.GRU`, or
`torch.nn.LSTM` layers.

## Project layout

```text
.
├── data/
│   ├── babynames-clean.csv  # Input names and gender labels
│   └── lstm.pkl             # Serialized LSTM artifact (written by main.py)
├── main.ipynb               # Exploratory notebook
├── main.py                  # Full experiment script
├── requirements.txt         # Python dependencies
└── ttloop.py                # Shared training and evaluation loops
```

The CSV is read without a header: its first column contains names and its
second contains gender labels (`boy` or `girl`).

## Setup

Use a Python environment and install the dependencies from the repository
root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead.
The script selects an available PyTorch accelerator when possible and falls
back to the CPU.

## Run

Run the complete workflow from the repository root:

```bash
python main.py
```

This is a research script rather than a command-line application: running it
starts the full sequence of training and analysis experiments. It can take a
while, uses Matplotlib to display plots, and writes the final serialized model
to `data/lstm.pkl`, replacing that artifact. There are no command-line
options; the experiment settings are currently defined in `main.py`.

The saved artifact is serialized with `joblib` and contains a PyTorch model.
Only load it from a trusted source, and use a compatible version of the
project and its dependencies.

## Reproducibility

The script seeds Python, NumPy, and PyTorch random number generators with
`42`. Training is still subject to differences between PyTorch versions,
hardware, and accelerator backends.

## License

This project is distributed under the [MIT License](./LICENSE).
