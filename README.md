[![Stars](https://img.shields.io/github/stars/ioanniskarvelis/Dependency-Parsing?style=social)](https://github.com/ioanniskarvelis/Dependency-Parsing/stargazers) [![Forks](https://img.shields.io/github/forks/ioanniskarvelis/Dependency-Parsing?style=social)](https://github.com/ioanniskarvelis/Dependency-Parsing/network/members) [![Issues](https://img.shields.io/github/issues/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing/issues) [![Pull Requests](https://img.shields.io/github/issues-pr/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing/pulls)
[![Last Commit](https://img.shields.io/github/last-commit/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing/commits) [![Repo Size](https://img.shields.io/github/repo-size/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing) [![Code Size](https://img.shields.io/github/languages/code-size/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing) [![Top Language](https://img.shields.io/github/languages/top/ioanniskarvelis/Dependency-Parsing)](https://github.com/ioanniskarvelis/Dependency-Parsing)

## Dependency Parsing

This repository showcases two modern dependency parsing approaches implemented from scratch in PyTorch, highlighting different modeling paradigms:

- A: Transition-based feed-forward parser (CS224N-style) — a lightweight shift-reduce parser that predicts actions (Shift/Left-Arc/Right-Arc) using fixed, handcrafted features and pretrained word embeddings.
- B: Graph-based BiLSTM parser (BIST-style) — a global, graph-scoring parser that encodes sentences with BiLSTMs and decodes trees with a projective algorithm (Eisner), optimizing arc and relation scores.

Both include training and evaluation pipelines, data loading utilities, and example outputs.

### Project structure
- `A/` — Transition-based parser with pretrained embeddings and classic shift-reduce transitions
- `B/` — Graph-based BiLSTM parser with Eisner decoding

### Requirements
Use the consolidated `requirements.txt` at the repo root. Key dependencies by part:

- Common
  - Python 3.7–3.10
  - PyTorch (CPU or CUDA)
  - NumPy
  - tqdm

- Part A (Transition-based)
  - Uses only PyTorch, NumPy, tqdm
  - Pretrained embeddings file `A/data/en-cw.txt` (provided path, file is git-ignored)

- Part B (Graph-based)
  - PyTorch, NumPy, tqdm
  - NLTK for LAS/UAS evaluation (`nltk.parse.DependencyEvaluator`)

Install all requirements:

```bash
python -m venv .venv
. .venv/Scripts/activate  # on Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional: `matplotlib` is included for plotting example curves.

### Data
- Part A expects CoNLL files in `A/data/` and an embeddings file `A/data/en-cw.txt`.
- Part B expects CoNLL files in `B/data/`.

You can place your own datasets in those directories using the same file names (`train.conll`, `dev.conll`, `test.conll`).

### Usage

#### A: Transition-based parser
From the `A/` directory:

```bash
python run.py  # trains, saves best model to A/results/<timestamp>/model.weights and evaluates
```

- Toggle debug mode: `python run.py --debug`
- Results and weights are written under `A/results/`.

#### B: Graph-based BiLSTM parser
From the `B/` directory:

Train:
```bash
python main.py --train_path data/train.conll --dev_path data/dev.conll --epochs 5 --lr 1e-3
```

Evaluate a trained model:
```bash
python main.py --test_path data/test.conll --model_dir results/<your_experiment_dir> --do_eval
```

Artifacts (logs, configs, vocab, checkpoints) are saved in `B/results/<experiment>/`.

### Plots
- Example plots and figures are in `docs/` and `A/plots/`.

### References
- Kiperwasser, E. and Goldberg, Y. (2016). Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations.
- CS224N Assignment materials (Transition-based parser specification).

### License
This code is for portfolio/educational purposes. Original paper references are included in subfolders.
