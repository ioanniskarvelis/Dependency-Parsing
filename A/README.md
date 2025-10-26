## A — Transition-based Dependency Parser

Feed-forward transition-based parser (CS224N-style) implemented in PyTorch.

### Requirements
- Python 3.7+
- See root `requirements.txt` and optional `A/local_env.yml` for a conda env example.

### Data
Place CoNLL-formatted files in `A/data/` as:
- `train.conll`
- `dev.conll`
- `test.conll`

Pretrained embeddings file:
- `A/data/en-cw.txt` (ignored by git)

### Train & Evaluate
From inside `A/`:

```bash
python run.py            # trains for 10 epochs, saves best model under A/results/<timestamp>/
python run.py --debug    # quick debug run
```

- Best weights: `A/results/<timestamp>/model.weights`
- During non-debug runs, the script restores best weights and reports final test UAS.

### Files
- `run.py` — training loop and evaluation
- `parser_model.py` — feed-forward network
- `utils/parser_utils.py` — data loading, feature extraction, and batching
- `parser_transitions.py` — transition system and minibatch parsing

### Notes
- GPU is optional; CPU training works for short demos.
- For reproducible environments you may adapt `A/local_env.yml`.
