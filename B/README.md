## B — Graph-based Dependency Parser (BiLSTM + Eisner)

PyTorch implementation of a graph-based dependency parser inspired by the BIST parser.
Based on the paper “Simple and Accurate Dependency Parsing Using Bidirectional LSTM Feature Representations”.

### Requirements
- Python 3.7+
- Install via root `requirements.txt`:
  - PyTorch (CPU or CUDA)
  - NumPy, tqdm
  - NLTK (for evaluation)

### Data format
- Expects CoNLL-formatted files with at least token, POS, head, and relation columns.
- Default paths (relative to `B/`): `data/train.conll`, `data/dev.conll`, `data/test.conll`.

### Training
From inside `B/`:

```bash
python main.py \
  --train_path data/train.conll \
  --dev_path data/dev.conll \
  --epochs 5 \
  --lr 1e-3 \
  --w_emb_dim 100 \
  --pos_emb_dim 25 \
  --lstm_hid_dim 125 \
  --mlp_hid_dim 100 \
  --n_lstm_layers 2
```

Common flags:
- `--alpha` word dropout (default 0.25)
- `--ext_emb PATH` path to external word embeddings (optional)
- `--seed` for reproducibility (default 1234)
- `--no_cuda` to force CPU even if CUDA is available

Artifacts are saved under an auto-generated directory in `B/results/` based on run parameters and date.

### Evaluation
Evaluate a trained model on test data:

```bash
python main.py \
  --test_path data/test.conll \
  --model_dir results/<your_experiment_dir> \
  --do_eval
```

This loads `parser.pt` and reports LAS/UAS on the console and in output files.

### Outputs
Inside `B/results/<experiment_dir>/` you will typically find:
- `train.log` — training logs
- `eval.log` — evaluation logs (when `--do_eval`)
- `config.pkl` — serialized run configuration
- `vocab.pkl` — vocabulary and mappings
- `parser.pt` — best model checkpoint
- `train_stats.pkl` — per-epoch metrics (when training)
- `dev_results.txt`, `test_results.txt` — summary metrics
- `dev_pred.conll`, `test_pred.conll` — predicted dependency graphs in CoNLL

### Tips
- Larger `epochs` and patience on `dev` typically improve `UAS/LAS`.
- External embeddings (`--ext_emb`) may help, provided their vocab overlaps your data.
- CPU works for quick tests; for speed enable CUDA if available.
