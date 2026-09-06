# Contributing

This started as a personal project, but contributions are welcome.

## Setup

```bash
git clone https://github.com/Ashu368/quantis.git
cd quantis
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pytest tests/ -v
```

## Before opening a PR

- `ruff check .` must pass (CI enforces this)
- `pytest tests/ -v` must pass
- New strategies should implement `strategies.base.Strategy` and add a test verifying the model (if any) is only ever fit on training data — see `tests/test_ml_classifier.py::test_model_is_only_fit_once_on_training_data` for the pattern.
- No new strategy should be merged without at least a walk-forward result, not just a single train/test split — single splits are too easy to accidentally overfit to.

## Ideas for contributions

- Additional strategies (momentum, pairs trading, volatility breakout)
- Additional data sources beyond `yfinance` (crypto, futures)
- More microstructure analysis techniques in `analysis/`
- Live trading integration (Alpaca, Interactive Brokers)

Open an issue before starting large changes so we can align on approach first.
