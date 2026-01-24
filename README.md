# Basa OCR Suite 

## Setup

```bash
uv sync
```

## Training tasks

```bash
# MNIST dataset
uv run -m basa.mnist.train

# EMNIST dataset
uv run -m basa.emnist.train

# Captcha with fixed length
uv run -m basa.captcha_fixed.train

# Captcha with variable length
uv run -m basa.captcha_var.train
```

## Eval tasks

```bash
# Captcha with variable length
uv run -m basa.captcha_var.eval
```
