# DGA Detection and XAI

## Setup

**Requirements**

- python = 3.10.x (use pyenv or some similar tool)
- poetry = 1.4.x

**Environment**

Create virtual environment:

```bash
python -m venv --upgrade-deps --copies env
```

Activate virtual environment:

```bash
source env/bin/activate
```

Install dependencies

```bash
poetry install
```

Alternatively, if you don't have poetry, use

```bash
pip install -r requirements.txt
```

## Train

### N-grams DGA Classification

```bash
python -m dga.train binary -n 2 -n 3 -l 256
```

### N-grams DGA Algorithm Classification

```bash
python -m dga.train multiclass -n 2 -n 3 -l 256
```

### Features MLP DGA Classification

```bash
python -m dga.train features
```

## Predict

### N-grams DGA Classification

```bash
python -m dga.predict binary -n {2,3} 'some-domain.com'
```

### N-grams DGA Algorithm Classification

```bash
python -m dga.predict multiclass -n {2,3} 'some-domain.com'
```

### Features MLP DGA Classification

```bash
python -m dga.predict features 'some-domain.com'
```

## Plots

The code for all the plots is in `dga.plots`.

To see a summary of the plots being created, take a look and run the `plots.ipynb` notebook. Please make sure that all
the training scripts have run.
