# Type 2 Diabetes Mellitus Neural Network (DiabNet)

A Neural Network to predict type 2 diabetes (T2D) using a collection of SNPs highly correlated with T2D.

## Installation & Configuration

To install the latest version, clone this repository:

```bash
git clone https://github.com/zgcarvalho/diabnet.git
```

Install [poetry](https://python-poetry.org/) with [pip](https://pip.pypa.io/en/stable/):

```bash
pip install poetry
```

Configure poetry to create a virtual environment inside the project:

```bash
poetry config virtualenvs.create true
poetry config virtualenvs.in-project true
```

Install dependencies:

```bash
poetry install
```

To start the virtual environment, run:

```bash
poetry shell
```
