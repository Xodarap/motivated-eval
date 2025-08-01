# Motivated Interpretation Evaluation

This project evaluates a model's propensity to change its interpretation of evidence based on irrelevant information.

## Setup

### Prerequisites

1. Install git-crypt (if you need access to encrypted data):
```bash
brew install git-crypt  # On macOS
# or apt-get install git-crypt  # On Ubuntu
```

2. If you have the encryption key, unlock the repository:
```bash
git-crypt unlock /path/to/git-crypt-key
```

### Installation

Install dependencies with uv:
```bash
uv sync
```


## Running the Evaluation


Run the full evaluation:
```bash
uv run inspect eval motivated_interpretation.py --model anthropic/claude-3-5-haiku-20241022
```

To run it with a prompt to be biased:

```bash
uv run inspect eval motivated_interpretation.py --model anthropic/claude-3-5-haiku-20241022 -T bias=True
```

## Data Encryption

The `data/` folder is encrypted using git-crypt. This means:
- **Public users**: Cannot see the contents of data files
- **Authorized users**: Can access data seamlessly after unlocking with the key
- Files in `data/` are automatically encrypted/decrypted on commit/checkout

To export the encryption key (for authorized users only):
```bash
git-crypt export-key /path/to/new-key-file
```