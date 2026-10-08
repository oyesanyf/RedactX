# Contributing to RedactX

We welcome contributions from researchers, clinical informaticists, and privacy engineers!

## Development Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/oyesanyf/RedactX.git
   cd RedactX
   ```

2. **Create and activate a virtual environment** (Python 3.10 to 3.12):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install in editable mode with development dependencies**:
   ```bash
   pip install -e .[dev,train,production]
   ```

## Running the Test Suite

Run the full pytest suite locally:
```bash
pytest -q
```

To run the locator regression suite:
```bash
python tests/test_locator_regression.py
```

## Coding Guidelines

- **Zero-Touch Environment**: Code, tests, and scripts must never alter global system registry or machine environment variables.
- **Type Annotations**: Ensure all functions and methods have clear type hints.
- **Reproducibility**: Changes to model inference, calibration, or benchmark metrics must retain mathematical reproducibility.
- **HIPAA Standards**: Span classification and masking logic must conform strictly to 45 CFR § 164.514(b)(2). Non-identifying clinical variables (such as ages $\le 89$ and gender descriptors) should be preserved under Safe Harbor to maintain clinical utility.

## Submitting Pull Requests

1. Fork the repo and create a new feature branch (`git checkout -b feature/your-feature-name`).
2. Verify that all tests pass (`pytest -q`).
3. Commit your changes with descriptive commit messages.
4. Push to your fork and submit a Pull Request targeting `main`.
