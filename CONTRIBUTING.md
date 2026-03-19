# Contributing to TaniFi

First off, thank you for considering contributing to TaniFi! It's people like you that make federated learning for agriculture a reality.

## Code of Conduct

By participating in this project, you are expected to uphold our [Code of Conduct](CODE_OF_CONDUCT.md).

## How Can I Contribute?

### Reporting Bugs
This section guides you through submitting a bug report for TaniFi. Following these guidelines helps maintainers and the community understand your report, reproduce the behavior, and find related reports.

- **Check open issues** before submitting a new bug to avoid duplicates.
- **Use the Bug Report template** when opening an issue.
- **Provide reproducible steps**, especially details on data format and configuration YAML used.

### Suggesting Enhancements
Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion, please use the Feature Request issue template and provide clear, documented reasoning for why the enhancement is needed, particularly regarding agricultural applicability or federated learning improvements.

### Pull Requests
The process described here has several goals:
- Maintain TaniFi's code quality and architecture.
- Fix problems that are important to users.
- Enable a sustainable system for TaniFi's maintainers to review contributions.

Please follow these steps to have your contribution considered by the maintainers:
1. Fork the repo and create your branch from `main` or the appropriate feature branch.
2. If you've added code that should be tested, add tests in the `tests/` directory.
3. Ensure the test suite passes (run `pytest`).
4. Format your code consistently. We prefer PEP 8 standards with type hinting where possible.
5. Create a Pull Request using the provided PR template.

## Setup for Development

To set up a local development environment:

1. Clone your fork of the repository: `git clone https://github.com/your-username/TaniFi.git`
2. Create a virtual environment: `python3 -m venv venv`
3. Activate it: `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Download datasets using the provided scripts (do not commit large datasets to the repository).

Thank you for contributing!
