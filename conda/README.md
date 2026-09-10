# Conda Environment Setup

### Create and activate the environment

```bash
conda env create -f conda/environment.yml
conda activate foundry
```

This creates the `audio` environment with Python 3.12. Then install the pip dependencies:

### Install dependencies

```bash
pip install -r conda/requirements.txt
```

This installs the core utility packages necessary to run the project.
