# THESIS

## Overview
This repository contains the code and implementation for our ballet classification thesis project.

## Prerequisites
- Python 3.x (<= 3.11)
- Required dependencies (install via `pip install -r requirements.txt`)
- Download the pretrained model (~400MB) from https://drive.google.com/drive/folders/1HdewpErj1nQ5kivXYVxfJ6gmq4gDHxvY and place in the `Transpondancer/models` directory
    - Unless you want to pretrain yourself. Note that datasets are not provided here as they are many GB in size

# Create virtual environment macOS/Linux (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install packages
pip install -r requirements.txt

## Installation
1. Clone this repository
2. Navigate to the project directory
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your data according to the required format
2. Run the main script:
    ```bash
    python orchestrator.py
    ```
3. Results will be saved in the `output/` directory

## Deactivate the virtual environment (if used)
deactivate

## Datasets
For datasets used in this project, please refer to the [Transpondancer paper](https://arxiv.org/abs/2310.07847) or contact the repository maintainer for access instructions.

## Structure
```
.
├── orchestrator.py       # Main file orchestrating the 3 stages:
├── Transpondancer/       # Source code for CNN 
├── Motion_Classifier/    # Source code for Motion Feature Extractor
├── Ontology_Reasoner/    # Source code for the OWL reasoner
├── l2d_eval/             # Test data files
├── l2d_train/            # Training data files
├── output/               # Generated results
└── README.md             # This file
```

## Contact
For questions about datasets or implementation details, please open an issue or contact the maintainer.
