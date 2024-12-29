# 🍲 AAIS-AI 

### ZINDI Project on Crop Health in Telangana State

## 🎯 Project Objective
The goal of this competition is to build a machine learning model capable of classifying the health status of various crops in Telangana State, south-central India, using data on farming practices and historical Sentinel-2 time series data. This model will help farmers and government agencies monitor crop conditions and take proactive measures to mitigate negative effects on crop health, thereby improving harvest yields.

## Prerequisites
Before running the code, ensure the following are installed and configured:
- **Python 3.11** or higher
- **Poetry** for dependency management. [Install Poetry](https://python-poetry.org/docs/#installation)

Environment variables are stored in the `.env` file.

## Installation Guide

### Step 1: Clone the Repository
```bash
git clone https://github.com/mbathe/zindi_telangana_crop_health_challenge.git
cd zindi_telangana_crop_health_challenge
```

### Step 2: Install Dependencies
```bash
poetry install
```

### Step 3: Download the Dataset
Run the following command at the project root to download the TIFF image dataset of chronological plant evolution and save it to the default location `./data/images/` (defined by the **DIR_DATASET** environment variable).
```bash
poetry run python scripts/download_dataset.py
```