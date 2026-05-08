## Installation

```bash
git clone https://github.com/Wilbur-dev/clip-bias-audit.git
cd clip-bias-audit
```
```bash
pip install -r requirements.txt
```

---

## Dataset

This project uses the **FairFace dataset** for demographic analysis.

FairFace provides balanced annotations for:
- Gender
- Race
- Age

### Download

The dataset can be downloaded from the official repository:

https://github.com/joojs/fairface

Download the following files:
- Images (train/val)
- `fairface_label_train.csv`


### Expected Directory Structure

After downloading, organize the dataset as follows:


data/
	fairface/
		train/
		fairface_label_train.csv


---




## Usage

### Step 1: Extract embeddings

```bash
python -m scripts.extract_embeddings
```


### Step 2: Compute similarity

```bash
python -m scripts.compute_similarity
```

### Step 3: Stability analysis

```bash
python -m scripts.compute_prompt_stability
python -m scripts.compute_model_agreement
python -m scripts.compute_bootstrap
```

### Step 4: Visualization

```bash
python -m scripts.plot_results
```