
# PoLixNano-MSML — Complete User Guide

---

## Contents
- [1. Repository Overview](#1-repository-overview)
- [2. Quick Start (No Programming Required)](#2-quick-start-no-programming-required)
  - [2.1 Download and Extract](#21-download-and-extract)
  - [2.2 One-Time Installation](#22-one-time-installation)
  - [2.3 One-Click End-to-End Prediction (DemoData)](#23-One-Click-End-to-End-Prediction-(DemoData))
  - [2.4 Self-Check](#24-self-check)
- [3. MSML Two-Stage Framework](#3-msml-two-stage-framework)
- [4. Data Files](#4-data-files)
- [5. Using Your Own Data](#5-using-your-own-data)
  - [5.1 Case A: Modify Formulation of Existing Monomers](#51-Modify-Formulation-of-Existing-Monomers)
  - [5.2 Case B: Polymers with new monomers](#52-Polymers-with-new-monomers)
  - [5.3 Run the Prediction](#53-Run-the-Prediction)
- [6. Reproducibility](#6-Reproducibility)
- [7. Data & Code Availability and Citation](#7-data--code-availability-and-citation)

---

## 1. Repository Overview

```
PoLixNano/
│
├─ data/
│   ├─ PP-set.csv               # Stage 2 dataset (particle-property prediction)
│   ├─ TE-set.csv               # Stage 1 dataset (TE index prediction)
│   ├─ DemoData.csv           # Example input for end-to-end prediction
│   ├─ SmiDesc.csv           # calculated descriptors for monomer SMILES
│   ├─ MonSMILES.csv           # SMILESs of monomers
│   └─ HS.csv           # High throughput screening chemical space

├─ results/ # saved custom results 
│
├─ results-check/ # expected results
│   ├─ TE/		# results of Stage 1 models (TE index)
│   │  ├─ ImportanceTE.csv         # Feature importance of TE index prediction
│   │  ├─ MetricsTE.csv         # Evaluation results of TE index prediction
│   │  ├─ modelTE.pkl         # model weights of TE index prediction
│   │  ├─ PredictionsTE.csv         # Predicted results of TE index prediction
│   │  └─ SHAP-logTE.pdf         # SHAP plot of TE index prediction
│   │  
│   ├─ PP/           # results of Stage 2 models (Size, PDI, EE, NDIs, Deff)
│   │  ├─ Deff.pkl         # model weights of Deff prediction
│   │  ├─ EE.pkl         # model weights of EE prediction
│   │  ├─ PDI.pkl         # model weights of PDI prediction
│   │  ├─ Size.pkl         # model weights of Size prediction
│   │  ├─ NDI-EE.pkl         # model weights of NDI-EE prediction
│   │  ├─ NDI-PDI.pkl         # model weights of NDI-PDI prediction
│   │  ├─ NDI-Size.pkl         # model weights of NDI-Size prediction
│   │  ├─ importance-Deff.csv         # Feature importance of Deff prediction
│   │  ├─ importance-EE.csv         # Feature importance of EE prediction
│   │  ├─ importance-PDI.csv        # Feature importance of PDI prediction
│   │  ├─ importance-Size.csv        # Feature importance of Size prediction
│   │  ├─ importance-NDI-EE.csv        # Feature importance of NDI-EE prediction
│   │  ├─ importance-NDI-PDI.csv        # Feature importance of NDI-PDI prediction
│   │  ├─ importance-NDI-Size.csv        # TFeature importance of NDI-Size prediction
│   │  ├─ shap-Deff.pdf        # SHAP plot of Deff prediction
│   │  ├─ shap-EE.pdf        # SHAP plot of EE prediction
│   │  ├─ shap-Size.pdf        # SHAP plot of Size prediction
│   │  ├─ shap-PDI.pdf        # SHAP plot of PDI prediction
│   │  ├─ shap-NDI-EE.pdf        # SHAP plot of NDI-EE prediction
│   │  ├─ shap-NDI-PDI.pdf        # SHAP plot of NDI-PDI prediction
│   │  ├─ shap-NDI-Size.pdf        # SHAP plot of NDI-Size prediction
│   │  ├─ predictions.csv       # Predicted results of Deff, EE, PDI, Size, NDIs prediction
│   │  └─ metrics.csv        # Evaluation results of Deff, EE, PDI, Size, NDIs prediction
│   │ 
│   └─ HS/
│   │  └─ res.csv         # predicted results of high throughput screen
│   │ 
│   └─ demo/
│       └─ res.csv         # predicted results of DemoData
│
├─ pcc.py			# calculating Pearson correlation coefficient and Spearman coefficient
├─ TE-pcc.csv          # results of Pearson correlation coefficient and Spearman coefficient
├─ TEPrediction.py          # model training for TE index prediction
├─ PPPrediction.py          # model training for PP prediction
├─ CalcMonDesc.py          # calculating descriptors for every monomer
├─ WeightedDesc.py		# calculating weighted descriptors for polymer
├─ predictor.py          # Runs both stages in sequence (one-click prediction)
├─ requirements.txt         # Python dependencies
│
└─ README.md                    # This file
```

---

## 2. Quick Start

### 2.1 Download and Extract
1. Click **Code → Download ZIP** on GitHub.
2. Right-click and extract the ZIP to any folder (avoid Chinese characters or spaces in the path).

### 2.2 One-Time Installation
1. Install **Python 3.9** from [python.org](https://www.python.org/downloads/release/python-390/).  
   - During installation, select **“Add Python to PATH”**.
2. Open a terminal (Windows: Command Prompt/PowerShell, macOS: Terminal).
For windows:Click the Start button (Windows icon)------Type cmd (for Command Prompt) or powershell-----Press Enter to open the terminal window.
3. Navigate to the project directory: # Note: Before running any code, please unzip the provided PoLixNano.zip file.
After unzipping, a folder named PoLixNano (or similar) should appear in the same directory as this README.
   ```bash
   cd PoLixNano
   ```
4. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

### 2.3 One-Click End-to-End Prediction (DemoData)
```bash
python WeightedDesc.py
```
This produces `data/demo-feat.csv` with calculated molecular descriptors
```bash
python Predictor.py -i data/demo-feat.csv -PP results/PP -TE results/TE/modelTE.pkl -o results/demo
```
This produces `results/demo/res.csv` with:
- Predicted **particle properties**: Size, PDI, EE, NDI-Size, NDI-PDI, NDI-EE, Deff  
- Predicted **TE index**: main score for ranking formulations

### 2.4 Self-Check
Compare your `results/demo/res.csv` with `results-check/demo/res.csv`.  
If they match (or differ only slightly due to rounding), your setup is correct.

---

## 3. MSML Two-Stage Framework

| Stage | Input | Output | Purpose |
|-------|-------|-------|--------|
| **Stage 1** | Polymer descriptors + particle properties | TE index | Links particle features to transfection efficiency |
| **Stage 2** | Polymer descriptors + concentration | Particle properties (Size, PDI, EE, NDIs, Diff) | Predicts particle properties from polymer chemistry |

The **end-to-end predictor** first uses Stage 2 to compute particle properties, then feeds these into Stage 1 to estimate TE index.  
This structure **improves interpretability** and reveals how polymer structure determines nanoparticle function.

---

## 4. Data Files

- **PP-Set.csv**: 268 samples for training Stage 2.
- **TE-Set.csv**: 88 samples for Stage 1 (TE index) training (log10-transformed TE values).
- **descriptors_list.csv**: All 208 RDKit molecular descriptors with concise explanations.
- **DemoData.csv**: Example input for quick testing.
- **demo-feat.csv**: Input file for demo testing.
- **HS.csv**: Input file for high throughput screening.
- **SmiDesc.csv**: Calculated descriptors for each SMILES.
- **MonSMILES.csv**: SIMLES list for all used monomers.

All files are ready to use offline.

---

## 5. Using Your Own Data

### 5.1 Case A: Modify Formulation of Existing Monomers
If monomers of your polymer are already included (e.g., EO, PO):
1. Modify 'm1' (monomer type 1), 'c1' (degree of monomer type 1), 'm2' (monomer type 2) and 'c2' (degree of monomer type 2) in `data/DemoData.csv`.
2. Modify `concentration' (mM were used, 1, 5, 10, and 20 mM correspond to polymer: mRNA weight ratio (w/w) of 5.9:1, 29.4:1, 58.8:1 and 117.6:1),_'arms' (e.g., 1, 4), 'hydrophilicity' (1: hydrophilic, 0: amphiphilic, -1: hydrophobic).
3. Keep all `desc_*` descriptor columns unchanged.

### 5.2 Case B: Polymers with new monomers
If your polymer have new monomer type:
- Add the new monomer type and its SMILES in MonSMILES.csv.
- Run CalcMonDesc.py to updata SmiDesc.csv
```bash
python CalcMonDesc.py
```
- Follow steps in 5.1.

### 5.3 Run the Prediction
```bash
python WeightedDesc.py
python predictor.py -i data/demo-feat.csv -PP results/PP -TE results/TE/modelTE.pkl -o results/demo
```
The output (results/demo/res.csv) will include predicted particle properties and the final TE index for each row.

---

## 6. Reproducibility

- Pearson correlation coefficient and Spearman coefficient:
  ```bash
  python pcc.py
  ```
- Stage 2 (particle properties):
  ```bash
  python PPPrediction.py
  ```
- Stage 1 (TE index):
  ```bash
  python TEPrediction.py
  ```
All scripts perform in 8:2 training-test splitting of corresponding dataset with hyperparameter optimization and save all results including model weights and evaluation results to `results/`. Due to initialization differences, there may be a slight error (less than 1%).

---

## 7. Data & Code Availability and Citation

All datasets, pre-trained models, and scripts are included and free to use.  
If you use this repository for research, please cite the corresponding publication.

---

### Sample Input Template (ready to edit)

A ready-to-edit CSV template is provided below. Save this text as `DemoData.csv` and fill in your own values.

```csv
Num., polymer_name, m1, c1, m2, c2, Arms, hydrophilicity, Concentration, MW
1, polymer1, PEG, 60, PPG, 68, 4, 0, 20, 6700,XXXX,XXXX,XXXX,....
2, polymer2, PEG, 345, PPG, 621, 4, 0, 20, 64722,XXXX,XXXX,XXXX,....
```
- Replace `XXXX` with computed or known descriptor values.
- Keep all columns in the same order.

This template ensures a smooth start for creating your own predictions.

---
