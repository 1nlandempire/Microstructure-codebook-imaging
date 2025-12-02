# Microstructure-codebook-imaging

Implementation code for  
**"DEMIC: Deep Learning Empowered Microstructure Codebook — New Paradigm for Multi-Parameter Tissue Characterization Estimation"**

This repository provides the implementation of DEMIC, including:

- Parameterization of multi-shell spherical mean signals (α, β, γ)
- Inference scripts for microstructure estimation
- A sample subject for quick testing

---

## 📦 Environment Setup

```bash
conda env create -f environment.yaml
```

## 🧊 Step 1 — Parameterize SMS (α, β, γ)

Use fit_αβγ.py to compute α, β, γ from raw DWI data.

```
python fit_αβγ.py \
    subjectDirectory \
    dwiFile \
    bvalFile \
    bvecFile \
    maskFile
```

**Arguments**

- `subjectDirectory` — Folder containing the subject's DWI data
- `dwiFile` — DWI NIfTI filename
- `bvalFile` — Name of b-value 
- `bvecFile` — Name of gradient vector
- `maskFile` — brain mask filename

The script will output parameterized α.nii.gz, β.nii.gz, γ.nii.gz in the subject directory.

## 🧠 Step 2 — Microstructure Estimation

Run inference using:

- `Model.py` — DEMIC architecture
- `forward.py` — testing pipeline
- `SUDMEX_60DWIs_40HCs.pth` — pretrained weights

Example:

```
python forward.py
```

This generates all predicted microstructure indices using the DEMIC framework.

## 📂 Sample Data

We provide a sample subject folder:

```
SUDMEX_sub-015/
    ├── A_60DWIs.nii.gz
    ├── B_60DWIs.nii.gz
    ├── C_60DWIs.nii.gz
    └── dwi_mask.nii.gz
```

Users can run Steps 2 directly on this sample to confirm correct installation.

