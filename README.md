# Machine learning links T cell function and spatial localization to neoadjuvant immunotherapy and clinical outcome in pancreatic cancer

This repository contains all code necessary to reproduce the results and figures of the manuscript "Machine learning links T cell function and spatial localization to neoadjuvant immunotherapy and clinical outcome in pancreatic cancer," which can be found as a preprint here: [https://www.biorxiv.org/content/10.1101/2023.10.20.563335v1](https://www.biorxiv.org/content/10.1101/2023.10.20.563335v1). All data, including the output of the multiplex immunohistochemistry computational imaging processing workflow for each tissue region and metadata for each patient/region, are available on Zenodo: [DOI: 10.5281/zenodo.8357193](https://doi.org/10.5281/zenodo.8357193).

# Steps to Create Figures

## 1) Clone repository

**a.** Open Terminal and navigate to desired directory: `cd Desired/Directory/Here/`

**b.** Clone repo: `git clone https://github.com/kblise/PDAC_mIHC_paper.git`

## 2) Create new conda environment

**a.** Install Conda if not already installed: [Instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)

**b.** Navigate into PDAC_mIHC_paper directory: `cd PDAC_mIHC_paper/`

**b.** Create new conda environment with necessary packages: `conda env create -f pdacEnv.yml`

**c.** Activate pdacEnv conda environment: `source activate pdacEnv` or `conda activate pdacEnv` depending on Conda version

## 3) Run bash script to create directories, download data, and run Python script to generate figures

**a.** Run bash script from command line: `bash PDAC_mIHC_paper.sh`

**Note: Csv files created to generate figures will be saved to the 'results/dfCreated' folder, figures will be saved to the 'results/figures' folder, and tables will be saved to the 'results/tables' folder.**

This program is intended for Python version 3.
