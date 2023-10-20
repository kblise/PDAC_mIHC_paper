#!/usr/bin/env bash

#Author: Katie Blise
#Date: October 2023
#This bash script will: create directories, download data files, and call the pdacMakeFigures.py python scprit to generate the results and figures in the manuscript "Machine learning links T cell function and spatial localization to neoadjuvant immunotherapy and clinical outcome in pancreatic cancer"
#Call "bash PDAC_mIHC_paper.sh" from the command line to run this script.

#create directory structure
mkdir {data,results}
cd results
mkdir {dfCreated,figures,tables} #three subfolders in results
cd dfCreated
mkdir updatedCsvs
cd ../figures
mkdir figureS4F
cd ../../data #enter data folder

#download data from Zenodo: DOI: 10.5281/zenodo.8357193
#all data lives in data.zip file, which contains 2 folders: mIHC_files and metadata
zenodo_get 8357193
#unzip data.zip file
unzip data.zip

#navigate back to main PDAC_mIHC_paper directory
cd ..

#run python script to generate results and figures
python pdacMakeFigures.py
