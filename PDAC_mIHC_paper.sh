#!/usr/bin/env bash

#Author: Katie Blise
#Date: September 2023
#This bash script will: create directories, download data files, and call the pdacMakeFigures.py python scprit to generate the results and figures in the manuscript "Machine learning interrogation of T cell function and spatial localization in the pancreatic tumor microenvironment in patients links biological pathways to clinical outcomes"
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

##download mIHC files and metadata from Zenodo into data folder
#zenodo_get RECORDNUMBERHERE
#
##navigate back to main PDAC_mIHC_paper directory
#cd ..
#
##run python script to generate results and figures
#python pdacMakeFigures.py
