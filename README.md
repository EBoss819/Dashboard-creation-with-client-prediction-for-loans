﻿# Scoring-model-Project
## Openclassroom project n°7

The purpose of this project is to create an dashboard environement for a bank to be able to use their data to decide whether to grant credit to its customers.
The data come from kaggle : https://www.kaggle.com/c/home-credit-default-risk/data

Description of the files :
The first 2 files will be deployed in an online VM with all the required data.
The needed data and scripts on the VM are updated on VM reboot.

-api.py 
is for the api used for prediction

-dashapp.py
is the dash application

-Projet7-featuring.ipynb 
is the feature ingeneering from the data from kaggle

-Projet7-classifier-choice.ipynb
is some different classifier tests to choose the most adapted

-Projet7-gridsearchs-on-chosen-model.ipynb 
is the gridsearch on the best model previously found and saving model localy with mlflow
-> final_model file is the so called model.
