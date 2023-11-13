#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model, decomposition, manifold, preprocessing
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import precision_recall_curve, roc_curve, auc, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import pickle
import warnings
import os
from particle import Particle
import argparse

def main(path, output_path):
    #concat files with input particle data
    files = os.listdir(path)
    files = [os.path.join(path,f) for f in files if "non_zero_examples_run" in f ]

    data = pd.read_csv(files[0])
    for file in files[1:]:
        data = pd.concat([data,pd.read_csv(file)])

    #filter to only the proton ZDC
    data = data[data['Detector']=="P"]
    data = data.reset_index(drop=True)

    #add mass and charge from Pfg
    data["mass"] = data["Pdg"].apply(lambda x: Particle.from_pdgid(x).mass)
    data["charge"] = data["Pdg"].apply(lambda x: Particle.from_pdgid(x).charge)

    #leave only examples with >10 photons for training GANs/VAE
    for_training_generative_models = data["PhotonSum"]>=10
    data = data[for_training_generative_models]

    #drop unnecessary columns and save
    data = data.drop(columns=["PhotonSum", "Detector", "Pdg"])
    np.savez(os.path.join(output_path,"data_nonrandom_particles.npz"), data)



    #concat files with neutron ZDC responses
    files = os.listdir(path)
    files = [os.path.join(path,f) for f in files if "proton_image_run" in f ]

    data_neu_response = np.loadtxt(files[0])
    data_neu_response = data_neu_response.reshape(-1,56,30)
    for file in files[1:]:
        data_neu_response2 = np.loadtxt(file)
        data_neu_response2 = data_neu_response2.reshape(-1,56,30)
        data_neu_response = np.append(data_neu_response,data_neu_response2,axis=0)

    #leave only examples with >10 photons for training GANs/VAE and save
    data_neu_response = data_neu_response[for_training_generative_models]
    np.savez(os.path.join(output_path,"data_nonrandom_responses.npz"), data_neu_response)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'preprocess data from GEANT txt output to .npz for training generative models')
    parser.add_argument('path', help='path to dir with non_zero_examples_run_N.txt and neutron_image_run_N.txt')
    parser.add_argument('out_path', help='path to output dir to save .npz files')
    args = parser.parse_args()

    main(args.path, args.out_path)
