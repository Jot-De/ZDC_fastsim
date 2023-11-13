# ML Pipeline

This directory provides code for running the whole ML pipeline, i.e.:
- Simulating the data in full simulations.  
- Preparing simulated data for using it in model training.  
- Training a model.  
- Converting trained model to the onnx format.  
  
The code provided here assumes training a VAE proton model, but with a few changes also other types of models can be trained.  
  
The following directory structure is required:  
On the lxplus:  
- HOME  
  - bash-scripts  
    - pipeline.sh  
    - datasimulation_pipeline.sh  
    - datasimulation_parallel.sh  
    - extract_data.sh  
    - data_preparation.sh  
    - training_vae.sh  
    - [other .sh files]  
  - zdcfastsim/data_preparation/Extract.C
  - grid_submit.sh (from: https://github.com/AliceO2Group/O2DPG/blob/master/GRID/utils/grid_submit.sh)
    
On the alibi:
- HOME
  - alice, containing alice O2 software built up to the Python-modules stage
  - python-scripts  
    - preprocess_txt_to_npz_protons.py  
    - retrain_VAE_protons.py  
    - [other .py files]  
    
  
The pipeline can be run using a pipeline.sh script:  
sh pipeline.sh o2_tag simulated_data_dir alibi_home model_name  
  - o2_tag: o2 tag used for simulating data on the grid  
  - simulated_data_dir: full path to the dir for saving simulated data  
  - alibi_home: path to the home directory on the alibi, /home/_user_  
  - model_name: name for a trained model  

After running this file, trained model weights are saved on the alibi. In order to convert the model to the onnx format, one should perform operations similar to those presented in the tf2onnx.ipynb notebook.  
  
## Files in the repo  
- bash-scripts  
  - copy_simulated_data_to_alibi.sh: helper script for copying simulated fullsim data to alibi.  
  - data_preparation.sh: batch script for converting txt to npz files (by calling python script).    
  - datasimulation_parallel.sh: helper script for simulating data in parallel.  
  - datasimulation_pipeline.sh: script responsible for simulating data on the grid and copying it to the machine. Calls other scripts.  
  - extract_data.sh: helper script for extracting data from the .root simulation file and moving it into the final destination.  
  - pipeline.sh: script for performing the whole ML pipeline.  
  - training_vae.sh: batch script for training the VAE model. Calls python script.  
data_preparation.sh, datasimulation_pipeline.sh, training_vae.sh can be used standalone.  
- python-scripts  
  - preprocess_txt_to_npz_protons.py: script for preprocessing txt files from simulations to npz files. Based on preprocess_txt_to_npz.py from this repo.  
  - retrain_VAE_protons.py: script for training the VAE model on protons data. Based on retrain_VAE.py from this repo.    
- tf2onnx.ipynb: notebook showing how to convert tensorflow model to onnx format.  


