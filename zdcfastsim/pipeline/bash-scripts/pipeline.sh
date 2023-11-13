#!/bin/bash

# File for running the whole ML pipeline, that is:
#   - data simulation (on the grid)
#   - preparing simulated data for training (on the alibi)
#   - training a model (on the alibi)
# This pipeline involves training a proton VAE model. 
#
# Running the script: sh pipeline.sh o2_tag simulated_data_dir alibi_home model_name
#   - o2_tag: o2 tag used for simulating data on the grid
#   - simulated_data_dir: full path to the dir for saving simulated data
#   - alibi_home: path to the home directory on the alibi, /home/<user>
#   - model_name: name for a trained model


o2_tag=$1
simulated_data_dir=$2
alibi_home=$3
model_name=$4

cd ~

#simulate data
echo "Start data simulation"
sh ~/bash-scripts/datasimulation_pipeline.sh $o2_tag $simulated_data_dir
echo "End data simulation"

#copy simulated data to the alibi
echo "Start copying data to the alibi"
salloc sh ~/bash-scripts/copy_simulated_data_to_alibi.sh $alibi_home $simulated_data_dir
echo "End copying data to the alibi"

#sbatch the data preparation job and wait for it to finish
echo "Sbatch data preparation job"
jobid=$(sbatch --export=ALL,alibi_home=$alibi_home,in_dir=$simulated_data_dir --output=$alibi_home/%x_%j.out --error=$alibi_home/%x_%j.err --parsable ~/bash-scripts/data_preparation.sh)
while [[ "$(squeue | grep $jobid | wc -l)" -ne 0 ]]
do
  echo "Waiting for data preparation job to finish"
  sleep 10
done
echo "Data preparation job has finished"

#sbatch the model training job and wait for it to finish 
echo "Sbatch model training job"
jobid=$(sbatch --export=ALL,alibi_home=$alibi_home,model_name=$model_name,data_dir=${simulated_data_dir}_output --output=$alibi_home/%x_%j.out --error=$alibi_home/%x_%j.err --parsable ~/bash-scripts/training_vae.sh)
while [[ "$(squeue | grep $jobid | wc -l)" -ne 0 ]]
do
  echo "Waiting for model training job to finish"
  sleep 600
done
echo "Model training job has finished. Trained weights can be found in the alibi directory specified in the python training script (e.g. models)"

