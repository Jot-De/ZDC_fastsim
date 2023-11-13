#!/bin/bash
#SBATCH --job-name=train_VAE
#SBATCH --output=$alibi_home/%x_%j.out
#SBATCH --error=$alibi_home/%x_%j.err
#SBATCH --mem=50G
#SBATCH --time=03:00:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#exported environment variables: alibi_home, model_name, data_dir


cd $alibi_home

cd alice
./aliBuild/alienv printenv Python-modules/latest-o2 > env.env
source env.env
cd ..

python3 python-scripts/retrain_VAE_protons.py --name $model_name --data $data_dir

