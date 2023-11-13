#!/bin/bash
#SBATCH --job-name=data_preparation
#SBATCH --output=$alibi_home/%x_%j.out
#SBATCH --error=$alibi_home/%x_%j.err
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

#exported environment variables: alibi_home, in_dir

out_dir=${in_dir}_output

cd $alibi_home

if [ -d $alibi_home/$out_dir ]
then
    echo "Output data directory exists"
    exit 1
fi
mkdir $out_dir

cd alice
./aliBuild/alienv printenv Python-modules/latest-o2 > env.env
source env.env
pip3 install particle

cd ..
tar -xf $in_dir.tar.gz
python3 python-scripts/preprocess_txt_to_npz_protons.py $in_dir $out_dir

