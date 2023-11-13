#!/bin/bash

# Script for simulating data on the grid. Saves the data in the ~/data_dir ~/data_dir.tar.gz locations.
# Args: o2_tag - tag for simulating data, data_dir - directory for saving simulated files.

cd ~
o2_tag=$1
DATA_DIR=$2
timestamp=$(date +%s)
SIM_DIR=tmp_$timestamp
ENV=env_$timestamp

if [ -d $DATA_DIR ]
then
    echo "Data directory exists"
    exit 1
fi

mkdir $DATA_DIR
mkdir $SIM_DIR
alienv printenv O2sim::v20220721-1 > $ENV
source ~/$ENV

n_runs=30
n_jobs=8

cd $SIM_DIR
~/grid_submit.sh --script ~/bash-scripts/datasimulation_parallel.sh --prodsplit $n_runs --o2tag $o2_tag &> stdout.txt
jobname=$(cat stdout.txt | grep -o '".*"' | sort --unique | tr -d '"')
while [[ "$(alien.py ps -f d | grep $jobname | wc -l)" -ne $n_runs+1 ]]
do
  echo "Waiting for simulation to finish"
  sleep 10
done

echo "Data simulation finished. Copying files..."

for run_no in $(seq -f "%03g" 1 $n_runs)
do
  alien.py cp selfjobs/$jobname/$run_no -select fullsim[0-9]\*.root file://~/$SIM_DIR
done

echo "Copying finished. Extracting data..."

for run_no in $(seq -f "%03g" 1 $n_runs)
do
  for (( job_no=0; job_no<$n_jobs; job_no++ ))
  do
    sh ~/bash-scripts/extract_data.sh $n_jobs $run_no $job_no $SIM_DIR $DATA_DIR &
  done
  wait
done

cd ~
tar -czvf $DATA_DIR.tar.gz $DATA_DIR
rm -r ~/$ENV
rm -r ~/$SIM_DIR

