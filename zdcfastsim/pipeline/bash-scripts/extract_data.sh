#!/bin/bash

# Script for extracting data from the .root simulation file and moving it into the final destination.
# Args: n_jobs, run_no, job_no, simulation_dir, final_data_dir

cd ~/$4/$2
tmp_dir=tmp_$3
mkdir $tmp_dir
cd $tmp_dir
root -q -b -l "~/zdcfastsim/data_preparation/Extract.C(\"~/$4/$2/fullsim$3.root\")"
i=$((10#$2))  # remove leading zeros
run_no=$(( $1*$(( $i-1 ))+$3 ))
mv ~/$4/$2/$tmp_dir/non_zero_examples.txt ~/$5/non_zero_examples_run$run_no.txt
mv ~/$4/$2/$tmp_dir/zero_examples.txt ~/$5/zero_examples_run$run_no.txt
mv ~/$4/$2/$tmp_dir/neutron_image.txt ~/$5/neutron_image_run$run_no.txt
mv ~/$4/$2/$tmp_dir/proton_image.txt ~/$5/proton_image_run$run_no.txt
rm -r ~/$4/$2/$tmp_dir


