#!/bin/bash

alibi_home=$1
simulated_data_dir=$2

if ssh alibicompute01.cern.ch "[ -d $alibi_home/$simulated_data_dir ] || [ -f $alibi_home/$simulated_data_dir.tar.gz ]"
then
    echo "Data directory or tar.gz file exists"
    exit 1
fi

scp $simulated_data_dir.tar.gz alibicompute01.cern.ch:/$alibi_home/
