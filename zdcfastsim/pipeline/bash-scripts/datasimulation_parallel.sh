#!/bin/bash

n_jobs=8

for (( i=0; i<$n_jobs; i++))
do
  o2-sim-serial -n 1 -e TGeant3 -g pythia8pp -m HALL PIPE ITS TPC ZDC --configKeyValues "align-geom.mDetectors=none;ZDCSimParam.recordSpatialResponse=true" -o fullsim$i &
done

wait
