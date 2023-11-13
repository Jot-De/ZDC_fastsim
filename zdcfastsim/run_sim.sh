#!/bin/bash

cd ..
mkdir -p run-gan-digit-fix

export ALICEO2_CCDB_LOCALCACHE=$PWD/.ccdb
export IGNORE_VALIDITYCHECK_OF_CCDB_LOCALCACHE=ON

#run original full sim
o2-sim -n 200 -e TGeant3 -g pythia8pp -m HALL PIPE ITS TPC ZDC -o run-gan-digit-fix/full_sim  --configKeyValues "align-geom.mDetectors=none;ZDCSimParam.recordSpatialResponse=true" 
#digitize full sim hits
mkdir -p ./run-gan-digit-fix/full_sim_digit
cd ./run-gan-digit-fix/full_sim_digit
o2-sim-digitizer-workflow  -b --onlyDet ZDC --sims ../full_sim
#run reconstruction
o2-zdc-digits-reco -b --run
cd ../.. 


#run machine learning fastsim
o2-sim -n 200 -g pythia8pp -m HALL PIPE ITS TPC ZDC -o run-gan-digit-fix/fast_sim --configKeyValues "align-geom.mDetectors=none;ZDCSimParam.useZDCFastSim=true;ZDCSimParam.ZDCFastSimClassifierPath=${PWD}/../alice/models/eon_classifier.onnx;ZDCSimParam.ZDCFastSimModelPathNeutron=${PWD}/../alice/models/sdi_gan.onnx;ZDCSimParam.ZDCFastSimModelScalesNeutron=${O2_ROOT}/share/Detectors/ZDC/fastsimulation/scales/gan_scales.txt;ZDCSimParam.ZDCFastSimModelPathProton=${PWD}/../alice/models/protons_vae/generator-protons-vae.onnx;ZDCSimParam.ZDCFastSimModelScalesProton=${PWD}/../alice/models/protons_vae/vae_scales.txt;ZDCSimParam.ZDCFastSimClassifierScales=${O2_ROOT}/share/Detectors/ZDC/fastsimulation/scales/eon_scales.txt;SimCutParams.maxAbsZTracking=1200"
#digitize fast sim hits
mkdir -p ./run-gan-digit-fix/fast_sim_digit
cd ./run-gan-digit-fix/fast_sim_digit
o2-sim-digitizer-workflow  -b --onlyDet ZDC --sims ../fast_sim
#run reconstruction
o2-zdc-digits-reco -b --run
cd ../..
