# this is the training script that can run on the GRID
# sim serial is needed for spatial response !!

SEED=-1
# we have 8 cores in a GRID job
for i in `seq 1 8`; do
    SEED=${ALIEN_PROC_ID:-1}
    SEED=${SEED:1}
    SEED="${i}${SEED}"
    # select number of events with -n (500 may be reasonable for GRID job)
    o2-sim-serial -e TGeant3 -g pythia8pp -n 100 --configKeyValues "SimCutParams.maxRTracking=100;ZDCSimParam.recordSpatialResponse=true" --seed ${SEED} --readoutDetectors ZDC --run 302000 -o sim${i} &> simlog${i} &
done

wait

# fetch the zdcfastsim code (for data conversion) with git
if [ ! -d "zdcfastsim" ]; then
   git clone https://gitlab.cern.ch/swenzel/zdcfastsim.git
fi

if [ ! -d "zdcfastsim" ]; then
  exit 1
fi

# convert to training images
for i in `seq 1 8`; do
  file="sim${i}.root"
  if [ -f "$file" ]; then
    C="zdcfastsim/data_preparation/Extract.C(\"${file}\")"
    root -q -b -l ${C}
  
    mv zero_examples.txt zero_examples_run${i}.txt
    mv non_zero_examples.txt non_zero_examples_run${i}.txt
    mv proton_image.txt proton_image_run${i}.txt
    mv neutron_image.txt neutron_image_run${i}.txt
  fi
done

exit 0
