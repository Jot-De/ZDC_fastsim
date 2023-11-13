## ZDC FastMLSim

# Project

- “Given a track entering ZDC (proton or neutron calo), generate/predict its photon response (number of photons generated in detector(s)”
- “Stop track leaving central barrel early if it will not have any impact in ZDC”

# Ideas:

- Use GAN or similar deep learning approaches

# Training data:

## simple low dimensional case

- tuples { entering track, a list of at most 3 integer pairs denoting the number of photons in some channel }
    - track will be something like (pdg, energy, entering position, direction, ….)
    - example: `{ {trackID, pdg, energy, posx, posy, posz } -> { {3, 0} , {4, 5}, { 20, 20 } } }`
- script ExtractData.C can produce the training data by going through the simulated data.
 

## higher dimensional data:

- vectors of { entering track, pair<NeutronImage, ProtonImage> } where each imagepixel has some integer number of photons.
- script ExtractResponseImages.C to extract this from simulated data