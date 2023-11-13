
// simple script that takes ZDC photon response images (in this case from a simulation)
// and transforms them directly to hits
void ZDCFastSim(const char* simfile="o2sim.root") {
  using namespace o2::zdc;

  // setup the detector needed to convert images to hits
  o2::zdc::Detector det;

  // open the simefile
  TFile infile(simfile);
  auto intree = (TTree*)infile.Get("o2sim");

  const auto hitbranch = intree->GetBranch("ZDCResponseImage");
  std::vector<std::pair<TParticle, std::pair<o2::zdc::SpatialPhotonResponse, o2::zdc::SpatialPhotonResponse>>> *zdcresponse = nullptr;
  hitbranch->SetAddress(&zdcresponse);

  // setup the output file
  TFile outfile("zdcfastsimhits.root", "RECREATE");
  // setup the output tree
  TTree tree("o2sim", "o2sim");
  // setup the output branch
  std::vector<o2::zdc::Hit> *hits = det.getHits(0);
  auto outbranch = tree.Branch("ZDCHits", &hits);
 
  for (int event = 0; event < intree->GetEntries(); ++event) {
    hitbranch->GetEntry(event);

    // loop over all images
    for (auto& response : *zdcresponse) {
      // extract neutron image
      auto& neutronimage = response.second.first;

      // TODO: need to distinguish between ZNA and ZNC
      det.createHitsFromImage(neutronimage, ZNA);

      // extract proton image
      auto& protonimage = response.second.second;
      det.createHitsFromImage(protonimage, ZPA);
    }
    tree.Fill();
    det.EndOfEvent();
  }
  tree.Write();
  outfile.Close();
}
