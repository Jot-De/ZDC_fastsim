// matches a track from the kinematics with the TParticle associated to the response
bool isClose(double x, double y) {
  return std::abs( x - y) < 1E-4;
}

bool matchTrack(o2::MCTrack const& t, o2::MCTrack const &p) {
  bool matches = true;
  matches &= isClose(t.Vx(), p.Vx());
  matches &= isClose(t.Vy(), p.Vy());
  matches &= isClose(t.Vz(), p.Vz());
  matches &= isClose(t.Px(), p.Px());
  matches &= isClose(t.Py(), p.Py());
  matches &= isClose(t.Pz(), p.Pz());
  matches &= isClose(t.T(), p.T());
  matches &= isClose(t.GetEnergy(), p.GetEnergy());
  matches &= t.GetPdgCode() == p.GetPdgCode();
  return matches;
}

void printHeader(std::ostream &str) {
  str << "Detector," << "Pdg," << "Energy," << "Vx," << "Vy," << "Vz," << "Px," << "Py," << "Pz," << "PhotonSum\n";
}

void printTrack(std::ostream &str, const char* det, int photons, o2::MCTrack const& t) {
  str << det << "," << t.GetPdgCode() << "," << t.GetEnergy() << ","
      << t.Vx() << "," << t.Vy() << "," << t.Vz() << ","
      << t.Px() << "," << t.Py() << "," << t.Pz() << "," << photons << "\n";
}

void Extract(const char* simfile = "o2sim.root") {
  TFile file(simfile, "OPEN");
  auto tree = (TTree*)file.Get("o2sim");
  if (!tree) {
    return;
  }
  
  const auto nevents = tree->GetEntries();
  std::cout << "Have " << nevents << "event \n";
   
  // load hit data
  const auto hitbranch = tree->GetBranch("ZDCResponseImage");
  if (!hitbranch) {
    std::cerr << "No response branch\n";
    return;
  }
  
  std::vector<std::pair<TParticle, std::pair<o2::zdc::SpatialPhotonResponse, o2::zdc::SpatialPhotonResponse>>> *zdcresponse = nullptr;
  hitbranch->SetAddress(&zdcresponse);

  // load the kinematics data (containing all primaries)
  const auto mctrackbranch = tree->GetBranch("MCTrack");
  std::vector<o2::MCTrack> *mctracks = nullptr;
  mctrackbranch->SetAddress(&mctracks);

  std::ofstream zero_examples;
  zero_examples.open ("zero_examples.txt");

  std::ofstream non_zero_examples;
  non_zero_examples.open ("non_zero_examples.txt");

  std::ofstream neutron_image;
  neutron_image.open ("neutron_image.txt");

  std::ofstream proton_image;
  proton_image.open ("proton_image.txt");

  printHeader(zero_examples);
  printHeader(non_zero_examples);

  // loop over events
  for (int ev = 0; ev < nevents; ++ev) {
    hitbranch->GetEntry(ev);
    mctrackbranch->GetEntry(ev);

    std::vector<o2::MCTrack> allPrimaries;
    // stupid way of fetching all primaries
    for (const auto& t : *mctracks) {
      if (t.getMotherTrackId() == -1) {
        allPrimaries.push_back(t);
      }
    }
    std::vector<o2::MCTrack> goodTracks; // collect primary particle that left an impact

    if (zdcresponse == nullptr) {
      std::cerr << "Could not read data";
      return;
    }

    // loop over all hits and printout its properties -- fetch
    for (auto& particleimagepair : *zdcresponse) {
      auto& imageNeutron = particleimagepair.second.first;
      auto& imageProton = particleimagepair.second.second;
      
      auto& part = particleimagepair.first;

      if(imageNeutron.getPhotonSum()>0 || imageProton.getPhotonSum()>0 ) {
        goodTracks.emplace_back(part);
        printTrack(non_zero_examples, "N", imageNeutron.getPhotonSum(), goodTracks.back());
        printTrack(non_zero_examples, "P", imageProton.getPhotonSum(), goodTracks.back());

	std::vector< std::vector<int> > tab;
	tab = imageNeutron.getImageData();
        // std::cout<<tab.size()<<" "<<tab[0].size()<<"\n";
        for( int i = 0; i < tab.size(); i++ ){
	   for( int j = 0; j < tab[0].size(); j++ ){
	      neutron_image << tab[i][j] << " ";
	   }
	   neutron_image<<"\n";
	}
	std::vector< std::vector<int> >().swap(tab);
	tab = imageProton.getImageData();
	// std::cout<<tab.size()<<" "<<tab[0].size()<<"\n";
	for( int i = 0; i < tab.size(); i++ ){
	   for( int j = 0; j < tab[0].size(); j++ ){
	      proton_image << tab[i][j] << " ";
	   }
	   proton_image<<"\n";
	}
      } // end if
    }

    // at this moment we have written the good examples
    // now let's collect examples without response
    for (auto track : allPrimaries) {
      for(auto good : goodTracks) {
        if (!matchTrack(track,good)) {
          printTrack(zero_examples, "N", 0, track);
          printTrack(zero_examples, "P", 0, track);
        }
      }
    }

  }
  zero_examples.close();
  non_zero_examples.close();
  neutron_image.close();
  proton_image.close();
}
