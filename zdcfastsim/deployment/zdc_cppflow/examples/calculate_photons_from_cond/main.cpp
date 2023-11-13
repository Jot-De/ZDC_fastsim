#include <iostream>

#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "NeutronZDCSim.h"

int main() {
    // sample input
    array<float,9> exampleConditionalData = {5.13318e+02,  1.45430e-08,  3.65051e-08, -2.73101e-03,
                         3.54560e-02, -5.18206e-02, -5.13318e+02,  0.00000e+00,
                         0.00000e+00};


    NeutronZDCSim gen("../models/VAE-generator");


    array<int,5> photons = gen.calculatePhotons(exampleConditionalData);

    for(int ch=0; ch<5; ch++) {
        cout << "ch" << ch+1 << " " << photons[ch] << " | ";
    }

    return 0;
}


