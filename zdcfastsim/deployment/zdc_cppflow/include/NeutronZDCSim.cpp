//
// Created by jan on 20.09.2021.
//

#include "NeutronZDCSim.h"
#include <string>
#include <iostream>
#include "cppflow/ops.h"
#include "cppflow/model.h"
#include "NeutronZDCSim.h"
#include <array>
#include <math.h>

NeutronZDCSim::NeutronZDCSim(string modelPath) : modelGenerator(modelPath)
{
    modelPath = modelPath;
    conditionalMeans = {1366.6713947506341,
                        -7.843879558672188e-08,
                        1.4495821821338432e-07,
                        0.001066061348953186,
                        0.009912433153308505,
                        0.019852346109538215,
                        39.10844489342062,
                        353.66195478091436,
                        0.06293706293706294};
    conditionalScales = {1259.76149755515,
                         4.15467111085297e-06,
                         4.416443169359055e-06,
                         0.024527054020515925,
                         0.23456173508581968,
                         0.25811363111381547,
                         1858.2947449457529,
                         454.9668554829095,
                         0.2565348696256425};
    noiseStdev = 0.0;

}


vector<float> NeutronZDCSim::scaleConditionalInput(array<float, 9> rawConditionalInput) {
    vector<float> scaledConditionalInput = {0,0,0,0,0,0,0,0,0};
    for(int var=0; var<9; var++) {
        scaledConditionalInput[var] = (rawConditionalInput[var] - conditionalMeans[var]) / conditionalScales[var];
    }
    return scaledConditionalInput;
}


cppflow::tensor NeutronZDCSim::generateResponse(cppflow::tensor input_1, cppflow::tensor input_2)
{
    // Calling model with two inputs, named "serving_default_input_1" and "serving_default_input_2"
    auto output = modelGenerator({{"serving_default_input_1:0", input_1}, {"serving_default_input_2:0", input_2}},{"StatefulPartitionedCall:0"});

    //print output
    //std::cout << output[0] << std::endl;

    return output[0];
}



array<int, 5> NeutronZDCSim::calculateChannelsFromResponse(cppflow::tensor calorimeterImage)
{
    array<float, 5> channels = {0}; //4 photon channels
    vector<float> flattedImageVector = calorimeterImage.get_data<float>();
    for (int i = 0; i<44; i++) {
        for (int j = 0; j<44; j++){
            if (i % 2 == j % 2) {
                if (i<22 && j<22) { channels[0] = channels[0] + flattedImageVector[i+j*44]; }
                else if (i>=22 && j<22) { channels[1] = channels[1] + flattedImageVector[i+j*44]; }
                else if (i<22 && j>=22) { channels[2] = channels[2] + flattedImageVector[i+j*44]; }
                else if (i>=22 && j>=22) { channels[3] = channels[3] + flattedImageVector[i+j*44]; }
            } else channels[4] = channels[4] + flattedImageVector[i+j*44];
        }
    }
    array<int, 5> channels_integers = {0};
    for (int ch = 0; ch < 5; ch++) { channels_integers[ch] = round(channels[ch]); }
    return channels_integers;
}


array<int, 5> NeutronZDCSim::calculatePhotons(array<float,9> unscaledConditionalInput)
/**
 * Calculates number of photons for 5 channels in response to input particle of provided properties.
 * @param unscaledConditionalInput nine conditional variables describing the input particle:
 * {'Energy','Vx','Vy',	'Vz',	'Px',	'Py',	'Pz',	'mass',	'charge'}
 * @return number of photons for 5 channels
 */
{
    auto conditionalInput = cppflow::tensor(scaleConditionalInput(unscaledConditionalInput), {1, 9});
    cppflow::tensor noiseInput =  cppflow::random_standard_normal({1,10},TF_FLOAT) * noiseStdev;

    cppflow::tensor responseImage = generateResponse(noiseInput, conditionalInput);
    array<int, 5> photons = calculateChannelsFromResponse(responseImage);
    return photons;
}

