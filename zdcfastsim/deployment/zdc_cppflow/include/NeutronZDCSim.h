////
//// Created by jan on 20.09.2021.
////

#ifndef EXAMPLE_NEUTRONZDCSIM_H
#define EXAMPLE_NEUTRONZDCSIM_H
#include <string>
#include "cppflow/ops.h"
#include "cppflow/model.h"
#include <array>

using namespace std;

class NeutronZDCSim {

    private:
        string modelPath;
        cppflow::model modelGenerator;
        array<float, 9> conditionalMeans;
        array<float, 9> conditionalScales;
        float noiseStdev;

    public:
            explicit NeutronZDCSim(string modelPath);
            cppflow::tensor generateResponse(cppflow::tensor input_1, cppflow::tensor input_2);
            vector<float> scaleConditionalInput(array<float, 9>);
            array<int, 5> calculateChannelsFromResponse(cppflow::tensor calorimeterImage);
            array<int, 5> calculatePhotons(array<float,9> unscaledConditionalInput);
            //Unscaled conditional input is a float array of nine variables:
            //{'Energy','Vx','Vy',	'Vz',	'Px',	'Py',	'Pz',	'mass',	'charge'}

};


#endif //EXAMPLE_NEUTRONZDCSIM_H
