#include "core.h"
#include "neuralnet.h"

using namespace torch;

class TESTNET : public nn::Module {
public:
    nn::Linear linear1{1028,128}, linear2{128, 10};
    nn::ReLU relu;

}
