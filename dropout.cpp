#include "dropout.h"
#include "ops1.h" // For mul_, etc.
#include <stdexcept>
#include <random>
#include <ctime>

// Constructor
Dropout::Dropout(double p) : p(p) {
    if (p < 0.0 || p >= 1.0) 
        throw std::invalid_argument("Dropout: p must be in [0, 1)");
}
