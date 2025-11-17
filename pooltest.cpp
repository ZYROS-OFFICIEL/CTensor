#include "tensor1.h"
#include "ops1.h"
#include "autograd.h" // <-- FIX: Include autograd
#include "pooling.h"  // <-- FIX: Include pooling
#include <iostream>
#include <cassert>
#include <vector> // <-- FIX: Include vector
#include <cmath>  // <-- FIX: Include cmath for fabs

using namespace std;

// Utility to compare tensors
void assert_allclose(const Tensor& a, const Tensor& b, float eps = 1e-5) {
    if (a.numel() != b.numel()) {
        cout << "Mismatch numel: " << a.numel() << " vs " << b.numel() << endl;
        assert(false);
    }
    for (size_t i = 0; i < a.numel(); i++) { // <-- FIX: Use size_t
        if (fabs(a.read_scalar(i) - b.read_scalar(i)) > eps) {
            cout << "Mismatch at index " << i
                 << ": " << a.read_scalar(i) << " vs " << b.read_scalar(i) << endl;
            assert(false);
        }
    }
}

void test_maxpool1d() {
    cout << "Running MaxPool1d test..." << endl;

    // --- FIX: Use Tensor::from_vector constructor ---
    Tensor x = Tensor::from_vector({1, 3, 2, 5, 0, 4}, {1,1,6}, DType::Double64, true); 
    MaxPool1d mp(2, 2);   // kernel 2, stride 2

    Tensor y = mp(x);

    // Expected: max over windows:
    // [1,3] -> 3
    // [2,5] -> 5
    // [0,4] -> 4
    Tensor expected = Tensor::from_vector({3,5,4}, {1,1,3});

    assert_allclose(y, expected);

    // backward test:
    Tensor dy = Tensor::from_vector({1,1,1}, {1,1,3}, DType::Double64, false);
    
    // --- FIX: Use correct backward API ---
    // We want to compute backward pass with dy as the upstream gradient.
    // This is equivalent to sum(y * dy).backward()
    Tensor loss = sum(y * dy);
    backward(loss); // This function (from autograd.h) zeros grads, sets loss.grad=1, and backprops

    // Only max positions get gradient:
    // input: [1,3,2,5,0,4]
    //             ^   ^   ^
    // grad:  [0,1,0,1,0,1]
    Tensor grad_expected = Tensor::from_vector({0,1,0,1,0,1}, {1,1,6});

    // --- FIX: Use correct grad checking API ---
    assert_allclose(tensor_from_grad(x), grad_expected);
}

void test_avgpool1d() {
    cout << "Running AvgPool1d test..." << endl;

    Tensor x = Tensor::from_vector({2,4,6,8,10,12}, {1,1,6}, DType::Double64, true);
    AvgPool1d ap(2, 2);

    Tensor y = ap(x);

    // expected: mean of each pair
    // [2,4] -> 3
    // [6,8] -> 7
    // [10,12] -> 11
    Tensor expected = Tensor::from_vector({3,7,11}, {1,1,3});

    assert_allclose(y, expected);

    // backward test:
    Tensor dy = Tensor::from_vector({1,1,1}, {1,1,3}, DType::Double64, false);
    
    // --- FIX: Use correct backward API ---
    Tensor loss = sum(y * dy);
    backward(loss);

    // avg distributes gradients:
    // each input in window gets 0.5
    Tensor grad_expected = Tensor::from_vector({
        0.5,0.5,
        0.5,0.5,
        0.5,0.5
    }, {1,1,6});

    // --- FIX: Use correct grad checking API ---
    assert_allclose(tensor_from_grad(x), grad_expected);
}

void test_maxpool2d() {
    cout << "Running MaxPool2d test..." << endl;

    // input shape: (1,1,2,4)
    Tensor x = Tensor::from_vector({
        1,5,2,4,
        7,0,3,6
    }, {1,1,2,4}, DType::Double64, true);

    MaxPool2d mp(2,2,2,2);

    Tensor y = mp(x);

    // windows:
    // [[1,5],[7,0]] → 7
    // [[2,4],[3,6]] → 6
    Tensor expected = Tensor::from_vector({7,6}, {1,1,1,2});

    assert_allclose(y, expected);

    // backward:
    Tensor dy = Tensor::from_vector({1,1}, {1,1,1,2}, DType::Double64, false);

    // --- FIX: Use correct backward API ---
    Tensor loss = sum(y * dy);
    backward(loss);

    Tensor grad_expected = Tensor::from_vector({
        0,0,0,0,
        1,0,0,1
    }, {1,1,2,4});

    // --- FIX: Use correct grad checking API ---
    assert_allclose(tensor_from_grad(x), grad_expected);
}

void test_avgpool2d() {
    cout << "Running AvgPool2d test..." << endl;

    Tensor x = Tensor::from_vector({
        1,3,5,7,
        2,4,6,8
    }, {1,1,2,4}, DType::Double64, true);

    AvgPool2d ap(2,2,2,2);

    Tensor y = ap(x);

    // windows:
    // [1,3,2,4] → (1+3+2+4)/4 = 2.5
    // [5,7,6,8] → (5+7+6+8)/4 = 6.5
    Tensor expected = Tensor::from_vector({2.5, 6.5}, {1,1,1,2});

    assert_allclose(y, expected);

    // backward:
    Tensor dy = Tensor::from_vector({1,1}, {1,1,1,2}, DType::Double64, false);

    // --- FIX: Use correct backward API ---
    Tensor loss = sum(y * dy);
    backward(loss);

    Tensor grad_expected = Tensor::from_vector({
        0.25,0.25,0.25,0.25,
        0.25,0.25,0.25,0.25
    }, {1,1,2,4});

    // --- FIX: Use correct grad checking API ---
    assert_allclose(tensor_from_grad(x), grad_expected);
}

int main() {
    // We need autograd.h, ops1.h, and tensor1.h for the test to compile
    // Assuming they are linked
    test_maxpool1d();
    test_avgpool1d();
    test_maxpool2d();
    test_avgpool2d();

    cout << "All pooling tests passed!" << endl;
    return 0;
}