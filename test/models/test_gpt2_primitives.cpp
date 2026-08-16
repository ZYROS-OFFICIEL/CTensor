#include <iostream>
#include <cassert>
#include <cmath>
#include "neuralnet.h"
#include "neuralnet/norm/layernorm.h"

static bool close(double a, double b, double tol = 1e-4) {
    return std::abs(a - b) < tol;
}

void test_softmax() {
    Tensor x = Tensor::from_vector({1.0, 2.0, 3.0, 1.0, 1.0, 1.0}, {2, 3});
    Tensor y = functional::softmax(x, -1);

    for (size_t r = 0; r < 2; ++r) {
        double s = 0.0;
        for (size_t c = 0; c < 3; ++c) s += y.read_scalar(r * 3 + c);
        assert(close(s, 1.0));
    }
    // uniform row -> each entry 1/3
    assert(close(y.read_scalar(3), 1.0 / 3.0));
    assert(close(y.read_scalar(4), 1.0 / 3.0));
    assert(close(y.read_scalar(5), 1.0 / 3.0));
}

void test_embedding_lookup() {
    Tensor weight = Tensor::from_vector(
        {0,0,0, 1,1,1, 2,2,2, 3,3,3}, {4, 3}, DType::Float32, true);
    Tensor indices = Tensor::from_vector({1, 3}, {2}, DType::Int32);

    Tensor out = embedding_lookup(weight, indices);
    assert(out.shape().size() == 2 && out.shape()[0] == 2 && out.shape()[1] == 3);
    for (size_t d = 0; d < 3; ++d) {
        assert(close(out.read_scalar(d), 1.0));
        assert(close(out.read_scalar(3 + d), 3.0));
    }

    Tensor loss = sum(out.flatten(), 0);
    loss.backward();
    Tensor g = weight.grad();
    // rows 1 and 3 should have gradient 1 (each used once), rows 0 and 2 should be 0
    for (size_t d = 0; d < 3; ++d) {
        assert(close(g.read_scalar(0 * 3 + d), 0.0));
        assert(close(g.read_scalar(1 * 3 + d), 1.0));
        assert(close(g.read_scalar(2 * 3 + d), 0.0));
        assert(close(g.read_scalar(3 * 3 + d), 1.0));
    }
}

void test_layernorm() {
    LayerNorm ln(4);
    Tensor x = Tensor::from_vector({1, 2, 3, 4, 10, 20, 30, 40}, {2, 4});
    Tensor y = ln(x);

    for (size_t r = 0; r < 2; ++r) {
        double mean = 0.0;
        for (size_t c = 0; c < 4; ++c) mean += y.read_scalar(r * 4 + c);
        mean /= 4.0;
        double var = 0.0;
        for (size_t c = 0; c < 4; ++c) {
            double d = y.read_scalar(r * 4 + c) - mean;
            var += d * d;
        }
        var /= 4.0;
        assert(close(mean, 0.0, 1e-3));
        assert(close(var, 1.0, 1e-2));
    }
}

void test_bmm() {
    // A: [2,2,3], B: [2,3,2]
    Tensor A = Tensor::from_vector(
        {1,2,3, 4,5,6,
         1,0,0, 0,1,0}, {2, 2, 3});
    Tensor B = Tensor::from_vector(
        {1,0, 0,1, 1,1,
         2,0, 0,2, 1,1}, {2, 3, 2});

    Tensor C = bmm(A, B);
    assert(C.shape()[0] == 2 && C.shape()[1] == 2 && C.shape()[2] == 2);

    // batch 0, row 0: [1,2,3] . cols -> [1*1+2*0+3*1, 1*0+2*1+3*1] = [4, 5]
    assert(close(C.read_scalar(0), 4.0));
    assert(close(C.read_scalar(1), 5.0));
    // batch 1, row 0: [1,0,0] . cols -> [2, 0]
    assert(close(C.read_scalar(4), 2.0));
    assert(close(C.read_scalar(5), 0.0));

    // gradient flow + finite-difference check on one entry of A
    Tensor Ag = A.clone();
    Ag.requires_grad_(true);
    Tensor Bg = B.clone();
    Bg.requires_grad_(true);
    Tensor loss = sum(bmm(Ag, Bg).flatten(), 0);
    loss.backward();

    double eps = 1e-3;
    Tensor A_plus = A.clone();
    A_plus.write_scalar(0, A_plus.read_scalar(0) + eps);
    double loss_plus = 0.0;
    {
        Tensor Cp = bmm(A_plus, B);
        for (size_t i = 0; i < Cp.numel(); ++i) loss_plus += Cp.read_scalar(i);
    }
    Tensor A_minus = A.clone();
    A_minus.write_scalar(0, A_minus.read_scalar(0) - eps);
    double loss_minus = 0.0;
    {
        Tensor Cm = bmm(A_minus, B);
        for (size_t i = 0; i < Cm.numel(); ++i) loss_minus += Cm.read_scalar(i);
    }
    double numerical_grad = (loss_plus - loss_minus) / (2 * eps);
    double analytical_grad = Ag.grad().read_scalar(0);
    assert(close(numerical_grad, analytical_grad, 1e-2));
}

void test_linear_3d() {
    Linear fc(4, 6);
    Tensor x = Tensor::ones({2, 3, 4}, DType::Float32, true);
    Tensor y = fc(x);
    assert(y.shape().size() == 3 && y.shape()[0] == 2 && y.shape()[1] == 3 && y.shape()[2] == 6);

    Tensor loss = sum(y.flatten(), 0);
    loss.backward();
    Tensor g = x.grad();
    assert(g.impl != nullptr);
    for (size_t i = 0; i < g.numel(); ++i) assert(std::isfinite(g.read_scalar(i)));

    // 2D path still works identically
    Tensor x2 = Tensor::ones({5, 4});
    Tensor y2 = fc(x2);
    assert(y2.shape()[0] == 5 && y2.shape()[1] == 6);
}

int main() {
    test_softmax();
    test_embedding_lookup();
    test_layernorm();
    test_bmm();
    test_linear_3d();
    std::cout << "test_gpt2_primitives passed\n";
    return 0;
}
