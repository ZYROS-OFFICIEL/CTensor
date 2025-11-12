#include <iostream>
#include <cmath>
#include "tensor1.h"
#include "ops1.h"
#include "autograd.h"
#include "loss.h"
#include <functional>
#include <iomanip>

static bool close(double a, double b, double tol=1e-5) {
    return std::fabs(a - b) < tol * std::max({1.0, std::fabs(a), std::fabs(b)});
}
void test_tensorftromgrad(){
    Tensor x = Tensor::zeros({2, 3}, DType::Float32);
    x.impl->storage->grad = std::shared_ptr<void>(malloc(x.numel()*4), free);
    float *p = (float*)x.impl->storage->grad.get();
    for (int i=0;i<x.numel();++i) p[i] = i + 1;

    Tensor g = tensor_from_grad(x);
    print_t(g);

}
void test_grad_storage_allocation() {
    Tensor x = Tensor::arange(0.0, 12.0, 1.0, DType::Float32).reshape({3,4});
    Tensor v = x.permute({1,0});  // view
    ensure_grad_buffer(v, true);
    std::cout << "storage size = " << v.impl->storage->size << std::endl;
    std::cout << "allocated grad bytes = " << v.impl->storage->size * v.dtype_bytes() << std::endl;
}

void test_accum_grad(){
    Tensor a({2,1}, DType::Float32);
    Tensor b = Tensor::ones({2,3}, DType::Float32);

    accumulate_grad(a, b);
    std::cout << "a.grad = "  ;
    print_t(tensor_from_grad(a));

}
double numerical_grad(std::function<double(double)> f, double x, double eps=1e-6) {
    return (f(x + eps) - f(x - eps)) / (2 * eps);
}
void test_huber_loss() {
    Tensor pred = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor target = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, false);

    Tensor loss = Loss::HuberLoss(pred, target, "mean", 1.0);
    std::cout << "Huber Loss: " << loss.read_scalar(0) << " expected 2.5\n";

    backward(loss);
    float* g = (float*)pred.impl->storage->grad.get();
    std::cout << "Grad pred: [" << g[0] << ", " << g[1] << "] expected [-1.0, -1.0]\n";
}
void test_mae() {
    Tensor pred = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor target = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, false);

    Tensor loss_mean = Loss::MAE(pred, target, "mean");
    std::cout << "MAE mean: " << loss_mean.read_scalar(0) << " expected 2.0\n";

    backward(loss_mean);
    float* g = (float*)pred.impl->storage->grad.get();
    std::cout << "Grad pred (mean): [" << g[0] << ", " << g[1] << "] expected [-0.5, -0.5]\n";
}

void test_cross_entropy() {
    using namespace std;

    cout << "=== Testing CrossEntropy Loss ===" << endl;

    Tensor pred = Tensor::from_vector({2.0, 1.0, 0.1,0.5, 2.5, 0.3}, {2, 3}, DType::Float32, true);
    Tensor target = Tensor::from_vector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0}, {2, 3}, DType::Float32, true);
    // Example: batch_size=2, num_classes=3

    // Forward pass
    Tensor loss = Loss::CrossEntropy(pred, target);
    cout << "Loss: " << loss.read_scalar(0) << endl;

    // Backward pass
    loss.backward();

    // Print gradients
    float* gpred = (float*)pred.impl->storage->grad.get();
    std::cout << "Grad pred: [" <<*gpred<< "]\n";
}

void check_mse() {
    std::cout << "=== MSE Loss Check ===\n";
    std::cout << std::fixed << std::setprecision(1);
    Tensor pred = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor target = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, false);
    Tensor loss = Loss::MSE(pred, target);
    std::cout << "MSE Loss: " << loss.read_scalar(0) << " expected 4.5\n";

    backward(loss);
    float* gpred = (float*)pred.impl->storage->grad.get();
    std::cout << "Grad pred: [" << gpred[0] << ", " << gpred[1] << "] expected [-2.0, -2.0]\n";
}
void check_grad_add() {
    std::cout << "=== Grad check: Add ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, true);
    Tensor y = add_(a, b);       // forward
    Tensor loss = sum(y);        // scalar loss
    backward(loss);              // compute backward
    float* ga = (float*)a.impl->storage->grad.get();
    float* gb = (float*)b.impl->storage->grad.get();
    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "]\n";
    std::cout << "Grad b: [" << gb[0] << ", " << gb[1] << "]\n";
}
void check_grad_mul_scalar() {
    std::cout << "=== Grad check: MulScalar ===\n";

    // Create test tensor
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    double scalar = 4.0;

    // Forward: y = a * scalar
    Tensor y = mult_scalar(a, scalar); // your operator for scalar mult
    Tensor loss = sum(y);              // simple sum to make grad = dy/da * dloss/dy
    backward(loss);

    // Read gradients
    float* ga = (float*)a.impl->storage->grad.get();

    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "]\n";
}

void check_grad_diff() {
    std::cout << "=== Grad check: Diff ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, true);
    Tensor y = diff_(a, b);       // forward
    Tensor loss = sum(y);        // scalar loss
    backward(loss);              // compute backward
    float* ga = (float*)a.impl->storage->grad.get();
    float* gb = (float*)b.impl->storage->grad.get();
    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "]\n";
    std::cout << "Grad b: [" << gb[0] << ", " << gb[1] << "]\n";
}
void check_grad_mul() {
    std::cout << "=== Grad check: Mul ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, true);
    Tensor y = mult_(a, b);      // elementwise multiply
    Tensor loss = sum(y);
    backward(loss);
    float* ga = (float*)a.impl->storage->grad.get();
    float* gb = (float*)b.impl->storage->grad.get();
    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "] expected b = [4,5]\n";
    std::cout << "Grad b: [" << gb[0] << ", " << gb[1] << "] expected a = [2,3]\n";
}
void mul_sclar() {
    std::cout << "=== scalr check: Mul ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, true);
    Tensor y = 5*a;      // elementwise multiply
    std::cout << " y: [" << y[0] << ", " << y[1] << "] \n";
}
void check_grad_div() {
    std::cout << "=== Grad check: Div ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({4.0, 5.0}, {2}, DType::Float32, true);
    Tensor y = div_(a, b);      // elementwise multiply
    Tensor loss = sum(y);
    backward(loss);
    float* ga = (float*)a.impl->storage->grad.get();
    float* gb = (float*)b.impl->storage->grad.get();
    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "] \n";
    std::cout << "Grad b: [" << gb[0] << ", " << gb[1] << "] \n";
}

void check_grad_pow() {
    std::cout << "=== Grad check: Pow ===\n";
    Tensor a = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor b = Tensor::from_vector({2.0, 3.0}, {2}, DType::Float32, true);
    Tensor y = pow_(a, b);
    Tensor loss = sum(y);
    backward(loss);

    float* ga = (float*)a.impl->storage->grad.get();
    float* gb = (float*)b.impl->storage->grad.get();
    std::cout << "Grad a: [" << ga[0] << ", " << ga[1] << "] expected b*a^(b-1) = [4,27]\n";
    std::cout << "Grad b: [" << gb[0] << ", " << gb[1] << "] expected a^b*ln(a) = [4*ln2,27*ln3]\n";
}

void check_grad_matmul() {
    std::cout << "=== Grad check: MatMul ===\n";
    Tensor A = Tensor::from_vector({1,2,3,4}, {2,2}, DType::Float32, true);
    Tensor B = Tensor::from_vector({5,6,7,8}, {2,2}, DType::Float32, true);
    Tensor Y = matmul_(A,B);
    Tensor loss = sum(Y);
    backward(loss);
    float* gA = (float*)A.impl->storage->grad.get();
    float* gB = (float*)B.impl->storage->grad.get();
    std::cout << "Grad A:\n";
    std::cout << gA[0] << " " << gA[1] << "\n" << gA[2] << " " << gA[3] << "\n";
    std::cout << "Grad B:\n";
    std::cout << gB[0] << " " << gB[1] << "\n" << gB[2] << " " << gB[3] << "\n";
}

int main() {
    test_grad_storage_allocation();
    test_accum_grad();
    test_tensorftromgrad();
    test_mae();
    test_cross_entropy();
    check_mse();
    check_grad_add();
    check_grad_diff();
    check_grad_mul();
    mul_sclar();
    check_grad_mul_scalar();
    check_grad_div();
    check_grad_pow();
    check_grad_matmul();
    std::cout << "All grad checks done.\n";
    return 0;
}
