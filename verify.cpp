#include <iostream>
#include <cmath>
#include "tensor1.h"
#include "ops1.h"
#include "autograd_core.h"

static bool close(double a, double b, double tol=1e-5) {
    return std::fabs(a - b) < tol * std::max({1.0, std::fabs(a), std::fabs(b)});
}

double numerical_grad(std::function<double(double)> f, double x, double eps=1e-6) {
    return (f(x + eps) - f(x - eps)) / (2 * eps);
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
    check_grad_add();
    check_grad_mul();
    check_grad_pow();
    check_grad_matmul();
    std::cout << "All grad checks done.\n";
    return 0;
}
