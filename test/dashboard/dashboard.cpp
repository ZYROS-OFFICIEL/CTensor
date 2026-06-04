#include <iostream>
#include <thread>
#include <chrono>
#include "neuralnet.h"

void test_dashboard_logging() {
    log_scalar("Test/Loss", 1, 0.75, 8080);
    log_metrics(1, 128, 0.50, 85.5, 8080);
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
}

int main() {
    test_dashboard_logging();
    std::cout << "test_dashboard passed\n";
    return 0;
}