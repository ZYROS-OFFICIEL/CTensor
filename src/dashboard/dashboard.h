#pragma once
#include <thread>
#include <string>
#include <cstdlib>
#include <iostream>

inline void start_dashboard_server(int port=8080) {
    std::cout << "Starting TensorBoard-lite server on http://localhost:" << port << "\n";
    
    // Construct the command to run Python.
    #ifdef _WIN32
        std::string cmd = "start /B python dashboard.py --port " + std::to_string(port);
    #else
        std::string cmd = "python3 dashboard.py --port " + std::to_string(port) + " > dashboard_log.txt 2>&1 &";
    #endif
    
    int ret = std::system(cmd.c_str());
    (void)ret; // Suppress warning
}


inline void log_scalar(const std::string& tag, int step, double value, int port = 8080) {
    std::thread([=]() {
        // Construct the JSON payload for a single scalar
        std::string json = "{\\\"tag\\\": \\\"" + tag + "\\\", " + 
                           "\\\"step\\\": " + std::to_string(step) + ", " + 
                           "\\\"value\\\": " + std::to_string(value) + "}";
        
        // Construct a curl POST command
        std::string cmd = "curl -s -X POST http://localhost:" + std::to_string(port) + "/update -H \"Content-Type: application/json\" -d \"" + json + "\"";
        
        // Execute quietly in the background
        int ret = std::system(cmd.c_str());
        (void)ret; // Suppress unused variable warning
    }).detach();
}

inline void log_metrics(int epoch, size_t samples, double loss, double acc, int port = 8080) {
    // Under the hood, we now split this into two separate scalar calls
    log_scalar("Loss/Train", epoch, loss, port);
    log_scalar("Accuracy/Train", epoch, acc, port);
}