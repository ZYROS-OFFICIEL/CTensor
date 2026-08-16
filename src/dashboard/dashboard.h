#pragma once
#include <thread>
#include <string>
#include <cstdlib>
#include <iostream>

void start_dashboard_server(int port=8080) ;

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

void log_metrics(int epoch, size_t samples, double loss, double acc, int port = 8080) ;