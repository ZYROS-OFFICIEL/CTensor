#pragma once
#include <thread>
#include <string>
#include <cstdlib>
#include <iostream>


inline void start_dashboard_server(int port=8080 ) {
    std::cout << "Starting live dashboard on http://localhost:" << port << "\n";
    
    // Construct the command to run Python.
    // We use OS-specific commands to ensure it runs in the background without blocking C++
    #ifdef _WIN32
        // Windows command
        std::string cmd = "start /B python dashborad.py --port " + std::to_string(port);
    #else
        // Linux / macOS command (the '&' sends it to the background)
        std::string cmd = "python3 dashborad.py --port " + std::to_string(port) + " > dashboard_log.txt 2>&1 &";
    #endif
    
    int ret = std::system(cmd.c_str());
    (void)ret; // Suppress warning
}
// Asynchronous HTTP POST request so it doesn't block the training loop
inline void api_log_metrics(int epoch, size_t samples, double loss, double acc,int port =8080) {
    std::thread([=]() {
        // Construct the JSON payload string safely
        std::string json = "{\\\"epoch\\\": " + std::to_string(epoch) + 
                           ", \\\"samples\\\": " + std::to_string(samples) + 
                           ", \\\"loss\\\": " + std::to_string(loss) + 
                           ", \\\"acc\\\": " + std::to_string(acc) + "}";
        
        // Construct a curl POST command
        std::string cmd = "curl -s -X POST http://localhost:" + std::to_string(port) + "/update -H \"Content-Type: application/json\" -d \"" + json + "\"";
        
        // Execute quietly in the background
        int ret = std::system(cmd.c_str());
        (void)ret; // Suppress unused variable warning
    }).detach();
}

