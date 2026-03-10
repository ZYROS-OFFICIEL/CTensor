import http.server
import socketserver
import json
import os
import argparse

# In-memory storage for our training metrics
training_data = []

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        # Serve the HTML dashboard
        if self.path == '/' or self.path == '/index.html':
            self.path = '/dashboard.html'
            return super().do_GET()
        
        # API Endpoint to fetch data for the graphs
        elif self.path == '/api/data':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.end_headers()
            self.wfile.write(json.dumps(training_data).encode('utf-8'))
            
        else:
            return super().do_GET()

    def do_POST(self):
        # API Endpoint for C++ to send new metrics
        if self.path == '/update':
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                try:
                    # Parse the JSON from C++
                    payload = json.loads(post_data.decode('utf-8'))
                    training_data.append(payload)
                    
                    print(f"Received -> Epoch: {payload['epoch']}, Loss: {payload['loss']:.4f}, Acc: {payload['acc']:.2f}%")
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success"}).encode('utf-8'))
                except Exception as e:
                    print("Error parsing data:", e)
                    self.send_response(400)
                    self.end_headers()
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    # Set up argument parsing for the port
    parser = argparse.ArgumentParser(description="Live Training Dashboard Server")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    args = parser.parse_args()
    
    PORT = args.port

    # Ensure dashboard.html exists in the same directory
    if not os.path.exists('dashboard.html'):
        print("Warning: dashboard.html not found in the current directory!")
        
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"🚀 Training Dashboard API listening at http://localhost:{PORT}")
        print("Waiting for C++ data...")
        httpd.serve_forever()