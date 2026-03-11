import http.server
import socketserver
import json
import os
import argparse
from collections import defaultdict

# Store metrics grouped by tag
# Format: {"Loss/Train": [{"step": 1, "value": 0.5}, ...], "Accuracy": [...]}
training_data = defaultdict(list)
# Keep track of the latest values to show in the UI sidebar/header
latest_stats = {}

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
            
            # Send both the arrays and the latest snapshots
            response_payload = {
                "metrics": training_data,
                "latest": latest_stats
            }
            self.wfile.write(json.dumps(response_payload).encode('utf-8'))
            
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
                    
                    tag = payload.get('tag', 'Metric')
                    step = payload.get('step', 0)
                    value = payload.get('value', 0.0)

                    # Append to specific tag history
                    training_data[tag].append({"step": step, "value": value})
                    latest_stats[tag] = value
                    
                    print(f"Logged -> [{tag}] Step: {step}, Value: {value:.4f}")
                    
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
    parser = argparse.ArgumentParser(description="TensorBoard-Lite Server")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to run the server on (default: 8080)")
    args = parser.parse_args()
    
    PORT = args.port

    if not os.path.exists('dashboard.html'):
        print("Warning: dashboard.html not found in the current directory!")
        
    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        print(f"🚀 TensorBoard-Lite API listening at http://localhost:{PORT}")
        print("Waiting for C++ scalar data...")
        httpd.serve_forever()