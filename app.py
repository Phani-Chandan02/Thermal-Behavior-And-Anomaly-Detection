from flask import Flask, jsonify, request, send_from_directory, render_template
import os
import json
import csv
import subprocess
import threading

app = Flask(__name__, static_folder='.', template_folder='templates')

CONFIG_FILE = 'config.json'
LOG_FILES = ['output/logs/events_log.csv', 'output/results/custom_events_log.csv']
SNAPSHOT_DIR = 'output/results/snapshots'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def config():
    if request.method == 'GET':
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return jsonify(json.load(f))
        return jsonify({})
    else:
        data = request.json
        with open(CONFIG_FILE, 'w') as f:
            json.dump(data, f, indent=4)
        return jsonify({"status": "success"})

@app.route('/api/logs', methods=['GET'])
def get_logs():
    logs = []
    for log_file in LOG_FILES:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['source'] = os.path.basename(log_file)
                    logs.append(row)
    # Return last 100 logs, reversed (newest first)
    return jsonify(logs[-100:][::-1])

@app.route('/api/snapshots', methods=['GET'])
def get_snapshots():
    if not os.path.exists(SNAPSHOT_DIR):
        return jsonify([])
    files = [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.jpg')]
    # Sort by creation time, newest first
    files.sort(key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x)), reverse=True)
    return jsonify([f"/{SNAPSHOT_DIR}/{f}" for f in files])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    stats = {"Intrusion": 0, "Aggressive Clustering": 0, "Running": 0, "Loitering": 0, "Normal": 0}
    for log_file in LOG_FILES:
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    behavior = row.get('Behavior', 'Normal')
                    if behavior in stats:
                        stats[behavior] += 1
    return jsonify(stats)

@app.route('/api/run', methods=['POST'])
def run_script():
    script_name = request.json.get('script')
    script_mapping = {
        'thermal_behavior_video.py': 'src/tracking/thermal_behavior_video.py',
        'evaluate_custom_dataset.py': 'src/tracking/evaluate_custom_dataset.py',
        'best_onnx.py': 'src/core/best_onnx.py'
    }
    if script_name not in script_mapping:
        return jsonify({"status": "error", "message": "Invalid script"}), 400
    
    def run():
        # Use the virtual environment python
        subprocess.run(['venv/Scripts/python.exe', script_mapping[script_name]])
    
    thread = threading.Thread(target=run)
    thread.start()
    return jsonify({"status": "started", "script": script_name})

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
