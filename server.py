from flask import Flask, request, jsonify, render_template_string
import subprocess
import pandas as pd
import os
import sys

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)
RESULTS_FILE = os.path.join(BASE_DIR, "evaluation_results.csv")
SIM_SCRIPT = os.path.join(BASE_DIR, "Serverless_Aggregator_Simulation.py")

@app.route('/')
def home():
    return render_template_string('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Federated Learning Demo</title>
            <script>
                async function callAggregate() {
                    document.getElementById("output").innerText = "Running Aggregation... please wait ⏳";
                    const response = await fetch('/aggregate', {method: 'POST'});
                    const result = await response.json();
                    document.getElementById("output").innerText = result.message;
                }

                async function callAccuracy() {
                    const response = await fetch('/accuracy');
                    const result = await response.json();

                    if (result.Centralized && result["FL+DP+SecAgg"]) {
                        document.getElementById("output").innerHTML = `
                            <table border="1" cellpadding="8">
                                <tr><th>Setup</th><th>Round</th><th>Accuracy</th></tr>
                                <tr><td>Centralized</td><td>${result.Centralized.round}</td><td>${result.Centralized.accuracy.toFixed(2)}%</td></tr>
                                <tr><td>FL+DP+SecAgg</td><td>${result["FL+DP+SecAgg"].round}</td><td>${result["FL+DP+SecAgg"].accuracy.toFixed(2)}%</td></tr>
                            </table>
                        `;
                    } else {
                        document.getElementById("output").innerText = JSON.stringify(result, null, 2);
                    }
                }
            </script>
        </head>
        <body style="font-family:Arial; margin:40px;">
            <h2>Privacy-Preserving Federated Learning</h2>
            <button onclick="callAggregate()">Run Aggregation</button>
            <button onclick="callAccuracy()">Get Accuracy</button>
            <pre id="output" style="margin-top:20px; background:#f0f0f0; padding:10px;"></pre>
        </body>
        </html>
    ''')

@app.route('/aggregate', methods=['POST'])
def aggregate():
    try:
        subprocess.run([sys.executable, SIM_SCRIPT], check=True, cwd=BASE_DIR)
        return jsonify({"message": "✅ Training complete! Now click 'Get Accuracy'."})
    except Exception as e:
        return jsonify({"message": f"❌ Error during aggregation: {str(e)}"})

@app.route('/accuracy', methods=['GET'])
def accuracy():
    try:
        if os.path.exists(RESULTS_FILE):
            df = pd.read_csv(RESULTS_FILE)
            if df.empty:
                return jsonify({"message": "⚠️ Results file is empty."})

            latest = {}
            for setup in df["setup"].unique():
                last_row = df[df["setup"] == setup].iloc[-1].to_dict()
                latest[setup] = {
                    "round": int(last_row["round"]),
                    "accuracy": float(last_row["accuracy"])
                }

            return jsonify(latest)
        else:
            return jsonify({"message": "⚠️ No results found. Please run aggregation first."})
    except Exception as e:
        return jsonify({"message": f"❌ Error reading results: {str(e)}"})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
