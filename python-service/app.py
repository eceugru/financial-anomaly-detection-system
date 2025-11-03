from flask import Flask, request, jsonify
from sklearn.ensemble import IsolationForest
import numpy as np

app = Flask(__name__)

# Basit sahte model (örnek)
model = IsolationForest(n_estimators=100, contamination=0.01)
model.fit([[0]*5, [1]*5, [2]*5])  # örnek eğitim

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)
    score = -model.decision_function(features)[0]
    return jsonify({
        "is_anomaly": score > 0.8,
        "score": round(score, 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
