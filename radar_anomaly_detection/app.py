from flask import Flask, render_template, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import random
import threading
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global variables to store real-time data
anomaly_buffer = []
buffer_lock = threading.Lock()
MAX_BUFFER_SIZE = 100

# === LOAD PRODUCTION ASSETS ===
def load_production_assets():
    """Load trained model, scaler, and threshold"""
    try:
        model = tf.keras.models.load_model('champion_autoencoder_model.keras')
        scaler = joblib.load('nsl_kdd_scaler.joblib')
        with open('anomaly_threshold.txt', 'r') as f:
            threshold = float(f.read())
        print("✅ Production assets loaded successfully")
        return model, scaler, threshold
    except Exception as e:
        print(f"❌ Error loading assets: {e}")
        return None, None, None

# === GET MASTER COLUMNS ===
def get_master_columns():
    """Build master column list from training data"""
    col_names_full = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    df_train = pd.read_csv('KDDTrain+.txt', header=None, names=col_names_full)
    df_train = df_train.drop(['label', 'difficulty'], axis=1)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, dtype=int)
    return df_train_encoded.columns

# === LOAD DATA POOLS ===
def load_data_pools(master_columns, scaler):
    """Load normal and attack data pools"""
    col_names_full = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 
        'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
        'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 
        'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    df_train = pd.read_csv('KDDTrain+.txt', header=None, names=col_names_full)
    df_test = pd.read_csv('KDDTest+.txt', header=None, names=col_names_full)
    df_full = pd.concat([df_train, df_test], ignore_index=True)
    
    df_full['true_label'] = np.where(df_full['label'] == 'normal', 'Normal', 'Attack')
    df_full = df_full.drop(['label', 'difficulty'], axis=1)
    
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_encoded = pd.get_dummies(df_full.drop('true_label', axis=1), 
                                 columns=categorical_cols, dtype=int)
    df_reindexed = df_encoded.reindex(columns=master_columns, fill_value=0)
    
    normal_indices = df_full[df_full['true_label'] == 'Normal'].index
    attack_indices = df_full[df_full['true_label'] == 'Attack'].index
    
    normal_logs_scaled = scaler.transform(df_reindexed.loc[normal_indices])
    attack_logs_scaled = scaler.transform(df_reindexed.loc[attack_indices])
    normal_logs_display = df_full.loc[normal_indices]
    attack_logs_display = df_full.loc[attack_indices]
    
    print(f"✅ Data pools created. Normal: {len(normal_logs_scaled)}, Attack: {len(attack_logs_scaled)}")
    return (normal_logs_scaled, normal_logs_display), (attack_logs_scaled, attack_logs_display)

# === PREDICTION FUNCTIONS ===
def get_new_log_row(normal_pools, attack_pools, anomaly_chance=0.20):
    """Sample a random log from pools"""
    if random.random() < anomaly_chance:
        pool_scaled, pool_display = attack_pools
        true_label = "Attack"
    else:
        pool_scaled, pool_display = normal_pools
        true_label = "Normal"
    
    idx = random.randint(0, len(pool_scaled) - 1)
    log_scaled = pool_scaled[idx]
    log_display = pool_display.iloc[idx]
    
    return log_scaled, log_display, true_label

def get_severity(score, threshold):
    """Calculate severity based on score"""
    if score < threshold:
        return "Low", "Normal"
    
    severity_factor = score / threshold
    if severity_factor > 5:
        return "High", "CRITICAL"
    elif severity_factor > 2:
        return "Medium", "High"
    else:
        return "Medium", "Medium"

def predict_log(log_scaled, log_display, true_label, model, threshold):
    """Run prediction on a single log"""
    log_scaled = np.array([log_scaled])
    reconstruction = model.predict(log_scaled, verbose=0)
    mse = np.mean(np.power(log_scaled - reconstruction, 2), axis=1)[0]
    
    is_anomaly = (mse > threshold)
    prediction = "Attack" if is_anomaly else "Normal"
    severity, priority = get_severity(mse, threshold)
    
    # Generate reason based on score
    if is_anomaly:
        if mse > threshold * 5:
            reason = "Critical anomaly detected - Immediate investigation required"
        elif mse > threshold * 2:
            reason = "Suspicious activity pattern detected"
        else:
            reason = "Moderate deviation from normal behavior"
    else:
        reason = "Normal behavior pattern"
    
    return {
        'id': random.randint(1000, 9999),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'value': float(mse * 100),  # Scale for visualization
        'anomaly_score': float(mse),
        'is_anomaly': bool(is_anomaly),
        'true_label': true_label,
        'prediction': prediction,
        'severity': severity,
        'status': 'New' if is_anomaly else 'Closed',
        'reason': reason,
        'asset_id': f"{log_display['protocol_type']}-{log_display['service'][:10]}",
        'raw_data': f'{{"protocol": "{log_display["protocol_type"]}", "service": "{log_display["service"]}", "score": {mse:.4f}}}'
    }

# === BACKGROUND THREAD FOR CONTINUOUS DETECTION ===
def continuous_detection_worker():
    """Background thread that continuously generates predictions"""
    global anomaly_buffer
    
    model, scaler, threshold = load_production_assets()
    if model is None:
        print("❌ Cannot start detection - assets not loaded")
        return
    
    master_columns = get_master_columns()
    normal_pools, attack_pools = load_data_pools(master_columns, scaler)
    
    print("🚀 Starting continuous anomaly detection...")
    
    while True:
        try:
            # Generate prediction
            log_scaled, log_display, true_label = get_new_log_row(
                normal_pools, attack_pools, anomaly_chance=0.20
            )
            result = predict_log(log_scaled, log_display, true_label, model, threshold)
            
            # Add to buffer
            with buffer_lock:
                anomaly_buffer.append(result)
                if len(anomaly_buffer) > MAX_BUFFER_SIZE:
                    anomaly_buffer.pop(0)
            
            time.sleep(1)  # Generate one prediction per second
            
        except Exception as e:
            print(f"Error in detection worker: {e}")
            time.sleep(5)

# === FLASK ROUTES ===
@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/anomalies')
def get_anomalies():
    """API endpoint to get current anomalies"""
    with buffer_lock:
        data = anomaly_buffer.copy()
    
    # Calculate metrics
    total_anomalies = sum(1 for d in data if d['is_anomaly'])
    max_score = max([d['anomaly_score'] for d in data], default=0)
    
    return jsonify({
        'anomalies': data,
        'metrics': {
            'total_anomalies': total_anomalies,
            'max_score': round(max_score, 4),
            'total_events': len(data)
        }
    })

@app.route('/api/latest')
def get_latest():
    """Get only the latest anomaly"""
    with buffer_lock:
        if anomaly_buffer:
            return jsonify(anomaly_buffer[-1])
        return jsonify({})

if __name__ == '__main__':
    # Start background detection thread
    detection_thread = threading.Thread(target=continuous_detection_worker, daemon=True)
    detection_thread.start()
    
    # Start Flask server
    print("\n" + "="*60)
    print("🚀 RADAR Anomaly Detection System Starting...")
    print("="*60)
    print("📊 Dashboard: http://localhost:5000")
    print("🔌 API Endpoint: http://localhost:5000/api/anomalies")
    print("="*60 + "\n")
    
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)