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
import uuid 
import os   
import json 

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app) 

# Global variables to store real-time data
anomaly_buffer = []
buffer_lock = threading.Lock()
MAX_BUFFER_SIZE = 100

# Global variables for loaded assets (initialized below)
MODEL, SCALER, THRESHOLD = None, None, None
NORMAL_POOLS, ATTACK_POOLS = None, None
MASTER_COLUMNS = []
RUN_REAL_DETECTION = False

# === DEMO MODE FALLBACK FUNCTION ===
def generate_dummy_event():
    """Generates simple, predictable dummy data for dashboard testing."""
    value = random.uniform(0.1, 100.0)
    is_anomaly = value > 80
    
    if is_anomaly:
        severity = "High" if value > 95 else "Medium"
        reason = f"Simulated spike at {value:.2f}"
    else:
        severity = "Low" if value > 50 else "Medium"
        reason = "Normal simulation activity"

    timestamp_iso = datetime.now().isoformat() + "Z"

    return {
        'id': str(uuid.uuid4())[:8],
        'timestamp': timestamp_iso, 
        'value': float(value), 
        'anomaly_score': float(value / 100.0), 
        'is_anomaly': bool(is_anomaly),
        'true_label': "Simulated",
        'prediction': "Attack" if is_anomaly else "Normal",
        'severity': severity,
        'status': 'New' if is_anomaly else 'Closed',
        'reason': reason,
        'asset_id': f"DEMO_{random.randint(10, 99)}",
        'raw_data': json.dumps({"source": "DEMO_FEED", "value": value})
    }

# === LOAD PRODUCTION ASSETS (Now used for global initialization) ===
def load_production_assets():
    """Load trained model, scaler, and threshold"""
    global MODEL, SCALER, THRESHOLD
    try:
        MODEL = tf.keras.models.load_model('champion_autoencoder_model.keras')
        SCALER = joblib.load('nsl_kdd_scaler.joblib')
        with open('anomaly_threshold.txt', 'r') as f:
            THRESHOLD = float(f.read())
        print("✅ Production assets loaded successfully")
        return True
    except Exception as e:
        print(f"❌ Error loading assets (Running in DEMO MODE): {e}")
        MODEL, SCALER, THRESHOLD = None, None, None
        return False

# === GET MASTER COLUMNS (Now used for global initialization) ===
def get_master_columns():
    """Build master column list from training data"""
    global MASTER_COLUMNS
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
    
    try:
        df_train = pd.read_csv('KDDTrain+.txt', header=None, names=col_names_full, nrows=100) 
        df_train = df_train.drop(['label', 'difficulty'], axis=1)
        categorical_cols = ['protocol_type', 'service', 'flag']
        df_train_encoded = pd.get_dummies(df_train, columns=categorical_cols, dtype=int)
        MASTER_COLUMNS = df_train_encoded.columns
        return True
    except FileNotFoundError:
        print("❌ KDDTrain+.txt not found. Cannot determine master columns.")
        MASTER_COLUMNS = []
        return False

# === LOAD DATA POOLS (Now used for global initialization) ===
def load_data_pools():
    """Load normal and attack data pools using global scaler"""
    global NORMAL_POOLS, ATTACK_POOLS
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
    
    try:
        df_train = pd.read_csv('KDDTrain+.txt', header=None, names=col_names_full)
        df_test = pd.read_csv('KDDTest+.txt', header=None, names=col_names_full)
    except FileNotFoundError:
        print("❌ KDDTrain+.txt or KDDTest+.txt not found. Cannot load data pools.")
        NORMAL_POOLS, ATTACK_POOLS = (np.array([]), pd.DataFrame()), (np.array([]), pd.DataFrame())
        return False

    df_full = pd.concat([df_train, df_test], ignore_index=True)
    df_full['true_label'] = np.where(df_full['label'] == 'normal', 'Normal', 'Attack')
    df_full = df_full.drop(['label', 'difficulty'], axis=1)
    categorical_cols = ['protocol_type', 'service', 'flag']
    df_encoded = pd.get_dummies(df_full.drop('true_label', axis=1), columns=categorical_cols, dtype=int)
    
    df_reindexed = df_encoded.reindex(columns=MASTER_COLUMNS, fill_value=0)
    normal_rows = df_full[df_full['true_label'] == 'Normal']
    attack_rows = df_full[df_full['true_label'] == 'Attack']
    
    normal_logs_scaled = SCALER.transform(df_reindexed.loc[normal_rows.index])
    attack_logs_scaled = SCALER.transform(df_reindexed.loc[attack_rows.index])
    
    NORMAL_POOLS = (normal_logs_scaled, normal_rows)
    ATTACK_POOLS = (attack_logs_scaled, attack_rows)
    print(f"✅ Data pools created. Normal: {len(normal_logs_scaled)}, Attack: {len(attack_logs_scaled)}")
    return True

# === PREDICTION FUNCTIONS (Uses global variables) ===
def get_new_log_row(anomaly_chance=0.20):
    """Sample a random log from pools, using global pools"""
    if random.random() < anomaly_chance and len(ATTACK_POOLS[0]) > 0:
        pool_scaled, pool_display = ATTACK_POOLS
        true_label = "Attack"
    else:
        pool_scaled, pool_display = NORMAL_POOLS
        true_label = "Normal"
    
    if len(pool_scaled) == 0: 
        return None, None, "Error"
        
    idx = random.randint(0, len(pool_scaled) - 1)
    log_scaled = pool_scaled[idx]
    log_display = pool_display.iloc[idx]
    
    return log_scaled, log_display, true_label

def get_severity(score):
    """Calculate severity based on score, using global threshold"""
    if score < THRESHOLD:
        return "Low", "Normal"
    
    severity_factor = score / THRESHOLD
    if severity_factor > 5:
        return "High", "CRITICAL"
    elif severity_factor > 2:
        return "Medium", "High"
    else:
        return "Low", "Medium"

def predict_log(log_scaled, log_display, true_label):
    """Run prediction on a single log and format for frontend, using global model"""
    log_scaled = np.array([log_scaled])
    # The Keras predict call must happen inside the thread
    reconstruction = MODEL.predict(log_scaled, verbose=0)
    mse = np.mean(np.power(log_scaled - reconstruction, 2), axis=1)[0]
    
    is_anomaly = (mse > THRESHOLD)
    prediction = "Attack" if is_anomaly else "Normal"
    severity, priority = get_severity(mse)
    
    if is_anomaly:
        if mse > THRESHOLD * 5:
            reason = "Critical anomaly detected - Immediate investigation required"
        elif mse > THRESHOLD * 2:
            reason = "Suspicious activity pattern detected"
        else:
            reason = "Moderate deviation from normal behavior"
    else:
        reason = "Normal behavior pattern"
    
    timestamp_iso = datetime.now().isoformat() + "Z"

    return {
        'id': str(uuid.uuid4())[:8], 
        'timestamp': timestamp_iso, 
        'value': float(mse * 100),  
        'anomaly_score': float(mse),
        'is_anomaly': bool(is_anomaly),
        'true_label': true_label,
        'prediction': prediction,
        'severity': severity,
        'status': 'New' if is_anomaly else 'Closed',
        'reason': reason,
        'asset_id': f"{log_display['protocol_type']}-{log_display['service'][:10]}",
        'raw_data': json.dumps({"protocol": log_display["protocol_type"], "service": log_display["service"], "score": mse})
    }

# === BACKGROUND THREAD FOR CONTINUOUS DETECTION ===
def continuous_detection_worker():
    """Background thread that continuously generates predictions or demo data."""
    global anomaly_buffer, RUN_REAL_DETECTION
    
    print(f"Worker thread starting. Real Detection Mode: {RUN_REAL_DETECTION}")
    
    while True:
        try:
            if RUN_REAL_DETECTION:
                # Use real detection logic
                log_scaled, log_display, true_label = get_new_log_row(anomaly_chance=0.20)
                if log_scaled is None:
                    time.sleep(1)
                    continue
                result = predict_log(log_scaled, log_display, true_label)
            
            else:
                # Use DEMO MODE
                result = generate_dummy_event()
            
            # Add to buffer
            with buffer_lock:
                anomaly_buffer.append(result)
                if len(anomaly_buffer) > MAX_BUFFER_SIZE:
                    anomaly_buffer.pop(0)
            
            time.sleep(1) 
            
        except Exception as e:
            # If an error happens inside the thread, log it and keep the thread alive 
            # (it will revert to DEMO mode if the real assets were the cause)
            print(f"Error in detection worker: {e}")
            time.sleep(5)

# === FLASK ROUTES (Unchanged) ===
@app.route('/')
def index():
    """Serve the dashboard HTML"""
    return render_template('dashboard.html')

@app.route('/api/anomalies')
def get_anomalies():
    """API endpoint to get current anomalies, matching frontend's 'anomalies' and 'metrics' keys"""
    with buffer_lock:
        data = anomaly_buffer.copy()
    
    # Calculate metrics
    total_anomalies = sum(1 for d in data if d['is_anomaly'])
    max_score = max([d['anomaly_score'] for d in data], default=0.0)
    
    return jsonify({
        'anomalies': data, 
        'metrics': {        
            'total_anomalies': total_anomalies,
            'max_score': round(max_score, 4),
            'total_events': len(data)
        }
    })

# === GLOBAL INITIALIZATION BLOCK (Runs once when Gunicorn starts) ===
# 1. Load ML assets
if load_production_assets():
    # 2. If successful, load master columns
    if get_master_columns():
        # 3. If master columns successful, load data pools
        if load_data_pools():
            RUN_REAL_DETECTION = True
            print("🟢 Starting Real Detection Thread.")
        else:
             print("🟡 Starting Demo Thread (Data Pools Missing).")
    else:
        print("🟡 Starting Demo Thread (Master Columns Missing).")
else:
    print("🟡 Starting Demo Thread (ML Assets Missing).")


# 4. Start the background detection thread
detection_thread = threading.Thread(target=continuous_detection_worker, daemon=True)
detection_thread.start()
