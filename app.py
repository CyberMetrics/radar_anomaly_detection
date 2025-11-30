from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import psutil
import tensorflow as tf
import random
import threading
import time
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import RealDictCursor
import warnings

warnings.filterwarnings("ignore")

app = Flask(__name__, template_folder='templates')
CORS(app)

# ==========================================
# CONFIG
# ==========================================
PG_CONN = {
    "host": "localhost",
    "database": "radar",
    "user": "postgres",
    "password": "123", 
    "port": 5432
}

# GLOBAL STATE
TOTAL_EVENTS = 0
MAX_SCORE = 0.0
CURRENT_THRESHOLD = 0.01 # Default start value
anomaly_buffer = []
buffer_lock = threading.Lock()
MAX_BUFFER_SIZE = 100

def get_db():
    try: return psycopg2.connect(**PG_CONN)
    except: return None

# ==========================================
# AI ENGINE
# ==========================================
def load_assets():
    global CURRENT_THRESHOLD
    print("[SYSTEM] Loading AI Assets...")
    try:
        model = tf.keras.models.load_model("champion_autoencoder_model.keras")
        scaler = joblib.load("nsl_kdd_scaler.joblib")
        
        if hasattr(scaler, 'feature_names_in_'):
            master_cols = list(scaler.feature_names_in_)
        else:
            # Fallback columns (Safety)
            master_cols = ['duration','src_bytes','dst_bytes','flag_SF'] 
            
        # Load initial threshold from file, then use memory
        try:
            with open("anomaly_threshold.txt") as f: 
                CURRENT_THRESHOLD = float(f.read().strip())
        except: 
            CURRENT_THRESHOLD = 0.0025

        return model, scaler, master_cols
    except Exception as e:
        print(f"[FATAL] Asset Load Error: {e}")
        return None, None, None

def load_data_pools(master_cols, scaler):
    print("[SYSTEM] Processing Data Pools...")
    # (Simplified column list for brevity - ensures code runs)
    cols = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land', 'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised', 'root_shell','su_attempted','num_root','num_file_creations','num_shells', 'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count', 'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate', 'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count', 'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate', 'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate', 'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate', 'dst_host_srv_rerror_rate','label','difficulty']

    try:
        df = pd.read_csv("KDDTrain+.txt", header=None, names=cols, nrows=5000)
        df['is_attack'] = df['label'] != 'normal'
        df_encoded = pd.get_dummies(df.drop(['label', 'difficulty', 'is_attack'], axis=1))
        df_aligned = df_encoded.reindex(columns=master_cols, fill_value=0)
        df_scaled = scaler.transform(df_aligned)
        
        attacks = df[df['is_attack'] == True].index
        normals = df[df['is_attack'] == False].index
        
        return (df_scaled[normals], df.iloc[normals].reset_index(drop=True)), \
               (df_scaled[attacks], df.iloc[attacks].reset_index(drop=True))
    except: return None, None

# ==========================================
# WORKER
# ==========================================
def worker():
    global MAX_SCORE, TOTAL_EVENTS
    model, scaler, master_cols = load_assets()
    if not model: return
    normal_pool, attack_pool = load_data_pools(master_cols, scaler)
    
    print("🚀 LIVE STREAM ACTIVE")

    while True:
        try:
            # Dynamic Threshold Check
            thresh = CURRENT_THRESHOLD

            is_attack = random.random() < 0.2
            pool_vecs, pool_meta = attack_pool if is_attack else normal_pool
            idx = random.randint(0, len(pool_vecs)-1)
            
            vector = pool_vecs[idx]
            meta = pool_meta.iloc[idx]

            recon = model.predict(vector.reshape(1,-1), verbose=0)
            mse = float(np.mean((vector - recon)**2))
            
            # REAL LOGIC: Compare against dynamic threshold
            is_anom = mse > thresh
            
            sev = "Low"
            if mse > thresh * 5: sev = "High"
            elif mse > thresh * 2: sev = "Medium"
            if not is_anom: sev = "Safe"

            raw_json = f'{{"protocol":"{meta["protocol_type"]}", "service":"{meta["service"]}", "bytes":{meta["src_bytes"]}}}'

            record = {
                "id": random.randint(100000, 999999),
                "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                "value": mse * 100, 
                "anomaly_score": mse,
                "is_anomaly": is_anom,
                "severity": sev,
                "asset_id": f"{str(meta['service']).upper()}_NODE_{random.randint(1,9)}",
                "raw_data": raw_json,
                "reason": "Deviation > Threshold" if is_anom else "Normal"
            }

            conn = get_db()
            if conn:
                cur = conn.cursor()
                cur.execute("""
                    INSERT INTO anomalies (anomaly_id, timestamp, value, anomaly_score, is_anomaly, severity, asset_id, raw_data, reason)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (record['id'], record['timestamp'], record['value'], record['anomaly_score'], record['is_anomaly'], record['severity'], record['asset_id'], record['raw_data'], record['reason']))
                conn.commit()
                conn.close()

            with buffer_lock:
                anomaly_buffer.append(record)
                if len(anomaly_buffer) > MAX_BUFFER_SIZE: anomaly_buffer.pop(0)
                TOTAL_EVENTS += 1
                if mse > MAX_SCORE: MAX_SCORE = mse

            time.sleep(1) 

        except Exception as e:
            print(f"[WORKER ERROR] {e}")
            time.sleep(5)

# ==========================================
# ROUTES
# ==========================================
@app.route("/")
def index(): return render_template("index.html")
@app.route("/dashboard")
def dashboard(): return render_template("dashboard.html")
@app.route("/forensics")
def forensics(): return render_template("forensics.html")
@app.route("/neural")
def neural(): return render_template("neural.html")

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route("/api/vitals")
def vitals():
    # Get Real CPU & RAM
    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    
    return jsonify({
        "cpu": cpu,
        "ram_percent": ram.percent,
        "ram_used_gb": round(ram.used / (1024**3), 2), # Convert bytes to GB
        "ram_total_gb": round(ram.total / (1024**3), 2)
    })

# --- NEW: THRESHOLD CONTROL ---
@app.route("/api/settings/threshold", methods=["GET", "POST"])
def manage_threshold():
    global CURRENT_THRESHOLD
    if request.method == "POST":
        data = request.json
        new_val = float(data.get("threshold", 0.0025))
        CURRENT_THRESHOLD = new_val
        print(f"[SYSTEM] Threshold updated to {CURRENT_THRESHOLD}")
        return jsonify({"status": "updated", "value": CURRENT_THRESHOLD})
    return jsonify({"threshold": CURRENT_THRESHOLD})

@app.route("/api/stats")
def stats():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM anomalies")
        count = cur.fetchone()[0]
        conn.close()
    except: count = 0
    with buffer_lock: return jsonify({"db_total": count, "session_max": MAX_SCORE})

@app.route("/api/latest")
def latest():
    with buffer_lock: return jsonify(anomaly_buffer[-1] if anomaly_buffer else {})

@app.route("/api/anomalies")
def live():
    with buffer_lock: return jsonify({"anomalies": anomaly_buffer})

@app.route("/api/anomalies/history")
def history():
    rng = request.args.get("range")
    sample = int(request.args.get("sample", 0))
    end = datetime.utcnow()
    start = end - timedelta(hours=24)
    if rng == "week": start = end - timedelta(days=7)

    conn = get_db()
    if not conn: return jsonify({"anomalies": [], "stats": {}})
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM anomalies WHERE timestamp BETWEEN %s AND %s ORDER BY timestamp ASC", (start, end))
    rows = cur.fetchall()
    conn.close()

    view_total = len(rows)
    view_anoms = sum(1 for r in rows if r['is_anomaly'])
    view_max = max([r['anomaly_score'] for r in rows], default=0)

    if sample > 0 and len(rows) > sample:
        indices = sorted(random.sample(range(len(rows)), sample))
        rows = [rows[i] for i in indices]
    for r in rows:
        if isinstance(r['timestamp'], datetime): r['timestamp'] = r['timestamp'].strftime("%Y-%m-%dT%H:%M:%S")

    return jsonify({"anomalies": rows, "view_stats": {"total": view_total, "anomalies": view_anoms, "max_score": view_max}})

if __name__ == "__main__":
    threading.Thread(target=worker, daemon=True).start()
    app.run(host="0.0.0.0", port=5000)