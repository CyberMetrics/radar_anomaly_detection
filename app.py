from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
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
# POSTGRESQL CONFIG
# ==========================================
PG_CONN = {
    "host": "localhost",
    "database": "radar",
    "user": "postgres",
    "password": "123",    # <<< CHANGE THIS
    "port": 5432
}

def get_pg_connection():
    return psycopg2.connect(
        host=PG_CONN["host"],
        database=PG_CONN["database"],
        user=PG_CONN["user"],
        password=PG_CONN["password"],
        port=PG_CONN["port"]
    )

# ==========================================
# MEMORY BUFFER
# ==========================================
anomaly_buffer = []
buffer_lock = threading.Lock()
MAX_BUFFER_SIZE = 100


# ==========================================
# INSERT INTO POSTGRES
# ==========================================
def insert_anomaly(entry):
    conn = get_pg_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO anomalies 
        (anomaly_id, timestamp, value, anomaly_score, is_anomaly,
         true_label, prediction, severity, status, reason, asset_id, raw_data)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        entry["id"], entry["timestamp"], entry["value"], entry["anomaly_score"],
        entry["is_anomaly"], entry["true_label"], entry["prediction"],
        entry["severity"], entry["status"], entry["reason"],
        entry["asset_id"], entry["raw_data"]
    ))

    conn.commit()
    cur.close()
    conn.close()


# ==========================================
# QUERY POSTGRES
# ==========================================
def query_anomalies(start=None, end=None, limit=2000):
    conn = get_pg_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    q = "SELECT * FROM anomalies"
    params = []

    if start and end:
        q += " WHERE timestamp BETWEEN %s AND %s"
        params += [start, end]

    q += " ORDER BY timestamp ASC"

    if limit:
        q += " LIMIT %s"
        params.append(limit)

    cur.execute(q, params)
    rows = cur.fetchall()

    for r in rows:
        if isinstance(r["timestamp"], datetime):
            r["timestamp"] = r["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")

    cur.close()
    conn.close()
    return rows


# ==========================================
# MODEL LOADING & POOLS
# ==========================================
def load_assets():
    try:
        model = tf.keras.models.load_model("champion_autoencoder_model.keras")
    except:
        model = None

    try:
        scaler = joblib.load("nsl_kdd_scaler.joblib")
    except:
        scaler = None

    try:
        with open("anomaly_threshold.txt") as f:
            threshold = float(f.read().strip())
    except:
        threshold = 0.01

    return model, scaler, threshold


def get_master_columns():
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label','difficulty'
    ]
    df = pd.read_csv("KDDTrain+.txt", header=None, names=cols)
    df = df.drop(['label','difficulty'], axis=1)
    df = pd.get_dummies(df, columns=['protocol_type','service','flag'])
    return df.columns


def load_pools(master, scaler):
    cols = [
        'duration','protocol_type','service','flag','src_bytes','dst_bytes','land',
        'wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised',
        'root_shell','su_attempted','num_root','num_file_creations','num_shells',
        'num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count',
        'srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate',
        'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count',
        'dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
        'dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate',
        'dst_host_srv_rerror_rate','label','difficulty'
    ]

    train = pd.read_csv("KDDTrain+.txt", header=None, names=cols)
    test = pd.read_csv("KDDTest+.txt", header=None, names=cols)

    full = pd.concat([train,test], ignore_index=True)
    full["true_label"] = full["label"].apply(lambda x: "Normal" if x=="normal" else "Attack")
    full = full.drop(['label','difficulty'], axis=1)

    df = pd.get_dummies(full.drop('true_label', axis=1))
    df = df.reindex(columns=master, fill_value=0)

    normal_idxs = full[full["true_label"]=="Normal"].index
    attack_idxs = full[full["true_label"]=="Attack"].index

    normal_scaled = scaler.transform(df.loc[normal_idxs])
    attack_scaled = scaler.transform(df.loc[attack_idxs])

    normal_disp = full.loc[normal_idxs].reset_index(drop=True)
    attack_disp = full.loc[attack_idxs].reset_index(drop=True)

    return (normal_scaled, normal_disp), (attack_scaled, attack_disp)


# ==========================================
# PREDICTION MODEL
# ==========================================
def predict_event(log, disp, true, model, threshold):
    try:
        recon = model.predict(log.reshape(1,-1), verbose=0)
        mse = float(np.mean((log - recon)**2))
    except:
        mse = float(np.mean(np.abs(log)) * 1e-3)

    sev = "Low"
    if mse > threshold * 5: sev = "High"
    elif mse > threshold * 2: sev = "Medium"

    is_anom = mse > threshold

    return {
        "id": random.randint(100000,999999),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
        "value": mse*100,
        "anomaly_score": mse,
        "is_anomaly": is_anom,
        "true_label": true,
        "prediction": "Attack" if is_anom else "Normal",
        "severity": sev,
        "status": "New" if is_anom else "Closed",
        "reason": "Suspicious activity detected" if is_anom else "Normal",
        "asset_id": f"{disp['protocol_type']}-{str(disp['service'])[:8]}",
        "raw_data": f'{{"protocol":"{disp["protocol_type"]}","service":"{disp["service"]}","score":{mse:.6f}}}'
    }


# ==========================================
# BACKGROUND THREAD
# ==========================================
def detection_worker():
    model, scaler, threshold = load_assets()
    master = get_master_columns()
    normal_pool, attack_pool = load_pools(master, scaler)

    while True:
        scaled, disp, true = (
            *(attack_pool if random.random() < 0.2 else normal_pool),
            "Attack" if random.random() < 0.2 else "Normal"
        )

        idx = random.randint(0, len(scaled)-1)
        entry = predict_event(scaled[idx], disp.iloc[idx], true, model, threshold)

        with buffer_lock:
            anomaly_buffer.append(entry)
            if len(anomaly_buffer) > MAX_BUFFER_SIZE:
                anomaly_buffer.pop(0)

        insert_anomaly(entry)
        time.sleep(1)


# ==========================================
# API ROUTES
# ==========================================
@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/anomalies")
def api_live():
    with buffer_lock:
        data = anomaly_buffer.copy()

    return jsonify({
        "anomalies": data,
        "error": None
    })


@app.route("/api/latest")
def latest():
    with buffer_lock:
        return jsonify(anomaly_buffer[-1] if anomaly_buffer else {})


@app.route("/api/anomalies/history")
def api_history():
    rng = request.args.get("range")
    start = request.args.get("start")
    end = request.args.get("end")
    limit = int(request.args.get("limit", 2500))

    if rng:
        now = datetime.utcnow()
        if rng == "today":
            start = datetime(now.year,now.month,now.day).strftime("%Y-%m-%dT%H:%M:%S")
            end = now.strftime("%Y-%m-%dT%H:%M:%S")
        elif rng == "yesterday":
            today = datetime(now.year,now.month,now.day)
            start = (today - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
            end = (today - timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S")
        elif rng == "2days":
            end = now.strftime("%Y-%m-%dT%H:%M:%S")
            start = (now - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S")

    items = query_anomalies(start, end, limit)
    return jsonify({"anomalies": items, "error": None})


@app.route("/api/anomalies/<int:aid>")
def api_single(aid):
    conn = get_pg_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM anomalies WHERE anomaly_id=%s LIMIT 1", (aid,))
    row = cur.fetchone()
    cur.close()
    conn.close()

    if row and isinstance(row["timestamp"], datetime):
        row["timestamp"] = row["timestamp"].strftime("%Y-%m-%dT%H:%M:%S")

    return jsonify(row if row else {})


# ==========================================
# START SERVER
# ==========================================
if __name__ == "__main__":
    threading.Thread(target=detection_worker, daemon=True).start()
    print("🚀 RADAR running at http://localhost:5000")
    app.run(host="0.0.0.0", port=5000)

