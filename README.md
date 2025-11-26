# R.A.D.A.R | Real-time Anomaly Detection & Analysis Response

**Next-Gen SIEM (Security Information and Event Management) System**

R.A.D.A.R is a high-fidelity security dashboard that visualizes network traffic anomalies in real-time using a trained Autoencoder Neural Network. It combines a Flask backend with a futuristic, immersive frontend powered by Three.js and Plotly.

---

## 📋 Table of Contents

- [System Requirements](#-system-requirements)
- [Installation Guide](#%EF%B8%8F-installation-guide)
- [Database Setup](#%EF%B8%8F-database-setup)
- [Running the Application](#-running-the-application)
- [Web UI Component Walkthrough](#-web-ui-component-walkthrough)
- [Troubleshooting](#-troubleshooting)

---

## 💻 System Requirements

To run R.A.D.A.R locally, your system needs:

| Component | Requirement |
|-----------|-------------|
| **OS** | Windows 10/11, macOS, or Linux |
| **Python** | Version 3.11 or higher (Check via `python --version`) |
| **Database** | PostgreSQL 14 or higher |
| **RAM** | Minimum 4GB (8GB recommended for smooth 3D rendering) |
| **Browser** | Chrome, Edge, or Firefox (with WebGL enabled) |

---

## ⚙️ Installation Guide

Follow these steps to set up the environment from scratch.

### 1. Clone the Repository

Open your terminal (Command Prompt, PowerShell, or Bash) and run:
```bash
git clone <YOUR_REPO_URL_HERE>
cd radar_anomaly_detection
```

### 2. Create a Virtual Environment

It's best practice to keep dependencies isolated.

**Windows:**
```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

> **Note:** You should see `(venv)` at the start of your command line

### 3. Install Dependencies

Install all required Python libraries using the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## 🗄️ Database Setup

The application requires a PostgreSQL database to store anomaly logs. You must create this manually before running the app.

### 1. Install PostgreSQL

If you don't have it, download it from [postgresql.org](https://www.postgresql.org/).

### 2. Create Database & Table

Open your terminal and log in to the PostgreSQL shell:
```bash
psql -U postgres
```

> Enter your postgres password when prompted

Run the following SQL commands one by one:
```sql
-- 1. Create the database
CREATE DATABASE radar;

-- 2. Connect to it
\c radar;

-- 3. Create the anomalies table
CREATE TABLE anomalies (
    anomaly_id BIGINT PRIMARY KEY,
    timestamp TIMESTAMP,
    value FLOAT,
    anomaly_score FLOAT,
    is_anomaly BOOLEAN,
    true_label VARCHAR(50),
    prediction VARCHAR(50),
    severity VARCHAR(20),
    status VARCHAR(20),
    reason TEXT,
    asset_id VARCHAR(100),
    raw_data TEXT
);

-- 4. Exit
\q
```

### 3. Update Connection Config

1. Open the file `app.py` in your code editor
2. Locate the `PG_CONN` dictionary near the top:
```python
PG_CONN = {
    "host": "localhost",
    "database": "radar",
    "user": "postgres",
    "password": "YOUR_PASSWORD_HERE",  # <--- Change this to your PostgreSQL password
    "port": 5432
}
```

3. Update the `password` field with your actual PostgreSQL password

---

## 🚀 Running the Application

### 1. Activate the Environment

If not already active:

**Windows:**
```bash
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### 2. Start the Server
```bash
python app.py
```

### 3. Access the App

Open your web browser and navigate to:

👉 **http://127.0.0.1:5000**

---

## 🌐 Web UI Component Walkthrough

R.A.D.A.R is split into four distinct, interconnected modules.

### 1. The Landing Hub (`/`)

**Purpose:** The entry point. Sets the aesthetic tone and provides navigation.

**Key Features:**
- **3D Spline Robot:** A fully interactive 3D model that tracks your mouse cursor
- **Live Metrics:** The "Neural Telemetry" cards pull real-time summary stats from the backend even before you log in
- **Parallax Scroll:** Content sections glide over the fixed 3D background

---

### 2. The Dashboard (`/dashboard`)

**Purpose:** High-level operational monitoring.

**Key Features:**
- **KPI Row:** Displays "Lifetime Events" (Total DB count) vs. "View Events" (24H/7D filtered)
- **Main Signal Chart:** A neon-style area chart visualizing the Anomaly Score (MSE) over time
- **Dual HUD:** Floating boxes above the chart showing the raw JSON stream and the instant analysis result
- **Attack Surface Radar:** A spiderweb chart visualizing which protocols (HTTP, SSH, etc.) are being targeted most frequently
- **3D Scatter Plot:** A rotating 3D cube plotting anomalies by Time vs. Score vs. Value

---

### 3. Forensics Lab (`/forensics`)

**Purpose:** Investigation and response for specific threats.

**Key Features:**
- **Interactive Network Graph:** A physics-based topology map. When you select a threat, it highlights the compromised node in red
- **Visual Action:** 
  - Clicking `[ ISOLATE ]` physically cuts the link to the node on the graph
  - Clicking `[ BLOCK ]` turns the node into a locked box
- **Command Console:** A scrollable CLI that logs all your actions (simulating real terminal commands)

---

### 4. Neural Engine (`/neural`)

**Purpose:** Model configuration and system health.

**Key Features:**
- **3D Brain:** A Three.js visualization of the actual Autoencoder architecture (122 → 64 → 16 → 64 → 122). You can rotate, pan, and zoom into the layers
- **Data Packets:** Animated particles travel through the network to simulate data processing
- **Threshold Tuner:** A slider that lets you adjust the sensitivity of the AI model in real-time. Changes here immediately affect how the Dashboard classifies anomalies
- **System Vitals:** Real CPU and RAM usage charts pulled from your host machine via `psutil`

---

## 🔧 Troubleshooting

### "Database Connection Failed"
- Check if your PostgreSQL service is running
- Verify the password in `app.py` is correct
- Ensure the `radar` database exists

### Empty Charts
- Ensure `KDDTrain+.txt` and `KDDTest+.txt` are in the root folder
- The data generator needs these files to run properly

### 3D Visualizations Not Loading
- Ensure WebGL is enabled in your browser
- Try using Chrome or Firefox for best compatibility
- Check browser console for JavaScript errors

### Virtual Environment Issues
- Make sure you've activated the virtual environment before running commands
- Reinstall dependencies if needed: `pip install -r requirements.txt --force-reinstall`

---

## 📝 License

[Add your license information here]

## 👥 Contributors

[Add contributor information here]

## 📧 Contact

[Add contact information here]

---

**Built with ❤️ using Flask, Three.js, Plotly, and TensorFlow**
