# 🧠 VAVE-AI: AI-Powered Cost Reduction Platform for JSW-MGI

VAVE-AI is a cost optimization assistant designed to help JSW Morris Garages India streamline cost-saving initiatives. It combines GPT2, FAISS, and a rich SQLite3 idea database to deliver insights and generate automated reports (CSV, PPT, Audio).

---

## 📁 Project Structure

```
VAVE_AI-JSW-MGI/
├── app.py                    # Main Flask app
├── model/
│   ├── gpt2.pt               # TorchScript model
│   └── faiss_index.bin       # FAISS vector search index
├── cost_reduction.db         # SQLite database with idea records
├── static/
│   └── images/
│       └── mg.png            # MG India branding
├── templates/
│   └── index.html            # UI frontend
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container configuration
├── runtime.txt               # Python runtime spec
├── .dockerignore
└── AIML Dummy Ideas Data.xlsx # Source Excel file
```

---

## ⚙️ Requirements

- Python 3.10.11
- `pip install -r requirements.txt`
- Optional: CUDA-enabled GPU for acceleration
- Docker (for containerized deployment)

---

## 🚀 Run the Application

### ✅ 1. Run Locally (Development Mode)

```bash
pip install -r requirements.txt
python app.py
```

Then open your browser: http://localhost:5000

---

### 🐳 2. Run in Docker (Local Physical Servers)

```bash
# Build Docker image
docker build -t vave-ai-jsw .

# Run the container
docker run -d -p 5000:5000 vave-ai-jsw
```

Open: http://localhost:5000

---

### ☁️ 3. Push to Docker Hub

```bash
# Login to DockerHub
docker login

# Tag the image
docker tag vave-ai-jsw yourdockerhubusername/vave-ai-jsw

# Push it
docker push yourdockerhubusername/vave-ai-jsw
```

---

### 🌍 4. Deploy to Remote Server (AWS, GCP, and Cloud based remote servers)

SSH into your server:

```bash
# Install Docker
sudo apt update && sudo apt install docker.io -y

# Pull image from DockerHub
docker pull yourdockerhubusername/vave-ai-jsw

# Run it
docker run -d -p 5000:5000 yourdockerhubusername/vave-ai-jsw
```

Access from browser using: http://`<server-ip>`:5000

---

## 🔧 Manual Setup Notes

- Ensure these files exist before first run:
  - `model/gpt2.pt`
  - `model/faiss_index.bin`
  - `cost_reduction.db`
- The app will load and embed existing cost ideas and serve smart search + generative summaries.
- Flask endpoints include `/chat`, `/stats`, `/download_csv`, `/download_ppt`.

---

## ✨ Features

- 🔍 Natural language search across cost ideas
- 💡 GPT2-generated insights and summaries
- 📊 Auto-generated CSV and PowerPoint reports
- 🔈 Audio narration for every result
- 📈 Live statistics of idea status & savings
- 🌙 Dark UI theme with branding

---

## 📞 Support

For internal help, contact the JSW-MGI VAVE-AI team.

---

> _Built with 💡 by JSW-MG motor India for smarter cost engineering_
>
