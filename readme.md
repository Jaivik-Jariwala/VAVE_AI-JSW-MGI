# 🧠 VAVE-AI: Advanced Cost Optimization & Engineering Assistant

> 🚨 **STRICTLY CONFIDENTIAL & PROPRIETARY** 🚨
>
> **THIS IS A COMPANY'S OFFICE TRADE MARK SECRET TOOL AND KIT AND IN PROCESS FOR PATENT AND FOLLOWING TO KEEP THIS AS INTERNAL COMPANY USE ONLY.**
>
> **Unauthorized access, distribution, or reproduction of this repository or its contents is strictly prohibited.**

## ℹ️ About
**VAVE-AI (Value Analysis and Value Engineering AI)** is a proprietary intelligent system developed exclusively for **JSW Morris Garages India**. It leverages advanced Large Language Models (LLM), Visual Language Models (VLM), and vector search technologies to streamline cost reduction initiatives, automate engineering validation, and generate comprehensive technical reports.

---

## 🚀 Key Features

### 1. 🤖 AI-Powered Idea Generation & Analysis
-   **LLM Integration**: Uses GPT-2 and other advanced models to analyze cost-saving ideas.
-   **RAG (Retrieval-Augmented Generation)**: Context-aware responses using FAISS vector search on a rich engineering idea database.
-   **Automated Validation**: Checks ideas against "Physics-Informed Matrices" to ensure engineering feasibility.

### 2. 👁️ Visual Intelligence (VLM Engine)
-   **Teardown Analysis**: Automatically processes images of vehicle parts (e.g., from competitive teardowns) to identify cost-saving opportunities.
-   **Technical Drawing Analysis**: Can interpret basic engineering constraints from visual data.
-   **Module**: `vlm_engine.py`

### 3. 📄 Automated Reporting & Documentation
-   **Smart PPT Generation**: Creates professional PowerPoint presentations with charts, financial summaries, and technical comparisons. (`vave_presentation_engine.py`)
-   **Patent Documentation**: Generates draft patent applications and strategy documents. (`patent.py`, `create_patent_word_doc.py`)
-   **Excel Exports**: Detailed cost-benefit analysis spreadsheets. (`excel_generator_engine.py`)

### 4. 🗄️ Robust Data Management
-   **Data Lake Architecture**: Ingests and processes raw data from Excel, images, and unstructured text.
-   **Automated Backups**: Scripts for database snapshots and restoration (`backup.sh`, `restore.sh`).
-   **User Management**: Secure role-based access control (`users.db`).

---

## 📦 Releases & Packages
-   **[Release Notes](RELEASES.md)**: View version history, recent features, and bug fixes.
-   **[Packages & Artifacts](PACKAGES.md)**: Information on internal distribution, Docker containers, and build artifacts.

## 📂 Project Structure

```bash
VAVE_AI-JSW-MGI/
├── app.py                      # 🚀 Main Flask Application Entry Point
├── agent.py                    # 🧠 Core AI Agent Logic
├── vlm_engine.py               # 👁️ Visual Language Model Engine
├── vave_presentation_engine.py # 📊 PPT Generation Engine
├── ppt_engine.py               # 📉 Base Presentation Utilities
├── excel_generator_engine.py   # 📑 Excel Report Generator
├── data_processor.py           # 🔄 Data Ingestion & Processing
├── patent.py                   # 📜 Patent Strategy Module
│
├── model/                      # 🤖 AI Models (Gitignored - See Setup)
│   ├── gpt2.pt                 # Pre-trained LLM weights
│   └── faiss_index.bin         # Vector embeddings index
│
├── cost_reduction.db           # 💾 Main SQLite Database
├── users.db                    # 👥 User Authentication Databases
│
├── static/                     # 🖼️ Static Assets (CSS, JS, Images)
├── templates/                  # 🌐 HTML Templates (Jinja2)
├── data_lake/                  # 🌊 Raw Data Storage
│
├── deploy.sh                   # 🚀 Deployment Script
├── backup.sh                   # 🛡️ Database Backup Script
├── health_check.sh             # 💓 System Health Monitor
└── Dockerfile                  # 🐳 Container Configuration
```

---

## ⚙️ Installation & Setup

### Prerequisites
-   **Python 3.10+**
-   **Docker** (optional, for containerized deployment)
-   **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/Jaivik-Jariwala/VAVE_AI-JSW-MGI.git
cd VAVE_AI-JSW-MGI
```

### 2. ⚠️ Important: Model Files
The following large model files are **NOT** included in the repository due to size constraints. You must obtain them manually or generate them using the provided scripts:
-   `model/gpt2.pt`
-   `model/faiss_index.bin`
-   `images.zip` (if needed for testing)

**To generate/setup models:**
Run the initialization scripts (if available) or contact the administrator for the model artifacts.

### 3. Install Dependencies
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Initialize Database
```bash
python create_database.py
python create_user_db.py
```

---

## 🏃 Usage

### Run Locally
```bash
python app.py
```
Access the application at: **http://localhost:5000**

### Run with Docker
```bash
# Build the image
docker build -t vave-ai-jsw .

# Run the container
docker run -d -p 5000:5000 --name vave_container vave-ai-jsw
```

---

## 🛠️ Deployment & Operations

The repository includes utility scripts for Linux-based production environments:

-   **Deploy**: `./deploy.sh` - Pulls latest code, rebuilds Docker containers, and restarts services.
-   **Backup**: `./backup.sh` - Creates timestamped backups of `cost_reduction.db`.
-   **Restore**: `./restore.sh` - Restores the database from a backup file.
-   **Health Check**: `./health_check.sh` - Verifies system status.

---

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

---

## ⚖️ License & Legal
**PROPRIETARY AND CONFIDENTIAL**.
Copyright © 2026 JSW Morris Garages India. All Rights Reserved.

This software is protected by copyright law and international treaties. Unauthorized reproduction or distribution of this program, or any portion of it, may result in severe civil and criminal penalties, and will be prosecuted to the maximum extent possible under the law.

See [LICENSE](LICENSE) for full terms.

## 📞 Support & Credits

**Developed for**: JSW Morris Garages India  
**Team**: VAVE-AI Development Team  
**Contact**: Jaivik Jariwala (Repository Owner)

> _"Engineering Excellence through Artificial Intelligence"_
