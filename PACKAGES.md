# Packages & Artifacts

## Distribution Packages
This software is distributed internally as a containerized application and a set of Python modules.

### 1. Docker Container
The primary deployment artifact is the Docker image `vave-ai-jsw`.
*   **Source**: Built from the root `Dockerfile`.
*   **Registry**: Internal GitLab registry (or local build).
*   **Usage**: `docker pull registry.internal.jswgmylie.com/vave-ai/app:latest` (Example)

### 2. Python Modules
The application is structured as several modular Python components.
*   **`vave_ai_core`**: The central logic (`agent.py`, `app.py`).
*   **`vlm_engine`**: Visual Language Model processing (`vlm_engine.py`).
*   **`presentation_engine`**: Automated reporting (`vave_presentation_engine.py`, `ppt_engine.py`).
*   **`patent_engine`**: Patent strategy generation (`patent.py`).

### 3. Build Dependencies
*   **Python Environment**: Specified in `requirements.txt`.
*   **Large Model Files**: Managed via Git LFS or manual download scripts.
    *   `model/gpt2.pt`
    *   `model/faiss_index.bin`

## Installation
To install the package locally for development:

```bash
pip install -r requirements.txt
python setup_vlm_dirs.py
python create_database.py
```

## Security Note
Do **NOT** publish these packages to public repositories like PyPI or npm. They contain proprietary business logic and sensitive configuration data.
