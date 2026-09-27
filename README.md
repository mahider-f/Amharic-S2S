# Amharic Speech-to-Speech Demo

**Microphone → Ethio-ASR → Gemma 3 4B → MMS-TTS → Speech**

## Recommended environment
- Python 3.11 or 3.12
- 8 GB RAM or more
- Internet connection for initial model downloads and Gemini API calls
- CPU is supported; GPU is optional

## Linux / macOS / WSL
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python demo_app_gemma.py
```
Open http://127.0.0.1:7860


The first launch downloads the ASR and TTS checkpoints and will be slower. Do not commit API keys.
