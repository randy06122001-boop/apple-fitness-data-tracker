# ⌚ Apple Watch Ultra Data Analyzer

An AI-powered, privacy-first health data analysis tool that parses your Apple Watch Ultra data, uncovers personalized trends, and syncs actionable insights directly to your Obsidian vault.

## ✨ Features

- **📥 Smart XML Parser** — Memory-efficient streaming parser handles multi-GB `export.xml` files with Parquet caching
- **📈 Trend Analysis** — Rolling averages, correlations, anomaly detection, sleep & workout breakdowns
- **🤖 AI Health Coach** — Local LLM via Ollama provides personalized summaries, recommendations, and interactive Q&A
- **📝 Obsidian Integration** — Auto-generates weekly summaries, metric tracking pages, coaching logs, and anomaly reports
- **🔒 100% Local** — All data stays on your machine. No cloud. No third-party APIs.

## 📋 Prerequisites

- **Python 3.10+**
- **Ollama** (optional, for AI features) — [Install Ollama](https://ollama.com/)
- **Obsidian** (optional, for vault export) — [Install Obsidian](https://obsidian.md/)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd apple-watch-analyzer
pip install -r requirements.txt
```

### 2. Export Your Apple Health Data

On your iPhone:
1. Open the **Health** app
2. Tap your **profile picture** (top right)
3. Scroll down and tap **Export All Health Data**
4. Transfer the `export.xml` file to your PC

### 3. Configure Paths

Edit `config.py` to set your paths:

```python
# Path to your export.xml
EXPORT_XML_PATH = Path(r"C:\Users\YourName\Documents\apple_health_export\export.xml")

# Your Obsidian vault path
OBSIDIAN_VAULT_PATH = Path(r"C:\Users\YourName\Documents\ObsidianVault")

# Ollama model to use
OLLAMA_MODEL = "llama3"
```

Or use environment variables:
```bash
set HEALTH_EXPORT_PATH=C:\path\to\export.xml
set OBSIDIAN_VAULT_PATH=C:\path\to\vault
set OLLAMA_MODEL=llama3
```

### 4. Start Ollama (Optional)

```bash
ollama serve
ollama pull llama3
```

### 5. Run the Analyzer

```bash
# Full pipeline: parse → analyze → AI coach → Obsidian export
python main.py

# Skip AI features (no Ollama needed)
python main.py --skip-ai

# Force re-parse cached data
python main.py --force-parse

# Enter interactive chat mode after analysis
python main.py --chat

# Skip Obsidian export
python main.py --skip-obsidian
```

## 📁 Project Structure

```
apple-watch-analyzer/
├── main.py              # Entry point — orchestrates the full pipeline
├── config.py            # All paths, settings, and metric definitions
├── parser.py            # Streaming XML parser with Parquet caching
├── analyzer.py          # Trend analysis, correlations, anomaly detection
├── ai_coach.py          # Ollama LLM integration for AI coaching
├── obsidian_export.py   # Markdown export engine for Obsidian vault
├── requirements.txt     # Python dependencies
└── README.md            # This file
```

## 📊 Metrics Tracked

| Category | Metrics |
|----------|---------|
| **Cardiovascular** | Resting Heart Rate, HRV, VO2 Max, Walking HR Average |
| **Activity** | Step Count, Active Energy, Exercise Minutes |
| **Sleep** | Sleep Duration, Sleep Stages (Core, Deep, REM, Awake) |
| **Ultra-Specific** | Running Power, Ground Contact Time, Vertical Oscillation, Water Temperature, Underwater Depth |
| **Environment** | Wrist Temperature, Environmental Audio Exposure |

## 📝 Obsidian Vault Structure

After running the analyzer, your vault will contain:

```
Health_Data/
├── Summaries/
│   └── Weekly Summary 2026-06-12.md
├── Metrics/
│   ├── Resting Heart Rate.md
│   ├── Heart Rate Variability.md
│   └── ...
├── AI_Coaching/
│   └── Coaching Log.md
└── Anomalies/
    └── Anomaly Log.md
```

## 💬 Interactive Chat Examples

```
You: Why was my resting heart rate high last week?
🤖 Coach: Your RHR averaged 58.2 bpm last week, up from your 30-day average of 53.1...

You: Does running late at night affect my sleep?
🤖 Coach: Looking at your data, workouts after 7 PM correlate with a 12% reduction...

You: What should I focus on this week?
🤖 Coach: Based on your trends, I recommend focusing on recovery...
```

## 🔒 Privacy

This tool processes **all data locally**. Your health data never leaves your machine:
- XML parsing happens in-memory
- Cached data is stored as local Parquet files
- The AI coach runs through a local Ollama instance
- Obsidian notes are written to your local vault
