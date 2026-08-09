# NeuroSpark

A live dashboard that reads EEG signals, classifies cognitive workload (low / moderate / high) using a trained Random Forest model and adapts the music being played based on that state (calm/ambient when you're overloaded, upbeat when you're understimulated, steady instrumental for sustained focus).

Built with Streamlit, scikit-learn and the Jamendo API for track recommendations.

## Why

Most "focus music" apps just play a static playlist. This tries to actually close the loop: read your brain state, decide what you need right now, and change the music accordingly, all live.


## Running it

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Get a free Jamendo client ID from https://devportal.jamendo.com, then:

\`\`\`bash
cp .env.example .env
# edit .env and paste your JAMENDO_CLIENT_ID in
\`\`\`

\`\`\`bash
streamlit run streamlit_app.py
\`\`\`

By default it replays real EEG samples from the dataset (`SIMULATION_MODE = "dataset_replay"` in `config.py`), so you don't need any hardware to see it work. Switch `EEG_SOURCE` to `"arduino"` in `config.py` if you've got the actual board wired up.

## Model

Random Forest trained on 8 band-power features, evaluated with a held-out subject split (never trained and tested on the same person). Current held-out macro F1 sits around 0.53 — cognitive workload from EEG is a genuinely hard 3-way classification problem, not a solved one, so this is honest performance rather than an inflated number. See `notebooks/01_eda_and_training.ipynb` for the full EDA + training walkthrough.

## Project structure

\`\`\`
neurospark/
├── streamlit_app.py          # main dashboard
├── eeg_source.py              # EEG input abstraction (arduino/dataset/synthetic)
├── music_engine.py            # Jamendo recommendations + feedback logging
├── config.py                  # single place to flip source, tags, thresholds
├── train.py                   # model training script
├── debug_jamendo.py           # standalone script to sanity-check the Jamendo API
├── src/
│   ├── features.py             # band-power feature extraction
│   └── dataset.py              # builds the training feature table
├── models/                    # trained classifier + metadata
├── data/                      # EEG dataset (feedback log is gitignored)
├── notebooks/                  # EDA + training notebook
└── arduino/                   # firmware for the real hardware
\`\`\`
