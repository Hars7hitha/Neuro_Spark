# NeuroSpark

A live dashboard that reads EEG signals, classifies cognitive workload (**low / moderate / high**) using a trained Random Forest model, and adapts the music being played based on that state: calm/ambient when you're overloaded, upbeat when you're understimulated, and steady instrumental for sustained focus.

Built with **Streamlit, scikit-learn, and the Jamendo API** for track recommendations.

## Why

Most "focus music" apps just play a static playlist. NeuroSpark tries to close the loop: **read your brain state → determine what you need → adapt the music**, all in real time.

## Running it

```bash
pip install -r requirements.txt
```

Get a free Jamendo client ID from [Jamendo Developer Portal](https://devportal.jamendo.com), then:

```bash
cp .env.example .env
```

Edit `.env` and add your `JAMENDO_CLIENT_ID`.

Then run:

```bash
streamlit run streamlit_app.py
```

By default, NeuroSpark replays real EEG samples from the dataset (`SIMULATION_MODE = "dataset_replay"` in `config.py`), so no hardware is required. To use the actual Arduino board, switch `EEG_SOURCE` to `"arduino"` in `config.py`.

## Model

A **Random Forest** model trained on 8 band-power features is used for cognitive workload classification. Evaluation uses a **held-out subject split**, ensuring the model is not trained and tested on the same person.

The current held-out macro-F1 is around **0.53**. Cognitive workload classification from EEG is a challenging 3-class problem, so this represents realistic rather than inflated performance.

See `notebooks/01_eda_and_training.ipynb` for the complete EDA and training workflow.

## Project Structure

```text
neurospark/
├── streamlit_app.py          # Main dashboard
├── eeg_source.py             # EEG input abstraction
├── music_engine.py           # Jamendo recommendations + feedback logging
├── config.py                 # Configuration and settings
├── train.py                  # Model training script
├── debug_jamendo.py          # Jamendo API testing script
├── src/
│   ├── features.py           # Band-power feature extraction
│   └── dataset.py            # Training feature table generation
├── models/                   # Trained classifier + metadata
├── data/                     # EEG dataset
├── notebooks/                # EDA + training notebook
└── arduino/                  # Arduino firmware
```
