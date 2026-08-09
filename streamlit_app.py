import sys
import os
import time
from collections import Counter, deque

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import config
from eeg_source import create_eeg_source
from features import extract_features, FEATURE_NAMES
import music_engine

st.set_page_config(page_title="NeuroSpark — Live EEG", layout="wide")

MODEL_PATH = os.path.join("models", "workload_classifier.joblib")
if "eeg_source" not in st.session_state:
    st.session_state.eeg_source = create_eeg_source()
    st.session_state.eeg_source.start()

source = st.session_state.eeg_source

if "model" not in st.session_state:
    st.session_state.model = joblib.load(MODEL_PATH)
model = st.session_state.model

if "shown_track_ids" not in st.session_state:
    st.session_state.shown_track_ids = set()
if "current_music_state" not in st.session_state:
    st.session_state.current_music_state = None
    st.session_state.current_tracks = []
if "liked_track_ids" not in st.session_state:
    st.session_state.liked_track_ids = set()
if "state_votes" not in st.session_state:
    st.session_state.state_votes = deque(maxlen=config.STATE_VOTE_WINDOW)
if "confirmed_state" not in st.session_state:
    st.session_state.confirmed_state = None
if "last_state_switch_ts" not in st.session_state:
    st.session_state.last_state_switch_ts = 0.0
if "latest_raw_state" not in st.session_state:
    st.session_state.latest_raw_state = None
if "latest_proba" not in st.session_state:
    st.session_state.latest_proba = {}


def _fading_audio_player(url, key, fade_seconds=2.0):
    fade_ms = int(fade_seconds * 1000)
    html = f"""
    <div style="font-family: sans-serif;">
      <audio id="p_{key}" src="{url}" style="width:100%;"></audio>
      <button id="btn_{key}" style="display:none; margin-top:4px;">▶ Tap to play</button>
    </div>
    <script>
      const audio = document.getElementById("p_{key}");
      const btn = document.getElementById("btn_{key}");
      audio.volume = 0;

      function fadeIn() {{
        const steps = 20;
        const stepTime = {fade_ms} / steps;
        let i = 0;
        const iv = setInterval(() => {{
          i += 1;
          audio.volume = Math.min(1.0, i / steps);
          if (i >= steps) clearInterval(iv);
        }}, stepTime);
      }}

      const playPromise = audio.play();
      if (playPromise !== undefined) {{
        playPromise.then(fadeIn).catch(() => {{
          btn.style.display = "inline-block";
          btn.onclick = () => {{
            audio.play().then(fadeIn);
            btn.style.display = "none";
          }};
        }});
      }} else {{
        fadeIn();
      }}
    </script>
    """
    components.html(html, height=60)

st.sidebar.header("EEG Input")
st.sidebar.markdown(f"**Active source:** `{config.EEG_SOURCE}`")
st.sidebar.caption("Change this in config.py (EEG_SOURCE), not here.")

if config.EEG_SOURCE == "simulation":
    st.sidebar.markdown(f"**Simulation mode:** `{config.SIMULATION_MODE}`")
    if config.SIMULATION_MODE == "synthetic":
        state_choice = st.sidebar.select_slider(
            "Simulated cognitive state (dev only)",
            options=["relaxed", "moderate", "focused"],
            value=source.cognitive_state,
        )
        source.cognitive_state = state_choice


@st.fragment(run_every=1)
def render_sidebar_metrics():
    st.metric("Total samples received", source.total_samples_received)
    st.metric("Buffer fill", f"{source.buffered_samples}/{config.BUFFER_SAMPLES}")
    if source.error:
        st.error(f"EEG source error: {source.error}")


with st.sidebar:
    render_sidebar_metrics()
st.title("🧠 NeuroSpark — Live EEG Stream")


@st.fragment(run_every=1)
def render_chart():
    window = source.get_latest_window(config.WINDOW_SAMPLES)

    if window is None:
        st.info(
            f"Buffering... {source.buffered_samples}/{config.WINDOW_SAMPLES} samples "
            f"(need {config.WINDOW_SECONDS}s of data before the first window is ready)."
        )
        return

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"Last {config.WINDOW_SECONDS}s of raw EEG")
        st.line_chart(window)

    with col2:
        st.subheader("Window stats")
        st.metric("Mean", f"{window.mean():.1f}")
        st.metric("Std dev", f"{window.std():.1f}")
        st.metric("Min / Max", f"{window.min():.0f} / {window.max():.0f}")
    feats = extract_features(window, fs=source.sample_rate)

    if feats is None:
        st.warning("Signal too flat/short this window — skipping prediction.")
        return

    x = pd.DataFrame([[feats[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
    raw_state = model.predict(x)[0]
    proba = dict(zip(model.classes_, model.predict_proba(x)[0]))

    st.session_state.state_votes.append(raw_state)
    st.session_state.latest_raw_state = raw_state
    st.session_state.latest_proba = proba


@st.fragment(run_every=config.MUSIC_FRAGMENT_INTERVAL_SECONDS)
def render_state_and_music():
    votes = st.session_state.state_votes
    raw_state = st.session_state.latest_raw_state
    proba = st.session_state.latest_proba

    st.divider()
    col_a, col_b = st.columns([1, 2])

    if not votes:
        with col_a:
            st.info("Waiting for predictions...")
        return

    vote_counts = Counter(votes)
    majority_state, majority_count = vote_counts.most_common(1)[0]
    vote_confidence = majority_count / len(votes)

    now = time.time()
    dwell_elapsed = now - st.session_state.last_state_switch_ts
    can_switch = dwell_elapsed >= config.MIN_STATE_DWELL_SECONDS

    is_first_confirmation = st.session_state.confirmed_state is None
    should_switch = majority_state != st.session_state.confirmed_state and (
        is_first_confirmation or can_switch
    )

    if should_switch:
        st.session_state.confirmed_state = majority_state
        st.session_state.last_state_switch_ts = now

    cognitive_state = st.session_state.confirmed_state

    with col_a:
        st.subheader("Cognitive state")
        st.metric("Driving music", cognitive_state.upper())
        st.caption(f"Instantaneous: {raw_state} · vote confidence: {vote_confidence:.0%}")
        if proba:
            st.caption("  ·  ".join(f"{k}: {v:.0%}" for k, v in sorted(proba.items())))
        if not can_switch and not is_first_confirmation:
            remaining = max(0, config.MIN_STATE_DWELL_SECONDS - dwell_elapsed)
            st.caption(f"Playlist locked for {remaining:.0f}s more (prevents rapid swapping)")

    with col_b:
        st.subheader("🎵 Adapted for this state")
        tags = config.MOOD_TAGS[cognitive_state]
        best_tag = music_engine.preferred_tag(cognitive_state, tags)

        state_changed = cognitive_state != st.session_state.current_music_state
        if state_changed:
            tracks = music_engine.recommend_tracks(
                cognitive_state,
                exclude_ids=st.session_state.shown_track_ids,
                preferred_tag=best_tag,
            )
            st.session_state.current_music_state = cognitive_state
            st.session_state.current_tracks = tracks
            for t in tracks:
                st.session_state.shown_track_ids.add(t["id"])
            if len(st.session_state.shown_track_ids) > 30:
                st.session_state.shown_track_ids = set(
                    list(st.session_state.shown_track_ids)[-30:]
                )
        else:
            tracks = st.session_state.current_tracks

        if music_engine.last_error:
            st.caption(f"⚠️ Jamendo fallback: {music_engine.last_error}")

        # only the top track is shown -- highlighted as "now playing"
        # instead of a list, since one clear pick reads better than a
        # scroll of five during a live demo
        now_playing = tracks[0]

        with st.container(border=True):
            img_col, info_col = st.columns([1, 3])
            with img_col:
                if now_playing.get("image"):
                    st.image(now_playing["image"], width=110)
            with info_col:
                st.caption("▶ NOW PLAYING")
                st.markdown(f"### {now_playing['name']}")
                st.markdown(f"*{now_playing['artist_name']}*")

            if now_playing["audio"]:
                # fading player, keyed by track id -- Streamlit keeps the
                # same mounted component (no restart, no re-fade) across
                # reruns as long as the key is unchanged; it only
                # remounts when the track actually changes.
                # No "start session" gate needed: the player tries to
                # autoplay directly and falls back to an inline
                # "Tap to play" button if the browser blocks it.
                _fading_audio_player(now_playing["audio"], key=now_playing["id"])

            already_liked = now_playing["id"] in st.session_state.liked_track_ids

            fb_cols = st.columns(2)
            like_label = "❤️ Liked" if already_liked else "👍 Like"
            if fb_cols[0].button(
                like_label,
                key=f"up_{now_playing['id']}",
                use_container_width=True,
                disabled=already_liked,
            ):
                music_engine.log_feedback(
                    now_playing["id"], cognitive_state, now_playing["tag"], True
                )
                st.session_state.liked_track_ids.add(now_playing["id"])
                st.toast(f"Liked — will favor \"{now_playing['tag']}\" more for {cognitive_state} state.")
                st.rerun(scope="fragment")

            if fb_cols[1].button("👎 Skip", key=f"down_{now_playing['id']}", use_container_width=True):
                music_engine.log_feedback(
                    now_playing["id"], cognitive_state, now_playing["tag"], False
                )
                # instant swap -- don't make the user wait for the next
                # state change to hear something different
                replacement = music_engine.swap_track(
                    cognitive_state, now_playing["tag"], st.session_state.shown_track_ids
                )
                st.session_state.shown_track_ids.add(replacement["id"])
                st.session_state.current_tracks[0] = replacement
                st.toast(f"Skipped — playing \"{replacement['name']}\" instead.")
                st.rerun(scope="fragment")


render_chart()
render_state_and_music()