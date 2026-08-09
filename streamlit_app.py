import sys
import os
import time
from collections import Counter, deque

from datetime import datetime
from fpdf import FPDF

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

if "session_start_ts" not in st.session_state:
    st.session_state.session_start_ts = time.time()
if "session_log" not in st.session_state:
    st.session_state.session_log = []
if "tracks_played" not in st.session_state:
    st.session_state.tracks_played = []

def _fading_audio_player(url, key, fade_seconds=2.0):
    fade_ms = int(fade_seconds * 1000)
    html = f"""
    <div style="font-family: sans-serif; display:flex; align-items:center; gap:10px;">
      <button id="btn_{key}" title="Play/Pause" style="
        width:36px; height:36px; border-radius:50%; border:none;
        background:#1D9E75; color:#fff; display:flex; align-items:center;
        justify-content:center; cursor:pointer; flex-shrink:0;
        transition: background 0.15s ease, transform 0.1s ease;
      " onmouseover="this.style.background='#178a64'" onmouseout="this.style.background='#1D9E75'">
        <svg id="icon_{key}" width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
          <path id="path_{key}" d="M4 2.5v11l9-5.5-9-5.5z"/>
        </svg>
      </button>
      <audio id="p_{key}" src="{url}" style="flex:1; height:32px;"></audio>
    </div>
    <script>
      const audio = document.getElementById("p_{key}");
      const btn = document.getElementById("btn_{key}");
      const iconPath = document.getElementById("path_{key}");
      audio.volume = 0;

      const PLAY_PATH = "M4 2.5v11l9-5.5-9-5.5z";
      const PAUSE_PATH = "M3.5 2.5h3v11h-3v-11zM9.5 2.5h3v11h-3v-11z";

      function syncIcon() {{
        iconPath.setAttribute("d", audio.paused ? PLAY_PATH : PAUSE_PATH);
      }}
      audio.addEventListener("play", syncIcon);
      audio.addEventListener("pause", syncIcon);
      audio.addEventListener("ended", syncIcon);

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

      btn.onclick = () => {{
        if (audio.paused) {{
          audio.play().then(fadeIn);
        }} else {{
          audio.pause();
        }}
      }};

      const playPromise = audio.play();
      if (playPromise !== undefined) {{
        playPromise.then(fadeIn).catch(() => {{
          syncIcon();
        }});
      }} else {{
        fadeIn();
      }}
      syncIcon();
    </script>
    """
    components.html(html, height=48)


@st.fragment(run_every=1)
def render_sidebar_metrics():
    st.metric("Total samples received", source.total_samples_received)
    st.metric("Buffer fill", f"{source.buffered_samples}/{config.BUFFER_SAMPLES}")
    if source.error:
        st.error(f"EEG source error: {source.error}")


with st.sidebar:
    render_sidebar_metrics()
st.title("NeuroSpark: Live EEG Stream")


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
        st.warning("Signal too flat/short this window: skipping prediction.")
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
    st.session_state.session_log.append({"ts": time.time(), "cognitive_state": cognitive_state})

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
        st.subheader("Adapted for this state")
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

            st.session_state.tracks_played.append({
    "ts": time.time(), "name": tracks[0]["name"], "artist": tracks[0]["artist_name"],
    "cognitive_state": cognitive_state, "tag": tracks[0]["tag"],
})
            for t in tracks:
                st.session_state.shown_track_ids.add(t["id"])
            if len(st.session_state.shown_track_ids) > 30:
                st.session_state.shown_track_ids = set(
                    list(st.session_state.shown_track_ids)[-30:]
                )
        else:
            tracks = st.session_state.current_tracks

        if music_engine.last_error:
            st.caption(f"Jamendo fallback: {music_engine.last_error}")

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

                st.session_state.tracks_played.append({
    "ts": time.time(), "name": replacement["name"], "artist": replacement["artist_name"],
    "cognitive_state": cognitive_state, "tag": replacement["tag"],
})
                st.toast(f"Skipped — playing \"{replacement['name']}\" instead.")
                st.rerun(scope="fragment")
def _build_session_pdf(state_counts, total_windows, fb_stats, tracks_played):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "NeuroSpark session summary", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Cognitive state distribution", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for state, count in state_counts.items():
        pdf.cell(0, 7, f"  {state}: {count} windows ({count/total_windows:.0%})", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Feedback", ln=True)
    pdf.set_font("Helvetica", "", 10)
    if fb_stats["total"]:
        pdf.cell(0, 7, f"  Likes: {fb_stats['likes']}  Dislikes: {fb_stats['dislikes']}  Ratio: {fb_stats['like_ratio']:.0%}", ln=True)
    else:
        pdf.cell(0, 7, "  No feedback recorded this session", ln=True)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Tracks played", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for t in tracks_played:
        ts_str = datetime.fromtimestamp(t["ts"]).strftime("%H:%M:%S")
        line = f"  [{ts_str}] {t['name']} - {t['artist']} ({t['cognitive_state']}/{t['tag']})"
        pdf.multi_cell(0, 6, _pdf_safe(line))

    return bytes(pdf.output())

@st.fragment(run_every=config.MUSIC_FRAGMENT_INTERVAL_SECONDS)
def render_session_export():
    st.divider()
    st.subheader("Session summary")
    if not st.session_state.session_log:
        return

    state_counts = Counter(r["cognitive_state"] for r in st.session_state.session_log)
    total_windows = sum(state_counts.values())
    fb_stats = music_engine.get_session_feedback_stats(st.session_state.session_start_ts)

    col1, col2 = st.columns(2)
    with col1:
        st.caption("State distribution (this session)")
        for state, count in state_counts.items():
            st.write(f"{state}: {count/total_windows:.0%}")
    with col2:
        st.caption("Feedback")
        if fb_stats["total"]:
            st.write(f"👍 {fb_stats['likes']} · 👎 {fb_stats['dislikes']} ({fb_stats['like_ratio']:.0%} like ratio)")
        else:
            st.write("No likes/skips yet")

    csv_df = pd.DataFrame(st.session_state.tracks_played)
    csv_bytes = csv_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇Download CSV", csv_bytes, file_name="neurospark_session.csv", mime="text/csv")
def _pdf_safe(text):
   
    return text.encode("latin-1", "replace").decode("latin-1")

    pdf_bytes = _build_session_pdf(state_counts, total_windows, fb_stats, st.session_state.tracks_played)
    st.download_button("⬇Download PDF", pdf_bytes, file_name="neurospark_session.pdf", mime="application/pdf")

render_chart()
render_state_and_music()
render_session_export()