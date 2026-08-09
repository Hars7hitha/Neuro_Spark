import json
import os
import time
import random

import requests

import config

JAMENDO_BASE_URL = "https://api.jamendo.com/v3.0/tracks/"
FEEDBACK_LOG_PATH = os.path.join("data", "feedback_log.json")


_FALLBACK_TRACKS = [
    {
        "id": "fallback-1",
        "name": "Offline placeholder track",
        "artist_name": "NeuroSpark",
        "audio": "",
        "image": "",
        "tag": "fallback",
    }
]


def _request_tracks(tags, limit):
    params = {
        "client_id": config.JAMENDO_CLIENT_ID,
        "format": "json",
        "limit": limit,
        "fuzzytags": "+".join(tags),
        "order": "popularity_total",
        "audioformat": "mp32",
        "include": "musicinfo",
    }
    resp = requests.get(JAMENDO_BASE_URL, params=params, timeout=8)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("headers", {}).get("status")
    if status != "success":
        raise ValueError(data.get("headers", {}).get("error_message", "Jamendo API error"))
    return data.get("results", [])


last_error = None  


def recommend_tracks(cognitive_state, exclude_ids=None, limit=None, preferred_tag=None):
 
    global last_error
    limit = limit or config.TRACKS_PER_RECOMMENDATION
    exclude_ids = exclude_ids or set()
    tags = config.MOOD_TAGS.get(cognitive_state, config.MOOD_TAGS["moderate"])

    search_tags = [preferred_tag] if preferred_tag else tags

    try:
        raw = _request_tracks(search_tags, limit=limit + 5)
        if preferred_tag and len(raw) < limit:
            raw = _request_tracks(tags, limit=limit + 5)
            search_tags = tags
    except (requests.RequestException, ValueError) as e:
        last_error = f"{type(e).__name__}: {e}"
        return _FALLBACK_TRACKS[:limit]

    tracks = []
    for t in raw:
        if str(t.get("id")) in exclude_ids:
            continue
        tracks.append(
            {
                "id": str(t.get("id")),
                "name": t.get("name", "Unknown"),
                "artist_name": t.get("artist_name", "Unknown artist"),
                "audio": t.get("audio", ""),
                "image": t.get("image", ""),
                "tag": search_tags[0],
            }
        )
        if len(tracks) >= limit:
            break

    if not tracks:
        last_error = (
            f"Jamendo returned {len(raw)} raw result(s) for tags={tags}, "
            f"but 0 remained after excluding {len(exclude_ids)} previously-shown "
            f"track ids." if raw else
            f"Jamendo returned 0 raw results for tags={tags} — API call succeeded "
            f"but the tag combo matched nothing."
        )
        return _FALLBACK_TRACKS[:limit]

    last_error = None
    return tracks

def _load_feedback_log():
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return []
    with open(FEEDBACK_LOG_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def log_feedback(track_id, cognitive_state, tag, liked):
    os.makedirs(os.path.dirname(FEEDBACK_LOG_PATH), exist_ok=True)
    log = _load_feedback_log()
    log.append(
        {
            "track_id": track_id,
            "cognitive_state": cognitive_state,
            "tag": tag,
            "liked": bool(liked),
            "ts": time.time(),
        }
    )
    with open(FEEDBACK_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2)


def preferred_tag(cognitive_state, default_tags):
    log = _load_feedback_log()
    now = time.time()
    halflife_seconds = config.FEEDBACK_HALFLIFE_DAYS * 86400
    scores = {}
    for tag in default_tags:
        rows = [r for r in log if r["cognitive_state"] == cognitive_state and r["tag"] == tag]
        if len(rows) < 3:
            continue
        weighted_sum = 0.0
        weight_total = 0.0
        for r in rows:
            age = now - r.get("ts", now)
            w = 0.5 ** (age / halflife_seconds) if halflife_seconds > 0 else 1.0
            weighted_sum += w * (1.0 if r["liked"] else 0.0)
            weight_total += w
        if weight_total > 0:
            scores[tag] = weighted_sum / weight_total

    if not scores:
        return default_tags[0]
    return max(scores, key=scores.get)


def swap_track(cognitive_state, tag, exclude_ids):
    try:
        raw = _request_tracks([tag], limit=10)
    except (requests.RequestException, ValueError) as e:
        global last_error
        last_error = f"{type(e).__name__}: {e}"
        return _FALLBACK_TRACKS[0]

    for t in raw:
        if str(t.get("id")) in exclude_ids:
            continue
        return {
            "id": str(t.get("id")),
            "name": t.get("name", "Unknown"),
            "artist_name": t.get("artist_name", "Unknown artist"),
            "audio": t.get("audio", ""),
            "image": t.get("image", ""),
            "tag": tag,
        }
    return _FALLBACK_TRACKS[0]


def get_feedback_summary():
    log = _load_feedback_log()
    counts = {}
    for r in log:
        s = r["cognitive_state"]
        counts.setdefault(s, {"cognitive_state": s, "likes": 0, "dislikes": 0})
        if r["liked"]:
            counts[s]["likes"] += 1
        else:
            counts[s]["dislikes"] += 1
    return list(counts.values())