# ── Standard Library ────────────────────────────────────────────
import os, sys, re, time, queue, threading, logging, shutil
import subprocess, webbrowser, collections, json, traceback, math
from pathlib import Path
from datetime import datetime

# ── Third-party ─────────────────────────────────────────────────
import numpy as np
import sounddevice as sd
import pyautogui

pyautogui.FAILSAFE = True          # move mouse to corner to abort
pyautogui.PAUSE    = 0.01


# ══════════════════════════════════════════════════════════════════
#  CONFIGURATION  (edit config.json instead of touching code)
# ══════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "sample_rate":          16000,
    "frame_duration_ms":    30,        # VAD frame: 10 / 20 / 30 ms only
    "vad_aggressiveness":   2,         # 0=permissive … 3=strict
    "silence_timeout_sec":  1.2,       # stop recording after N secs silence
    "max_record_sec":       12,        # hard cap per utterance
    "confidence_threshold": -1.05,     # whisper avg_logprob cutoff
    "whisper_model":        "base",    # tiny / base / small / medium
    "whisper_device":       "cpu",     # cpu / cuda
    "whisper_compute_type": "int8",    # int8 / float16 / float32
    "noise_profile_sec":    1.0,
    "wake_words":           ["start", "wake up", "activate"],
    "sleep_words":          ["sleep", "go to sleep", "deactivate"],
    "kill_words":           ["close program", "shut down", "exit program"],
    "log_to_file":          True,
    "log_file":             "smart_pen.log",
    "command_queue_max":    12,        # drop oldest if queue full
    "tray_icon":            True,
    # ── drawing canvas size defaults ───────────────────────────
    "canvas_width":         800,
    "canvas_height":        600,
    # ── FEATURE 1: live/streaming transcription ─────────────────
    "live_typing_enabled":  True,      # stream words while speaking
    "live_chunk_sec":       0.6,       # chunk size for streaming ASR
}

CONFIG_FILE = Path("smart_pen_config.json")

def load_config() -> dict:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                user = json.load(f)
            cfg = {**DEFAULT_CONFIG, **user}
            return cfg
        except Exception:
            pass
    with open(CONFIG_FILE, "w") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2)
    return DEFAULT_CONFIG.copy()

CFG = load_config()


# ══════════════════════════════════════════════════════════════════
#  LOGGING
# ══════════════════════════════════════════════════════════════════

handlers = [logging.StreamHandler(sys.stdout)]
if CFG["log_to_file"]:
    handlers.append(logging.FileHandler(CFG["log_file"], encoding="utf-8"))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    datefmt = "%H:%M:%S",
    handlers= handlers,
)
log = logging.getLogger("SmartPen")


# ══════════════════════════════════════════════════════════════════
#  OPTIONAL IMPORTS (graceful degradation)
# ══════════════════════════════════════════════════════════════════

# VAD
try:
    import webrtcvad
    VAD_AVAILABLE = True
except ImportError:
    VAD_AVAILABLE = False
    log.warning("webrtcvad not found → fixed-duration recording fallback.")

# Noise reduction
try:
    import noisereduce as nr
    NR_AVAILABLE = True
except ImportError:
    NR_AVAILABLE = False

# ASR — prefer faster-whisper
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER = True
except ImportError:
    try:
        import whisper as openai_whisper
        FASTER_WHISPER = False
        log.warning("faster-whisper not found → using openai-whisper (slower).")
    except ImportError:
        log.critical("No Whisper library found. Install faster-whisper.")
        sys.exit(1)

# Punctuation restoration
try:
    from deepmultilingualpunctuation import PunctuationModel as _PM
    _punct_instance = _PM()
    PUNCT_AVAILABLE = True
    log.info("Punctuation model loaded.")
except Exception:
    PUNCT_AVAILABLE = False
    log.warning("deepmultilingualpunctuation not found → basic punctuation only.")

# Active window detection
try:
    import pygetwindow as gw
    WINDOW_AVAILABLE = True
except ImportError:
    WINDOW_AVAILABLE = False

# Keyboard hotkey
try:
    import keyboard
    KEYBOARD_AVAILABLE = True
except ImportError:
    KEYBOARD_AVAILABLE = False
    log.warning("keyboard lib not found → always-listen mode only.")

# System tray icon
try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_AVAILABLE = True
except ImportError:
    TRAY_AVAILABLE = False

# Drawing support (tkinter is stdlib)
try:
    import tkinter as tk
    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    log.warning("tkinter not available → drawing commands disabled.")


# ══════════════════════════════════════════════════════════════════
#  THREAD-SAFE STATE
# ══════════════════════════════════════════════════════════════════

class PenState:
    MODES = {"typing", "presentation"}

    def __init__(self):
        self._lock         = threading.Lock()
        self._mode         = "typing"
        self._active       = False
        self._running      = True
        self._last_cmd: str = ""
        self._cmd_history: list = []   # (timestamp, text, action)
        # slideshow-search state
        self._in_slideshow: bool  = False
        self._slideshow_win       = None   # pygetwindow handle
        # FEATURE 1: live-typing state
        self._live_typed_words: list = []  # words already typed in current utterance

    # ── properties ──────────────────────────────────────────────
    @property
    def mode(self):
        with self._lock: return self._mode

    @mode.setter
    def mode(self, v):
        assert v in self.MODES
        with self._lock:
            self._mode = v
            log.info(f"Mode → {v.upper()}")

    @property
    def active(self):
        with self._lock: return self._active

    @active.setter
    def active(self, v):
        with self._lock:
            self._active = v
            log.info("🟢 ACTIVE" if v else "😴 SLEEPING")

    @property
    def running(self):
        with self._lock: return self._running

    @running.setter
    def running(self, v):
        with self._lock: self._running = v

    @property
    def in_slideshow(self):
        with self._lock: return self._in_slideshow

    @in_slideshow.setter
    def in_slideshow(self, v):
        with self._lock: self._in_slideshow = v

    def record_history(self, text: str, action: str):
        with self._lock:
            self._last_cmd = text
            self._cmd_history.append({
                "time":   datetime.now().isoformat(timespec="seconds"),
                "text":   text,
                "action": action,
            })
            if len(self._cmd_history) > 100:
                self._cmd_history.pop(0)

    @property
    def last_cmd(self):
        with self._lock: return self._last_cmd

    # FEATURE 1 helpers
    def clear_live_words(self):
        with self._lock: self._live_typed_words = []

    def get_live_words(self) -> list:
        with self._lock: return list(self._live_typed_words)

    def add_live_words(self, words: list):
        with self._lock: self._live_typed_words.extend(words)


STATE = PenState()

# Bounded queue — drops oldest if full to prevent stale backlog
_raw_queue: queue.Queue = queue.Queue(maxsize=CFG["command_queue_max"])

def enqueue(text: str):
    """Put command; drop oldest if full."""
    if _raw_queue.full():
        try: _raw_queue.get_nowait()
        except queue.Empty: pass
        log.warning("Queue full — dropped oldest command.")
    _raw_queue.put(text)


# ══════════════════════════════════════════════════════════════════
#  NOISE PROFILE (captured once at startup)
# ══════════════════════════════════════════════════════════════════

_noise_profile: np.ndarray | None = None

def capture_noise_profile():
    global _noise_profile
    if not NR_AVAILABLE:
        return
    secs = CFG["noise_profile_sec"]
    log.info(f"🔇 Recording noise profile ({secs}s) — stay quiet…")
    try:
        raw = sd.rec(int(secs * CFG["sample_rate"]),
                     samplerate=CFG["sample_rate"],
                     channels=1, dtype="float32")
        sd.wait()
        _noise_profile = raw.flatten()
        log.info("✅ Noise profile captured.")
    except Exception as e:
        log.warning(f"Noise profile failed: {e}")


# ══════════════════════════════════════════════════════════════════
#  AUDIO RECORDING  (VAD-based + fixed fallback)
# ══════════════════════════════════════════════════════════════════

def _denoise(audio: np.ndarray) -> np.ndarray:
    if NR_AVAILABLE and _noise_profile is not None and len(audio) > 0:
        try:
            return nr.reduce_noise(y=audio, sr=CFG["sample_rate"],
                                   y_noise=_noise_profile)
        except Exception:
            pass
    return audio


def record_vad() -> np.ndarray:
    """Record using VAD — stops on sustained silence."""
    if not VAD_AVAILABLE:
        return _record_fixed()

    sr        = CFG["sample_rate"]
    frame_ms  = CFG["frame_duration_ms"]
    n_samples = int(sr * frame_ms / 1000)
    silence_frames = int(CFG["silence_timeout_sec"] * 1000 / frame_ms)

    vad    = webrtcvad.Vad(CFG["vad_aggressiveness"])
    ring   = collections.deque(maxlen=silence_frames)
    frames = []
    speaking  = False
    start     = time.time()

    log.info("🎙  Listening…")

    try:
        stream = sd.InputStream(samplerate=sr, channels=1,
                                dtype="int16", blocksize=n_samples)
        with stream:
            while STATE.running:
                if time.time() - start > CFG["max_record_sec"]:
                    log.info("⏱  Max duration reached.")
                    break

                chunk, _ = stream.read(n_samples)
                flat      = chunk.flatten()
                raw_bytes = flat.tobytes()

                expected = n_samples * 2
                if len(raw_bytes) < expected:
                    continue

                is_speech = vad.is_speech(raw_bytes[:expected], sr)
                ring.append(is_speech)

                if is_speech:
                    speaking = True
                if speaking:
                    frames.append(flat.astype("float32") / 32768.0)

                if speaking and len(ring) == ring.maxlen and not any(ring):
                    break
    except Exception as e:
        log.error(f"VAD stream error: {e}")
        return np.array([], dtype="float32")

    if not frames:
        return np.array([], dtype="float32")

    audio = np.concatenate(frames)
    return _denoise(audio)


def _record_fixed(duration: float = 3.0) -> np.ndarray:
    sr = CFG["sample_rate"]
    log.info(f"🎙  Recording {duration}s (fixed)…")
    raw = sd.rec(int(duration * sr), samplerate=sr,
                 channels=1, dtype="float32")
    sd.wait()
    audio = raw.flatten()
    if np.max(np.abs(audio)) > 0:
        audio /= np.max(np.abs(audio))
    return _denoise(audio)


# ══════════════════════════════════════════════════════════════════
#  ASR — WHISPER TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════

_asr_model = None

def load_asr_model():
    global _asr_model
    if FASTER_WHISPER:
        log.info(f"⚡ Loading faster-whisper [{CFG['whisper_model']}] "
                 f"on {CFG['whisper_device']} ({CFG['whisper_compute_type']})…")
        _asr_model = WhisperModel(
            CFG["whisper_model"],
            device       = CFG["whisper_device"],
            compute_type = CFG["whisper_compute_type"],
        )
    else:
        log.info(f"🐢 Loading openai-whisper [{CFG['whisper_model']}]…")
        _asr_model = openai_whisper.load_model(CFG["whisper_model"])
    log.info("✅ ASR model ready.")


_INITIAL_PROMPT = (
    "typing mode, presentation mode, open youtube, delete word, "
    "new paragraph, next slide, undo, copy, paste, "
    "draw circle, draw rectangle, search on gemini, add slide"
)

def transcribe(audio: np.ndarray) -> tuple[str, float]:
    """Returns (text, confidence).  confidence in [0, 1]."""
    if len(audio) < CFG["sample_rate"] * 0.25:
        return "", 0.0

    try:
        if FASTER_WHISPER:
            segments, info = _asr_model.transcribe(
                audio,
                beam_size      = 5,
                initial_prompt = _INITIAL_PROMPT,
                vad_filter     = True,
                vad_parameters = {"min_silence_duration_ms": 500},
            )
            text = " ".join(s.text for s in segments).lower().strip()
            conf = getattr(info, "language_probability", 0.8)
            return text, conf

        else:
            result = _asr_model.transcribe(
                audio,
                fp16           = False,
                initial_prompt = _INITIAL_PROMPT,
            )
            text = result["text"].lower().strip()
            raw_conf = result.get("avg_logprob", -0.5)
            conf     = max(0.0, min(1.0, 1.0 + raw_conf))
            return text, conf

    except Exception as e:
        log.error(f"Transcription error: {e}")
        return "", 0.0


# ══════════════════════════════════════════════════════════════════
#  FEATURE 1: REAL-TIME / STREAMING TRANSCRIPTION
# ══════════════════════════════════════════════════════════════════

def _transcribe_chunk(audio: np.ndarray) -> str:
    """
    Lightweight transcription of a short audio chunk (0.5–1 s).
    Returns plain lowercase text; empty string on failure.
    Uses beam_size=1 for speed.
    """
    if len(audio) < CFG["sample_rate"] * 0.2:
        return ""
    try:
        if FASTER_WHISPER:
            segments, _ = _asr_model.transcribe(
                audio,
                beam_size      = 1,
                initial_prompt = _INITIAL_PROMPT,
                vad_filter     = False,
            )
            return " ".join(s.text for s in segments).lower().strip()
        else:
            result = _asr_model.transcribe(
                audio, fp16=False, initial_prompt=_INITIAL_PROMPT
            )
            return result["text"].lower().strip()
    except Exception:
        return ""


def _words_of(text: str) -> list[str]:
    """Split text into non-empty word tokens."""
    return [w for w in text.split() if w]


def _new_words(all_words: list[str], already_typed: list[str]) -> list[str]:
    """
    Return only the suffix of all_words that has not been typed yet.
    Uses a simple longest-prefix match so we never repeat words.
    """
    n = len(already_typed)
    if not n:
        return all_words
    # Walk forward matching; if the running transcript re-ordered or
    # hallucinated we take no new words (safe).
    if all_words[:n] == already_typed:
        return all_words[n:]
    # Partial prefix match — find longest common prefix
    matched = 0
    for a, b in zip(already_typed, all_words):
        if a == b:
            matched += 1
        else:
            break
    return all_words[matched:]


def record_vad_streaming() -> np.ndarray:
    """
    FEATURE 1: VAD-based recording that ALSO types words in real time
    as audio is captured.  Returns the full audio (same shape as
    record_vad) so the caller can do a final full-accuracy pass.

    Falls back to record_vad() if:
      • live_typing_enabled is False
      • VAD not available
      • mode is not typing
    """
    if (not CFG.get("live_typing_enabled", True)
            or not VAD_AVAILABLE
            or STATE.mode != "typing"):
        return record_vad()

    sr         = CFG["sample_rate"]
    frame_ms   = CFG["frame_duration_ms"]
    n_samples  = int(sr * frame_ms / 1000)
    silence_frames = int(CFG["silence_timeout_sec"] * 1000 / frame_ms)
    chunk_samples  = int(CFG.get("live_chunk_sec", 0.6) * sr)

    vad      = webrtcvad.Vad(CFG["vad_aggressiveness"])
    ring     = collections.deque(maxlen=silence_frames)
    frames   = []          # all frames (for final pass)
    speaking = False
    start    = time.time()

    # Live-typing bookkeeping
    STATE.clear_live_words()
    chunk_buf: list[np.ndarray] = []   # frames collected since last chunk
    chunk_sample_count = 0

    log.info("🎙  Listening (live-type mode)…")

    try:
        stream = sd.InputStream(samplerate=sr, channels=1,
                                dtype="int16", blocksize=n_samples)
        with stream:
            while STATE.running:
                if time.time() - start > CFG["max_record_sec"]:
                    log.info("⏱  Max duration reached.")
                    break

                chunk, _ = stream.read(n_samples)
                flat      = chunk.flatten()
                raw_bytes = flat.tobytes()

                expected = n_samples * 2
                if len(raw_bytes) < expected:
                    continue

                is_speech = vad.is_speech(raw_bytes[:expected], sr)
                ring.append(is_speech)

                if is_speech:
                    speaking = True

                if speaking:
                    float_frame = flat.astype("float32") / 32768.0
                    frames.append(float_frame)
                    chunk_buf.append(float_frame)
                    chunk_sample_count += len(float_frame)

                    # ── Every ~chunk_sec seconds, transcribe chunk ──
                    if chunk_sample_count >= chunk_samples:
                        chunk_audio = np.concatenate(chunk_buf)
                        chunk_buf   = []
                        chunk_sample_count = 0

                        # Run chunk transcription in same thread (fast, beam=1)
                        chunk_text  = _transcribe_chunk(_denoise(chunk_audio))
                        chunk_words = _words_of(chunk_text)

                        # Figure out what's new
                        already = STATE.get_live_words()
                        new_w   = _new_words(chunk_words, already)

                        if new_w:
                            to_type = " ".join(new_w) + " "
                            try:
                                pyautogui.write(to_type, interval=0.01)
                            except Exception as e:
                                log.warning(f"Live-type write error: {e}")
                            STATE.add_live_words(new_w)
                            log.info(f"⚡ Live typed: {new_w}")

                if speaking and len(ring) == ring.maxlen and not any(ring):
                    break
    except Exception as e:
        log.error(f"VAD stream error: {e}")
        return np.array([], dtype="float32")

    if not frames:
        STATE.clear_live_words()
        return np.array([], dtype="float32")

    audio = np.concatenate(frames)
    return _denoise(audio)


# ══════════════════════════════════════════════════════════════════
#  INTENT DETECTION  (keyword + pattern, not naïve substring)
# ══════════════════════════════════════════════════════════════════

_DELETE_PATTERN   = re.compile(r"\b(delete|remove|erase|clear)\b")
_OPEN_PATTERN     = re.compile(r"\bopen\b")
_HOTKEY_PATTERNS  = {
    "copy":       re.compile(r"\bcopy(\s+that)?\b"),
    "paste":      re.compile(r"\bpaste(\s+that)?\b(?!\s+\w)"),
    "undo":       re.compile(r"\bundo(\s+(that|last))?\b"),
    "redo":       re.compile(r"\bredo\b"),
    "select_all": re.compile(r"\bselect\s+all\b"),
    "save":       re.compile(r"\bsave(\s+(this|file|document))?\b"),
    "find":       re.compile(r"\b(find|search)\b"),
    "bold":       re.compile(r"\bmake\s+(it\s+)?bold\b|\bbold\b"),
    "italic":     re.compile(r"\bmake\s+(it\s+)?italic\b|\bitalic\b"),
    "underline":  re.compile(r"\bunderline\b"),
    "new_line":   re.compile(r"\bnew\s+line\b|\benter\b"),
    "tab":        re.compile(r"\binsert\s+tab\b|\bpress\s+tab\b"),
    "scroll_up":  re.compile(r"\bscroll\s+up\b"),
    "scroll_down":re.compile(r"\bscroll\s+down\b"),
    "backspace":  re.compile(r"\bbackspace\b"),
    "zoom_in":    re.compile(r"\bzoom\s+in\b"),
    "zoom_out":   re.compile(r"\bzoom\s+out\b"),
}

# Selection patterns
_SELECTION_PATTERNS = {
    "select_word":      re.compile(r"\bselect\s+(the\s+)?word\b"),
    "select_sentence":  re.compile(r"\bselect\s+(the\s+)?sentence\b"),
    "select_paragraph": re.compile(r"\bselect\s+(the\s+)?paragraph\b"),
    "select_line":      re.compile(r"\bselect\s+(the\s+)?line\b"),
    "select_all":       re.compile(r"\bselect\s+all\b"),
}

_PRESENTATION_PATTERNS = {
    "next":       re.compile(r"\bnext(\s+slide)?\b"),
    "prev":       re.compile(r"\b(previous|back|prev)(\s+slide)?\b"),
    "first":      re.compile(r"\bfirst\s+slide\b|\bgo\s+to\s+start\b"),
    "last":       re.compile(r"\blast\s+slide\b|\bgo\s+to\s+end\b"),
    "start_pres": re.compile(r"\bstart\s+(the\s+)?presentation\b|\bbegin\b"),
    "end_pres":   re.compile(r"\bend\s+(the\s+)?presentation\b|\bstop\s+slides\b"),
    "black":      re.compile(r"\bblack\s+screen\b"),
    "white":      re.compile(r"\bwhite\s+screen\b"),
    "zoom_in":    re.compile(r"\bzoom\s+in\b"),
    "zoom_out":   re.compile(r"\bzoom\s+out\b"),
    "add_slide_after":    re.compile(r"\badd\s+(a\s+)?slide\s+after\b|\bnew\s+slide\s+after\b"),
    "add_slide_begin":    re.compile(r"\badd\s+(a\s+)?slide\s+at\s+(the\s+)?beginning\b|\badd\s+(a\s+)?slide\s+at\s+(the\s+)?start\b"),
    "add_slide_end":      re.compile(r"\badd\s+(a\s+)?slide\s+at\s+(the\s+)?end\b|\badd\s+(a\s+)?slide\s+(at\s+)?last\b"),
    "add_slide_now":      re.compile(r"\badd\s+(a\s+)?slide\b|\bnew\s+slide\b"),
}

# Drawing patterns
_DRAW_PATTERN        = re.compile(r"\bdraw\b|\bsketch\b|\binsert\s+shape\b")
_DRAW_SHAPE_PATTERNS = {
    "circle":               re.compile(r"\bcircle\b|\bowl\b"),
    "equilateral_triangle": re.compile(r"\bequilateral\s+triangle\b"),
    "isosceles_triangle":   re.compile(r"\bisosceles\s+triangle\b"),
    "right_triangle":       re.compile(r"\bright\s+(angle\s+)?triangle\b|\bright-angle\s+triangle\b"),
    "triangle":             re.compile(r"\btriangle\b"),
    "rectangle":            re.compile(r"\brectangle\b|\bRect\b"),
    "square":               re.compile(r"\bsquare\b"),
    "ellipse":              re.compile(r"\bellipse\b|\boval\b"),
    "pentagon":             re.compile(r"\bpentagon\b"),
    "hexagon":              re.compile(r"\bhexagon\b"),
    "line":                 re.compile(r"\bstraight\s+line\b|\bdraw\s+line\b"),
    "arrow":                re.compile(r"\barrow\b"),
}

# AI Search patterns
_AI_SEARCH_PATTERN = re.compile(
    r"\bsearch\s+(?P<query>.+?)\s+on\s+(?P<engine>gemini|chatgpt|chat\s*gpt|copilot|bing)\b"
    r"|\bask\s+(?P<engine2>gemini|chatgpt|copilot)\s+(?P<query2>.+)",
    re.IGNORECASE
)

_NUMBER_MAP = {
    "one":1,"two":2,"three":3,"four":4,"five":5,
    "six":6,"seven":7,"eight":8,"nine":9,"ten":10,
    "twenty":20,"thirty":30,"fifty":50,
    "hundred":100,
}

def extract_number(text: str, default=1) -> int:
    for w in text.split():
        w = w.rstrip(".,")
        if w.isdigit():      return int(w)
        if w in _NUMBER_MAP: return _NUMBER_MAP[w]
    return default

def extract_float(text: str, default=100.0) -> float:
    """Extract the first float/int from text (e.g. '150 pixels' → 150.0)."""
    m = re.search(r"(\d+(?:\.\d+)?)", text)
    return float(m.group(1)) if m else default


# ══════════════════════════════════════════════════════════════════
#  SMART TEXT FORMATTER
# ══════════════════════════════════════════════════════════════════

_SPOKEN_PUNCT = [
    (r"\bcomma\b",           ","),
    (r"\bfull\s+stop\b",     "."),
    (r"\bperiod\b",          "."),
    (r"\bquestion\s+mark\b", "?"),
    (r"\bexclamation\b",     "!"),
    (r"\bcolon\b",           ":"),
    (r"\bsemicolon\b",       ";"),
    (r"\bdash\b",            " -"),
    (r"\bhyphen\b",          "-"),
    (r"\bopen\s+bracket\b",  "("),
    (r"\bclose\s+bracket\b", ")"),
    (r"\bopen\s+quote\b",    '"'),
    (r"\bclose\s+quote\b",   '"'),
]
_SPOKEN_PUNCT_COMPILED = [(re.compile(p), s) for p, s in _SPOKEN_PUNCT]

_PARA_BREAK   = re.compile(r"\bnew\s+paragraph\b|\bnext\s+paragraph\b|\bparagraph\s+break\b")
_BULLET_START = {"first","second","third","fourth","fifth","next","also","then","lastly","additionally"}

def _is_heading(words: list[str]) -> bool:
    if not (2 <= len(words) <= 5):
        return False
    if words[0] in _BULLET_START:
        return False
    return True

def smart_format(text: str) -> str:
    text = text.strip().lower()
    if not text:
        return ""
    if _PARA_BREAK.search(text):
        return "\n\n"
    for pattern, symbol in _SPOKEN_PUNCT_COMPILED:
        text = pattern.sub(symbol, text)
    if PUNCT_AVAILABLE:
        try:
            text = _punct_instance.restore_punctuation(text)
        except Exception:
            pass
    text = text.strip()
    words = text.split()
    if not words:
        return ""
    if _is_heading(words):
        return "\n" + text.upper() + "\n\n"
    if words[0].lower() in _BULLET_START:
        body = " ".join(words[1:])
        return "• " + body[:1].upper() + body[1:] + "\n"
    return text[:1].upper() + text[1:] + " "


# ══════════════════════════════════════════════════════════════════
#  APPLICATION & WEB CONTROL
# ══════════════════════════════════════════════════════════════════

SITE_MAP = {
    "youtube":   "https://youtube.com",
    "google":    "https://google.com",
    "github":    "https://github.com",
    "gmail":     "https://mail.google.com",
    "chatgpt":   "https://chat.openai.com",
    "maps":      "https://maps.google.com",
    "wikipedia": "https://wikipedia.org",
    "whatsapp":  "https://web.whatsapp.com",
    "instagram": "https://instagram.com",
    "twitter":   "https://twitter.com",
    "facebook":  "https://facebook.com",
    "linkedin":  "https://linkedin.com",
    "reddit":    "https://reddit.com",
    "netflix":   "https://netflix.com",
    "spotify":   "https://open.spotify.com",
    "gemini":    "https://gemini.google.com",
    "copilot":   "https://copilot.microsoft.com",
}

APP_MAP = {
    "chrome":      ["google-chrome", "chromium",
                    r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
    "notepad":     ["gedit", "mousepad", "notepad.exe"],
    "paint":       ["pinta", "kolourpaint", "mspaint.exe"],
    "vscode":      ["code",
                    r"C:\Users\%USERNAME%\AppData\Local\Programs\Microsoft VS Code\Code.exe"],
    "terminal":    ["gnome-terminal", "konsole", "xterm", "cmd.exe"],
    "explorer":    ["nautilus", "thunar", "explorer.exe"],
    "calculator":  ["gnome-calculator", "kcalc", "calc.exe"],
    "zoom":        ["zoom", r"C:\Users\%USERNAME%\AppData\Roaming\Zoom\bin\Zoom.exe"],
    "whatsapp":    ["whatsapp-desktop",
                    r"C:\Users\%USERNAME%\AppData\Local\WhatsApp\WhatsApp.exe"],
    "slack":       ["slack", r"C:\Users\%USERNAME%\AppData\Local\slack\slack.exe"],
    "discord":     ["discord", r"C:\Users\%USERNAME%\AppData\Local\Discord\app-*\Discord.exe"],
    "teams":       ["teams",
                    r"C:\Users\%USERNAME%\AppData\Local\Microsoft\Teams\current\Teams.exe"],
    "outlook":     [r"C:\Program Files\Microsoft Office\root\Office16\OUTLOOK.EXE",
                    r"C:\Program Files (x86)\Microsoft Office\root\Office16\OUTLOOK.EXE"],
    "word":        [r"C:\Program Files\Microsoft Office\root\Office16\WINWORD.EXE"],
    "excel":       [r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"],
    "powerpoint":  [r"C:\Program Files\Microsoft Office\root\Office16\POWERPNT.EXE"],
    "vlc":         ["vlc", r"C:\Program Files\VideoLAN\VLC\vlc.exe"],
    "spotify":     ["spotify", r"C:\Users\%USERNAME%\AppData\Roaming\Spotify\Spotify.exe"],
    "settings":    ["gnome-control-center", "systemsettings5", "ms-settings:"],
    "control panel": ["control", r"C:\Windows\System32\control.exe"],
    "task manager": ["gnome-system-monitor", "ksysguard",
                     r"C:\Windows\System32\Taskmgr.exe"],
    "file manager": ["nautilus", "thunar", "dolphin", "explorer.exe"],
}

def _launch_app(name: str) -> bool:
    paths = APP_MAP.get(name, [])
    for p in paths:
        p = os.path.expandvars(p)
        if p.startswith("ms-settings:"):
            try:
                os.startfile(p)
                return True
            except Exception:
                pass
        if os.path.isfile(p):
            try:
                if sys.platform == "win32":
                    os.startfile(p)
                else:
                    subprocess.Popen([p], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
        if shutil.which(p):
            subprocess.Popen([p], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
    return False


def _dynamic_open_app(name: str) -> bool:
    if name in APP_MAP:
        if _launch_app(name):
            return True
    if shutil.which(name):
        try:
            subprocess.Popen([name], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return True
        except Exception:
            pass
    if sys.platform == "win32":
        try:
            subprocess.Popen(
                ["powershell", "-Command", f"Start-Process '{name}'"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return True
        except Exception:
            pass
    if sys.platform.startswith("linux"):
        desktop_dirs = [
            Path.home() / ".local/share/applications",
            Path("/usr/share/applications"),
            Path("/usr/local/share/applications"),
        ]
        for d in desktop_dirs:
            if d.exists():
                for f in d.glob("*.desktop"):
                    if name.lower() in f.stem.lower():
                        try:
                            subprocess.Popen(["gtk-launch", f.stem],
                                             stdout=subprocess.DEVNULL,
                                             stderr=subprocess.DEVNULL)
                            return True
                        except Exception:
                            pass
    log.warning(f"Could not find/launch: {name}")
    return False


# ══════════════════════════════════════════════════════════════════
#  FEATURE 3: CONTEXT-BASED FILE SEARCH
# ══════════════════════════════════════════════════════════════════

# Stop-words to strip from search query when extracting keywords
_FILE_STOP_WORDS = {
    "open", "the", "a", "an", "in", "on", "at", "of", "and", "or",
    "file", "folder", "my", "me", "with", "for", "to", "please",
}

# Priority order for file type preference (lower index = higher priority)
_FILE_TYPE_PRIORITY = [".pptx", ".ppt", ".pdf", ".docx", ".doc",
                       ".xlsx", ".xls", ".txt", ".md"]

# Preposition tokens that signal a folder context
_FOLDER_SIGNALS = {"in", "from", "inside", "under", "within", "at"}


def _extract_file_keywords(text: str) -> tuple[list[str], list[str]]:
    """
    FEATURE 3: Parse 'open <file keywords> in <folder keywords>' style commands.

    Returns (file_keywords, folder_keywords).
    Both lists are lowercase tokens with stop-words removed.
    """
    text = text.lower()
    # Remove "open" trigger word
    text = re.sub(r"^open\s+", "", text).strip()

    # Split on folder-signal prepositions: "unit 4 in control system folder"
    folder_kws: list[str] = []
    file_kws:   list[str] = []

    # Try to find " in <something>" or " from <something>" etc.
    folder_split = None
    for signal in _FOLDER_SIGNALS:
        pattern = re.compile(rf"\s+{signal}\s+(.+)", re.IGNORECASE)
        m = pattern.search(text)
        if m:
            folder_part = m.group(1)
            file_part   = text[: m.start()].strip()
            # Strip trailing "folder" keyword
            folder_part = re.sub(r"\s+folder$", "", folder_part).strip()
            folder_kws  = [w for w in folder_part.split()
                           if w not in _FILE_STOP_WORDS and len(w) > 1]
            file_kws    = [w for w in file_part.split()
                           if w not in _FILE_STOP_WORDS and len(w) > 1]
            folder_split = True
            break

    if not folder_split:
        # No preposition — treat entire remainder as file keywords
        file_kws = [w for w in text.split()
                    if w not in _FILE_STOP_WORDS and len(w) > 1]

    return file_kws, folder_kws


def _score_path(path: Path, file_kws: list[str], folder_kws: list[str]) -> int:
    """
    FEATURE 3: Score a path candidate.
    Higher = better match.
    """
    score = 0
    name_lower = path.stem.lower()
    parts_lower = [p.lower() for p in path.parts]

    for kw in file_kws:
        if kw in name_lower:
            score += 2
    for kw in folder_kws:
        if any(kw in part for part in parts_lower):
            score += 1

    # Bonus for preferred file type
    try:
        prio = _FILE_TYPE_PRIORITY.index(path.suffix.lower())
        score += max(0, len(_FILE_TYPE_PRIORITY) - prio)
    except ValueError:
        pass

    return score


def _search_and_open_file(text: str) -> bool:
    """
    FEATURE 3: Recursively search for a file matching keywords extracted
    from 'text'.  Returns True if a file was found and opened.
    """
    file_kws, folder_kws = _extract_file_keywords(text)
    if not file_kws:
        return False

    log.info(f"📂 File search — file_kws={file_kws}  folder_kws={folder_kws}")

    # Search roots: common user dirs first, then home, then drive roots
    home = Path.home()
    search_roots: list[Path] = [
        home / "Desktop",
        home / "Documents",
        home / "Downloads",
        home,
    ]
    if sys.platform == "win32":
        # Add all drive roots
        import string
        for letter in string.ascii_uppercase:
            drive = Path(f"{letter}:\\")
            if drive.exists():
                search_roots.append(drive)
    else:
        search_roots.append(Path("/"))

    best_path  = None
    best_score = 0

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for candidate in root.rglob("*"):
                if not candidate.is_file():
                    continue
                # Quick pre-filter: at least one file keyword in name
                name_lower = candidate.stem.lower()
                if not any(kw in name_lower for kw in file_kws):
                    continue
                score = _score_path(candidate, file_kws, folder_kws)
                if score > best_score:
                    best_score = score
                    best_path  = candidate
        except PermissionError:
            continue
        except Exception as e:
            log.warning(f"File search error in {root}: {e}")
            continue

        # Stop after finding a good enough match in user dirs
        if best_path and best_score >= 3:
            break

    if best_path:
        log.info(f"📂 Opening file: {best_path}  (score={best_score})")
        try:
            if sys.platform == "win32":
                os.startfile(str(best_path))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(best_path)])
            else:
                subprocess.Popen(["xdg-open", str(best_path)])
            return True
        except Exception as e:
            log.error(f"Could not open file {best_path}: {e}")
            return False
    else:
        log.warning(f"📂 File not found for keywords: {file_kws}")
        return False


def handle_open(text: str) -> bool:
    # Existing: check SITE_MAP
    for site, url in SITE_MAP.items():
        if site in text:
            webbrowser.open(url)
            log.info(f"🌐 Opened {site}")
            return True

    # Existing: check APP_MAP
    for app in APP_MAP:
        if app in text:
            ok = _launch_app(app)
            log.info(f"🖥  {'Opened' if ok else 'Not found:'} {app}")
            return True

    # FEATURE 3: context-based file search
    # Triggered when "open" + non-app/site keywords suggest a file
    file_kws, folder_kws = _extract_file_keywords(text)
    if file_kws:
        if _search_and_open_file(text):
            return True

    # Existing: dynamic app extraction
    m = re.search(r"\bopen\s+(.+)", text)
    if m:
        target = m.group(1).strip().rstrip(".,")
        log.info(f"🔍 Dynamic open: '{target}'")
        ok = _dynamic_open_app(target)
        if ok:
            log.info(f"🖥  Dynamically opened: {target}")
            return True
        webbrowser.open(f"https://www.google.com/search?q={target}+download")
        log.info(f"🌐 Opened Google search for: {target}")
        return True

    return False


# ══════════════════════════════════════════════════════════════════
#  DELETE ENGINE  (original + custom-text delete)
# ══════════════════════════════════════════════════════════════════

def _find_and_delete_text(target: str) -> bool:
    if not target:
        return False
    try:
        pyautogui.hotkey("ctrl", "h")
        time.sleep(0.4)
        pyautogui.hotkey("ctrl", "a")
        pyautogui.write(target, interval=0.02)
        pyautogui.press("tab")
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("delete")
        pyautogui.hotkey("alt", "a")
        time.sleep(0.3)
        pyautogui.press("escape")
        log.info(f"🗑  Deleted text: '{target}'")
        STATE.record_history(target, "delete_custom_text")
        return True
    except Exception as e:
        log.error(f"Find-and-delete error: {e}")
        return False


def handle_delete(text: str) -> bool:
    count = extract_number(text)

    custom_match = re.search(
        r"\bdelete\s+(word|sentence|text)\s+(.+)", text, re.IGNORECASE
    )
    if custom_match:
        scope  = custom_match.group(1).lower()
        target = custom_match.group(2).strip().rstrip(".,")
        if target:
            log.info(f"🗑  Delete {scope}: '{target}'")
            return _find_and_delete_text(target)

    if "everything" in text or ("page" in text and "open" not in text):
        ans = pyautogui.confirm(
            text="This will delete all text. Are you sure?",
            title="Smart Pen — Safety Check",
            buttons=["Cancel", "Yes, Delete All"]
        )
        if ans != "Yes, Delete All":
            log.info("Delete all cancelled by user.")
            return True
        pyautogui.hotkey("ctrl", "a")
        pyautogui.press("backspace")
        STATE.record_history(text, "delete_all")
        return True

    if "paragraph" in text:
        for _ in range(count):
            pyautogui.hotkey("ctrl", "shift", "up")
        pyautogui.press("backspace")
        STATE.record_history(text, f"delete_paragraph×{count}")
        return True

    if "sentence" in text:
        for _ in range(count):
            pyautogui.hotkey("ctrl", "shift", "left")
        pyautogui.press("backspace")
        STATE.record_history(text, f"delete_sentence×{count}")
        return True

    if "line" in text:
        for _ in range(count):
            pyautogui.hotkey("shift", "home")
            pyautogui.press("backspace")
        STATE.record_history(text, f"delete_line×{count}")
        return True

    if "word" in text:
        for _ in range(count):
            pyautogui.hotkey("ctrl", "backspace")
        STATE.record_history(text, f"delete_word×{count}")
        return True

    for _ in range(count):
        pyautogui.press("backspace")
    STATE.record_history(text, f"backspace×{count}")
    return True


# ══════════════════════════════════════════════════════════════════
#  SELECTION COMMANDS
# ══════════════════════════════════════════════════════════════════

def handle_selection(key: str, text: str) -> bool:
    count = extract_number(text)

    if key == "select_all":
        pyautogui.hotkey("ctrl", "a")
        STATE.record_history(text, "select_all")
        return True
    if key == "select_word":
        pyautogui.hotkey("ctrl", "shift", "right")
        STATE.record_history(text, "select_word")
        return True
    if key == "select_line":
        pyautogui.press("home")
        pyautogui.hotkey("shift", "end")
        STATE.record_history(text, "select_line")
        return True
    if key == "select_sentence":
        pyautogui.press("home")
        pyautogui.hotkey("shift", "end")
        STATE.record_history(text, "select_sentence")
        return True
    if key == "select_paragraph":
        for _ in range(count):
            pyautogui.hotkey("ctrl", "shift", "down")
        STATE.record_history(text, f"select_paragraph×{count}")
        return True
    return False


# ══════════════════════════════════════════════════════════════════
#  FEATURE 4: UNIT-AWARE PIXEL CONVERSION
# ══════════════════════════════════════════════════════════════════

# Conversion table: unit name → pixels
_UNIT_PX: dict[str, float] = {
    # metric
    "mm":     1.0,
    "millimeter": 1.0,
    "millimeters": 1.0,
    "cm":     10.0,
    "centimeter": 10.0,
    "centimeters": 10.0,
    "m":      1000.0,
    "meter":  1000.0,
    "meters": 1000.0,
    # imperial
    "in":     25.0,
    "inch":   25.0,
    "inches": 25.0,
    "\"":     25.0,
    "ft":     300.0,
    "foot":   300.0,
    "feet":   300.0,
    "'":      300.0,
    # default (no unit)
    "px":     1.0,
    "pixel":  1.0,
    "pixels": 1.0,
}

# Pattern: optional decimal number + optional whitespace + unit keyword
_MEASURE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*"
    r"(millimeters?|centimeters?|meters?|inches?|feet|foot|mm|cm|ft|in|px|pixels?|\"|\')?\b",
    re.IGNORECASE,
)


def _to_px(value: float, unit: str) -> int:
    """Convert a numeric value with its unit string to pixels."""
    unit_clean = (unit or "").strip().lower().rstrip("s")  # plurals
    # special: "'" → foot, '"' → inch
    if unit_clean == "'":  unit_clean = "foot"
    if unit_clean == '"':  unit_clean = "inch"
    factor = _UNIT_PX.get(unit_clean, _UNIT_PX.get(unit_clean + "s", 1.0))
    return max(1, int(round(value * factor)))


def _extract_measurements(text: str) -> list[int]:
    """
    FEATURE 4: Extract all measurement values (with optional units) from
    the spoken command and return them as pixel integers.

    Handles:
      "3 cm and 4 cm"           → [30, 40]
      "5 inches"                → [125]
      "2 feet by 1 foot"        → [600, 300]
      "radius 100"              → [100]   (no unit → pixels)
      "1 foot 6 inches"         → [450]   (adds up: 300 + 150 = 450)
    """
    # Strip non-numeric noise around "by", "and", "x"
    cleaned = re.sub(r"\b(by|and|x)\b", " ", text, flags=re.IGNORECASE)

    # Find all (value, unit) pairs
    matches = _MEASURE_RE.findall(cleaned)

    px_values: list[int] = []
    i = 0
    while i < len(matches):
        val_str, unit = matches[i]
        if not val_str:
            i += 1
            continue
        val = float(val_str)

        # Check if the NEXT token is a unit-only continuation (e.g., "1 foot 6 inches")
        # This handles compound imperial measurements
        if (not unit or unit.lower() in ("foot", "feet", "ft", "'")) \
                and i + 1 < len(matches):
            next_val_str, next_unit = matches[i + 1]
            if next_val_str and next_unit and \
                    next_unit.lower() in ("inch", "inches", "in", '"'):
                # Compound: add foot + inch in pixels
                px_values.append(_to_px(val, unit) + _to_px(float(next_val_str), next_unit))
                i += 2
                continue

        px_values.append(_to_px(val, unit))
        i += 1

    return px_values


def _parse_draw_params(text: str, shape: str) -> dict:
    """
    FEATURE 4: Extract numeric parameters for each shape from the spoken
    command, with full unit conversion (cm, mm, inches, feet, meters → px).
    Falls back to sensible pixel defaults if no measurement is found.
    """
    params: dict = {}
    measurements = _extract_measurements(text)

    def _m(idx: int, default: int) -> int:
        return measurements[idx] if idx < len(measurements) else default

    if shape == "circle":
        params["radius"] = _m(0, 100)

    elif shape == "equilateral_triangle":
        params["side"] = _m(0, 180)

    elif shape in ("isosceles_triangle", "right_triangle"):
        params["base"]   = _m(0, 180)
        params["height"] = _m(1, 160)

    elif shape == "square":
        params["side"] = _m(0, 150)

    elif shape == "rectangle":
        params["width"]  = _m(0, 200)
        params["height"] = _m(1, 120)

    elif shape == "ellipse":
        params["rx"] = _m(0, 150)
        params["ry"] = _m(1, 90)

    elif shape in ("pentagon", "hexagon"):
        params["radius"] = _m(0, 120)

    elif shape in ("line", "arrow"):
        params["length"] = _m(0, 200)

    return params


# ══════════════════════════════════════════════════════════════════
#  DRAWING MATHEMATICAL SHAPES  (Tkinter canvas — non-slideshow)
# ══════════════════════════════════════════════════════════════════

def _draw_shape_window(shape: str, params: dict):
    """
    Open a Tkinter window and draw the requested shape.
    Runs in a separate thread to avoid blocking the main loop.
    Used when NOT in slideshow mode.
    """
    if not TKINTER_AVAILABLE:
        log.warning("Tkinter not available — cannot draw shapes.")
        return

    W = CFG["canvas_width"]
    H = CFG["canvas_height"]
    cx, cy = W // 2, H // 2

    def _run():
        root = tk.Tk()
        root.title(f"Smart Pen — {shape.replace('_', ' ').title()}")
        canvas = tk.Canvas(root, width=W, height=H, bg="white")
        canvas.pack()

        if shape == "circle":
            r = params.get("radius", 100)
            canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                               outline="black", width=2)
            canvas.create_text(cx, cy + r + 15,
                               text=f"Circle  r = {r}px", fill="gray")

        elif shape == "square":
            s = params.get("side", 150)
            half = s / 2
            canvas.create_rectangle(cx - half, cy - half,
                                    cx + half, cy + half,
                                    outline="black", width=2)
            canvas.create_text(cx, cy + half + 15,
                               text=f"Square  side = {s}px", fill="gray")

        elif shape == "rectangle":
            w = params.get("width",  200)
            h = params.get("height", 120)
            canvas.create_rectangle(cx - w/2, cy - h/2,
                                    cx + w/2, cy + h/2,
                                    outline="black", width=2)
            canvas.create_text(cx, cy + h/2 + 15,
                               text=f"Rectangle  {w}×{h}px", fill="gray")

        elif shape == "equilateral_triangle":
            s  = params.get("side", 180)
            h  = s * math.sqrt(3) / 2
            pts = [
                (cx,        cy - 2*h/3),
                (cx - s/2,  cy + h/3),
                (cx + s/2,  cy + h/3),
            ]
            canvas.create_polygon(pts, outline="black", fill="", width=2)
            canvas.create_text(cx, cy + h/3 + 18,
                               text=f"Equilateral  side = {s}px", fill="gray")

        elif shape == "isosceles_triangle":
            base   = params.get("base",  180)
            height = params.get("height", 160)
            pts = [
                (cx,            cy - 2*height/3),
                (cx - base/2,   cy + height/3),
                (cx + base/2,   cy + height/3),
            ]
            canvas.create_polygon(pts, outline="black", fill="", width=2)
            canvas.create_text(cx, cy + height/3 + 18,
                               text=f"Isosceles  base={base}  h={height}px",
                               fill="gray")

        elif shape == "right_triangle":
            base   = params.get("base",  180)
            height = params.get("height", 160)
            pts = [
                (cx - base/2, cy + height/3),
                (cx - base/2, cy - 2*height/3),
                (cx + base/2, cy + height/3),
            ]
            canvas.create_polygon(pts, outline="black", fill="", width=2)
            s2 = 14
            bx, by = cx - base/2, cy + height/3
            canvas.create_line(bx, by - s2, bx + s2, by - s2, fill="black")
            canvas.create_line(bx + s2, by, bx + s2, by - s2, fill="black")
            canvas.create_text(cx, cy + height/3 + 18,
                               text=f"Right Triangle  base={base}  h={height}px",
                               fill="gray")

        elif shape == "triangle":
            _draw_shape_window("equilateral_triangle", params)
            root.destroy()
            return

        elif shape == "ellipse":
            rx = params.get("rx", 150)
            ry = params.get("ry", 90)
            canvas.create_oval(cx - rx, cy - ry, cx + rx, cy + ry,
                               outline="black", width=2)
            canvas.create_text(cx, cy + ry + 15,
                               text=f"Ellipse  rx={rx}  ry={ry}px", fill="gray")

        elif shape in ("pentagon", "hexagon"):
            n_sides = 5 if shape == "pentagon" else 6
            r       = params.get("radius", 120)
            pts = []
            for i in range(n_sides):
                angle = math.radians(90 + 360 * i / n_sides)
                pts.append((cx + r * math.cos(angle),
                             cy - r * math.sin(angle)))
            canvas.create_polygon(pts, outline="black", fill="", width=2)
            canvas.create_text(cx, cy + r + 15,
                               text=f"{shape.title()}  r={r}px", fill="gray")

        elif shape == "line":
            length = params.get("length", 200)
            canvas.create_line(cx - length/2, cy, cx + length/2, cy,
                               fill="black", width=2)
            canvas.create_text(cx, cy + 18,
                               text=f"Line  {length}px", fill="gray")

        elif shape == "arrow":
            length = params.get("length", 200)
            canvas.create_line(cx - length/2, cy, cx + length/2, cy,
                               arrow=tk.LAST, fill="black", width=2)
            canvas.create_text(cx, cy + 18,
                               text=f"Arrow  {length}px", fill="gray")

        def save_image():
            try:
                from PIL import ImageGrab
                path = Path(f"smart_pen_shape_{shape}_{int(time.time())}.png")
                x  = root.winfo_rootx() + canvas.winfo_x()
                y  = root.winfo_rooty() + canvas.winfo_y()
                x1 = x + canvas.winfo_width()
                y1 = y + canvas.winfo_height()
                ImageGrab.grab(bbox=(x, y, x1, y1)).save(path)
                log.info(f"💾 Shape saved: {path}")
            except Exception as ex:
                log.warning(f"Save failed: {ex}")

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=6)
        tk.Button(btn_frame, text="💾 Save PNG", command=save_image).pack(side=tk.LEFT, padx=8)
        tk.Button(btn_frame, text="Close",       command=root.destroy).pack(side=tk.LEFT, padx=8)

        root.mainloop()

    t = threading.Thread(target=_run, daemon=True, name="DrawThread")
    t.start()


# ══════════════════════════════════════════════════════════════════
#  FEATURE 5: IN-SLIDE DRAWING (PowerPoint ink mode via pyautogui)
# ══════════════════════════════════════════════════════════════════

def _get_screen_center() -> tuple[int, int]:
    """Return the pixel coordinates of the screen centre."""
    w = pyautogui.size().width
    h = pyautogui.size().height
    return w // 2, h // 2


def _smooth_move(x1: int, y1: int, x2: int, y2: int,
                 steps: int = 40, duration: float = 0.4):
    """Move mouse from (x1,y1) to (x2,y2) while holding the button down."""
    pyautogui.moveTo(x1, y1, duration=0.05)
    pyautogui.mouseDown()
    for i in range(1, steps + 1):
        t = i / steps
        ix = int(x1 + (x2 - x1) * t)
        iy = int(y1 + (y2 - y1) * t)
        pyautogui.moveTo(ix, iy, duration=duration / steps)
    pyautogui.mouseUp()


def _activate_pen_mode():
    """
    FEATURE 5: Enter PowerPoint / LibreOffice Impress pen mode.
    Ctrl+P is the standard shortcut.
    """
    pyautogui.hotkey("ctrl", "p")
    time.sleep(0.3)


def _deactivate_pen_mode():
    """Exit pen/annotation mode — press Escape once to return to slide view."""
    pyautogui.press("escape")
    time.sleep(0.2)


def _draw_on_slide(shape: str, params: dict):
    """
    FEATURE 5: Draw a shape directly on the active PowerPoint slide using
    mouse movements while the presentation pen mode is active.
    Runs in a daemon thread to avoid blocking the command queue.
    """
    cx, cy = _get_screen_center()

    def _run():
        _activate_pen_mode()

        if shape == "circle":
            r     = params.get("radius", 80)
            steps = 60
            # Move to start point first
            sx = cx + r
            sy = cy
            pyautogui.moveTo(sx, sy, duration=0.1)
            pyautogui.mouseDown()
            for i in range(steps + 1):
                angle = math.radians(360 * i / steps)
                px = cx + int(r * math.cos(angle))
                py = cy + int(r * math.sin(angle))
                pyautogui.moveTo(px, py, duration=0.4 / steps)
            pyautogui.mouseUp()

        elif shape in ("equilateral_triangle", "triangle"):
            s  = params.get("side", 160)
            h  = int(s * math.sqrt(3) / 2)
            pts = [
                (cx,         cy - 2*h//3),
                (cx - s//2,  cy + h//3),
                (cx + s//2,  cy + h//3),
                (cx,         cy - 2*h//3),   # close
            ]
            pyautogui.moveTo(pts[0][0], pts[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in pts[1:]:
                pyautogui.moveTo(px, py, duration=0.3)
            pyautogui.mouseUp()

        elif shape == "right_triangle":
            base   = params.get("base",   160)
            height = params.get("height", 140)
            # right angle at bottom-left
            pts = [
                (cx - base//2, cy + height//3),
                (cx - base//2, cy - 2*height//3),
                (cx + base//2, cy + height//3),
                (cx - base//2, cy + height//3),   # close
            ]
            pyautogui.moveTo(pts[0][0], pts[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in pts[1:]:
                pyautogui.moveTo(px, py, duration=0.3)
            pyautogui.mouseUp()

        elif shape == "isosceles_triangle":
            base   = params.get("base",   160)
            height = params.get("height", 140)
            pts = [
                (cx,            cy - 2*height//3),
                (cx - base//2,  cy + height//3),
                (cx + base//2,  cy + height//3),
                (cx,            cy - 2*height//3),   # close
            ]
            pyautogui.moveTo(pts[0][0], pts[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in pts[1:]:
                pyautogui.moveTo(px, py, duration=0.3)
            pyautogui.mouseUp()

        elif shape == "rectangle":
            w = params.get("width",  180)
            h = params.get("height", 110)
            corners = [
                (cx - w//2, cy - h//2),
                (cx + w//2, cy - h//2),
                (cx + w//2, cy + h//2),
                (cx - w//2, cy + h//2),
                (cx - w//2, cy - h//2),   # close
            ]
            pyautogui.moveTo(corners[0][0], corners[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in corners[1:]:
                pyautogui.moveTo(px, py, duration=0.25)
            pyautogui.mouseUp()

        elif shape == "square":
            s = params.get("side", 150)
            corners = [
                (cx - s//2, cy - s//2),
                (cx + s//2, cy - s//2),
                (cx + s//2, cy + s//2),
                (cx - s//2, cy + s//2),
                (cx - s//2, cy - s//2),
            ]
            pyautogui.moveTo(corners[0][0], corners[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in corners[1:]:
                pyautogui.moveTo(px, py, duration=0.25)
            pyautogui.mouseUp()

        elif shape == "line":
            length = params.get("length", 200)
            _smooth_move(cx - length//2, cy, cx + length//2, cy)

        elif shape == "arrow":
            length  = params.get("length", 200)
            arrow_h = 18   # arrowhead size
            # Draw shaft
            _smooth_move(cx - length//2, cy, cx + length//2, cy)
            # Arrowhead — two short lines
            _smooth_move(cx + length//2, cy,
                         cx + length//2 - arrow_h, cy - arrow_h)
            _smooth_move(cx + length//2, cy,
                         cx + length//2 - arrow_h, cy + arrow_h)

        elif shape == "ellipse":
            rx = params.get("rx", 130)
            ry = params.get("ry", 80)
            steps = 60
            sx = cx + rx
            sy = cy
            pyautogui.moveTo(sx, sy, duration=0.1)
            pyautogui.mouseDown()
            for i in range(steps + 1):
                angle = math.radians(360 * i / steps)
                px = cx + int(rx * math.cos(angle))
                py = cy + int(ry * math.sin(angle))
                pyautogui.moveTo(px, py, duration=0.4 / steps)
            pyautogui.mouseUp()

        elif shape in ("pentagon", "hexagon"):
            n_sides = 5 if shape == "pentagon" else 6
            r       = params.get("radius", 110)
            pts = []
            for i in range(n_sides + 1):
                angle = math.radians(90 + 360 * i / n_sides)
                pts.append((cx + int(r * math.cos(angle)),
                             cy - int(r * math.sin(angle))))
            pyautogui.moveTo(pts[0][0], pts[0][1], duration=0.1)
            pyautogui.mouseDown()
            for px, py in pts[1:]:
                pyautogui.moveTo(px, py, duration=0.3)
            pyautogui.mouseUp()

        _deactivate_pen_mode()
        log.info(f"🎨 In-slide drawing complete: {shape}")

    t = threading.Thread(target=_run, daemon=True, name="SlideDrawThread")
    t.start()


def handle_draw(text: str) -> bool:
    for shape, pattern in _DRAW_SHAPE_PATTERNS.items():
        if pattern.search(text):
            params = _parse_draw_params(text, shape)
            log.info(f"🎨 Drawing {shape} with params {params}")

            # FEATURE 5: draw on slide if slideshow is active
            if STATE.in_slideshow:
                log.info("🖊  Drawing directly on slide (pen mode).")
                _draw_on_slide(shape, params)
            else:
                # Existing: draw in Tkinter window
                _draw_shape_window(shape, params)

            STATE.record_history(text, f"draw_{shape}")
            return True
    log.warning("Draw command detected but no shape matched.")
    return False


# ══════════════════════════════════════════════════════════════════
#  FEATURE 2: AI SEARCH COMMAND (with default ChatGPT fallback)
# ══════════════════════════════════════════════════════════════════

_AI_SEARCH_URLS = {
    "gemini":   "https://gemini.google.com/app?q={query}",
    "chatgpt":  "https://chat.openai.com/?q={query}",
    "chat gpt": "https://chat.openai.com/?q={query}",
    "copilot":  "https://copilot.microsoft.com/?q={query}",
    "bing":     "https://copilot.microsoft.com/?q={query}",
}

# FEATURE 2: default engine when no platform is specified
_AI_SEARCH_DEFAULT_ENGINE = "chatgpt"


def handle_ai_search(text: str) -> bool:
    """
    Parse 'search <query> on <engine>' or 'ask <engine> <query>'
    and open the appropriate AI assistant with the query pre-filled.

    FEATURE 2: If no platform is specified (plain 'search <query>'),
    falls back to ChatGPT automatically.

    Works in both typing mode and presentation mode (including slideshow).
    """
    import urllib.parse

    engine = None
    query  = None

    # Pattern 1: "search <query> on <engine>"
    m = re.search(
        r"search\s+(.+?)\s+on\s+(gemini|chatgpt|chat\s*gpt|copilot|bing)",
        text, re.IGNORECASE
    )
    if m:
        query  = m.group(1).strip()
        engine = m.group(2).strip().lower()

    # Pattern 2: "ask <engine> <query>"
    if engine is None:
        m = re.search(
            r"ask\s+(gemini|chatgpt|chat\s*gpt|copilot|bing)\s+(.+)",
            text, re.IGNORECASE
        )
        if m:
            engine = m.group(1).strip().lower()
            query  = m.group(2).strip()

    # FEATURE 2 — Pattern 3: plain "search <query>" with no platform
    if engine is None:
        m = re.search(r"\bsearch\s+(.+)", text, re.IGNORECASE)
        if m:
            query  = m.group(1).strip()
            engine = _AI_SEARCH_DEFAULT_ENGINE
            log.info(f"🤖 No platform specified — defaulting to {engine}.")

    if engine is None or query is None:
        return False

    engine = engine.replace(" ", "")   # "chat gpt" → "chatgpt"

    url_tpl = _AI_SEARCH_URLS.get(engine)
    if not url_tpl:
        log.warning(f"Unknown AI engine: {engine}")
        return False

    encoded = urllib.parse.quote(query)
    url     = url_tpl.replace("{query}", encoded)

    if STATE.in_slideshow and WINDOW_AVAILABLE:
        _search_from_slideshow(url)
    else:
        webbrowser.open(url)

    log.info(f"🤖 AI search [{engine}]: '{query}'")
    STATE.record_history(text, f"ai_search_{engine}")
    return True


# ══════════════════════════════════════════════════════════════════
#  SLIDESHOW SEARCH  (temp window switch + return)
# ══════════════════════════════════════════════════════════════════

def _search_from_slideshow(url: str):
    log.info("📊→🌐 Exiting slideshow to search…")

    pres_win = None
    if WINDOW_AVAILABLE:
        try:
            active_wins = [w for w in gw.getAllWindows() if w.isActive]
            if active_wins:
                pres_win = active_wins[0]
        except Exception:
            pass

    pyautogui.press("escape")
    time.sleep(0.5)
    webbrowser.open(url)
    log.info("🌐 Browser opened. Presentation paused.")

    def _resume():
        _slideshow_resume_event.clear()
        _slideshow_resume_event.wait(timeout=120)
        if pres_win is not None and WINDOW_AVAILABLE:
            try:
                pres_win.activate()
                time.sleep(0.4)
            except Exception:
                pass
        pyautogui.press("f5")
        STATE.in_slideshow = True
        log.info("📊 Slideshow resumed.")

    t = threading.Thread(target=_resume, daemon=True, name="SlideshowResume")
    t.start()

_slideshow_resume_event = threading.Event()


# ══════════════════════════════════════════════════════════════════
#  SLIDE MANAGEMENT (add slides in various positions)
# ══════════════════════════════════════════════════════════════════

def handle_slide_management(key: str, text: str) -> bool:
    was_in_slideshow = STATE.in_slideshow
    if was_in_slideshow:
        log.info("📊 Exiting slideshow to manage slides…")
        pyautogui.press("escape")
        time.sleep(0.6)
        STATE.in_slideshow = False

    if key == "add_slide_after":
        pyautogui.hotkey("ctrl", "m")
        log.info("➕ Added slide after current.")
        STATE.record_history(text, "add_slide_after")
    elif key == "add_slide_begin":
        pyautogui.hotkey("ctrl", "home")
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "m")
        pyautogui.hotkey("ctrl", "shift", "up")
        log.info("➕ Added slide at beginning.")
        STATE.record_history(text, "add_slide_begin")
    elif key == "add_slide_end":
        pyautogui.hotkey("ctrl", "end")
        time.sleep(0.2)
        pyautogui.hotkey("ctrl", "m")
        log.info("➕ Added slide at end.")
        STATE.record_history(text, "add_slide_end")
    elif key == "add_slide_now":
        pyautogui.hotkey("ctrl", "m")
        log.info("➕ Added new slide.")
        STATE.record_history(text, "add_slide_now")

    if was_in_slideshow:
        time.sleep(0.3)
        pyautogui.press("f5")
        STATE.in_slideshow = True
        log.info("📊 Slideshow resumed after slide management.")

    return True


# ══════════════════════════════════════════════════════════════════
#  HOTKEY MAP
# ══════════════════════════════════════════════════════════════════

_HOTKEY_EXEC = {
    "copy":       lambda: pyautogui.hotkey("ctrl","c"),
    "paste":      lambda: pyautogui.hotkey("ctrl","v"),
    "undo":       lambda: pyautogui.hotkey("ctrl","z"),
    "redo":       lambda: pyautogui.hotkey("ctrl","y"),
    "select_all": lambda: pyautogui.hotkey("ctrl","a"),
    "save":       lambda: pyautogui.hotkey("ctrl","s"),
    "find":       lambda: pyautogui.hotkey("ctrl","f"),
    "bold":       lambda: pyautogui.hotkey("ctrl","b"),
    "italic":     lambda: pyautogui.hotkey("ctrl","i"),
    "underline":  lambda: pyautogui.hotkey("ctrl","u"),
    "new_line":   lambda: pyautogui.press("enter"),
    "tab":        lambda: pyautogui.press("tab"),
    "scroll_up":  lambda: pyautogui.scroll(4),
    "scroll_down":lambda: pyautogui.scroll(-4),
    "backspace":  lambda: pyautogui.press("backspace"),
    "zoom_in":    lambda: pyautogui.hotkey("ctrl","+"),
    "zoom_out":   lambda: pyautogui.hotkey("ctrl","-"),
}

_PRES_EXEC = {
    "next":       lambda: pyautogui.press("right"),
    "prev":       lambda: pyautogui.press("left"),
    "first":      lambda: (pyautogui.hotkey("ctrl","home")),
    "last":       lambda: (pyautogui.hotkey("ctrl","end")),
    "start_pres": lambda: pyautogui.press("f5"),
    "end_pres":   lambda: pyautogui.press("escape"),
    "black":      lambda: pyautogui.press("b"),
    "white":      lambda: pyautogui.press("w"),
    "zoom_in":    lambda: pyautogui.hotkey("ctrl","+"),
    "zoom_out":   lambda: pyautogui.hotkey("ctrl","-"),
}


# ══════════════════════════════════════════════════════════════════
#  COMMAND EXECUTOR
# ══════════════════════════════════════════════════════════════════

def execute_command(text: str):
    text = text.strip()
    if not text:
        return

    log.info(f"▶ [{STATE.mode.upper()}] \"{text}\"")

    # ── Sleep ────────────────────────────────────────────────────
    if any(w in text for w in CFG["sleep_words"]):
        STATE.active = False
        return

    # ── Mode switch ──────────────────────────────────────────────
    if re.search(r"\btyping\s+mode\b", text):
        STATE.mode = "typing"
        return
    if re.search(r"\bpresentation\s+mode\b", text):
        STATE.mode = "presentation"
        return

    # ── Resume slideshow voice command ───────────────────────────
    if re.search(r"\bresume\s+(the\s+)?slideshow\b|\bresume\s+(the\s+)?presentation\b", text):
        _slideshow_resume_event.set()
        return

    # ── AI Search (FEATURE 2 — plain "search" included) ─────────
    if re.search(
        r"\bsearch\b.+\bon\s+(gemini|chatgpt|copilot|bing)\b"
        r"|\bask\s+(gemini|chatgpt|copilot|bing)\b"
        r"|\bsearch\b",
        text, re.I
    ):
        if handle_ai_search(text):
            return

    # ── Open app / website / file (FEATURE 3 integrated) ────────
    if _OPEN_PATTERN.search(text):
        if handle_open(text):
            STATE.record_history(text, "open")
            return

    # ── Draw shapes (FEATURE 4 unit-aware + FEATURE 5 in-slide) ─
    if _DRAW_PATTERN.search(text):
        if handle_draw(text):
            return

    # ── Typing mode ──────────────────────────────────────────────
    if STATE.mode == "typing":

        if _DELETE_PATTERN.search(text):
            handle_delete(text)
            return

        for sel_key, sel_pattern in _SELECTION_PATTERNS.items():
            if sel_pattern.search(text):
                handle_selection(sel_key, text)
                return

        for key, pattern in _HOTKEY_PATTERNS.items():
            if pattern.search(text):
                _HOTKEY_EXEC[key]()
                STATE.record_history(text, key)
                return

        for direction in ("up", "down", "left", "right"):
            if re.search(rf"\barrow\s+{direction}\b|\bmove\s+{direction}\b", text):
                count = extract_number(text)
                for _ in range(count):
                    pyautogui.press(direction)
                return

        # ── FEATURE 1: smart live-typing deduplication ──────────
        # If live typing already typed some words, the final full-accuracy
        # transcription may repeat them.  We only type what is new.
        already_typed = STATE.get_live_words()
        formatted     = smart_format(text)

        if already_typed and formatted.strip():
            # Compute what is not yet typed
            all_words  = _words_of(formatted.strip())
            new_w      = _new_words(all_words, already_typed)
            if new_w:
                tail = " ".join(new_w) + " "
                pyautogui.write(tail, interval=0.01)
            STATE.record_history(text, "type")
        elif formatted.strip():
            pyautogui.write(formatted, interval=0.01)
            STATE.record_history(text, "type")

        STATE.clear_live_words()

    # ── Presentation mode ────────────────────────────────────────
    elif STATE.mode == "presentation":

        for slide_key, slide_pattern in _PRESENTATION_PATTERNS.items():
            if slide_key.startswith("add_slide") and slide_pattern.search(text):
                handle_slide_management(slide_key, text)
                return

        if re.search(r"\bsearch\b", text, re.I):
            if handle_ai_search(text):
                return

        for key, pattern in _PRESENTATION_PATTERNS.items():
            if key.startswith("add_slide"):
                continue
            if pattern.search(text):
                if key == "start_pres":
                    STATE.in_slideshow = True
                elif key == "end_pres":
                    STATE.in_slideshow = False
                    _slideshow_resume_event.set()
                _PRES_EXEC[key]()
                STATE.record_history(text, f"pres_{key}")
                return

        log.info("No presentation command matched.")


# ══════════════════════════════════════════════════════════════════
#  WORKER THREAD  (single consumer — no interleaving)
# ══════════════════════════════════════════════════════════════════

def _command_worker():
    log.info("🔧 Command worker thread started.")
    while STATE.running:
        try:
            text = _raw_queue.get(timeout=0.5)
            try:
                execute_command(text)
            except Exception as e:
                log.error(f"Executor error: {e}\n{traceback.format_exc()}")
            finally:
                _raw_queue.task_done()
        except queue.Empty:
            continue
    log.info("🔧 Command worker stopped.")


# ══════════════════════════════════════════════════════════════════
#  CONFIDENCE GATE
# ══════════════════════════════════════════════════════════════════

def _confidence_ok(conf: float) -> bool:
    threshold = max(0.0, 1.0 + CFG["confidence_threshold"])
    return conf >= threshold


# ══════════════════════════════════════════════════════════════════
#  SYSTEM TRAY INDICATOR
# ══════════════════════════════════════════════════════════════════

_tray_icon = None

def _make_tray_image(color: str) -> "Image":
    img  = Image.new("RGB", (64, 64), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([8, 8, 56, 56], fill=color)
    return img

def _build_tray():
    if not (TRAY_AVAILABLE and CFG["tray_icon"]):
        return

    def on_quit(icon, item):
        STATE.running = False
        icon.stop()

    def on_toggle(icon, item):
        STATE.active = not STATE.active

    menu = pystray.Menu(
        pystray.MenuItem("Toggle Active", on_toggle),
        pystray.MenuItem("Quit",          on_quit),
    )
    icon = pystray.Icon(
        "SmartPen",
        _make_tray_image("green"),
        "Smart Pen — Idle",
        menu,
    )
    threading.Thread(target=icon.run, daemon=True).start()
    return icon


# ══════════════════════════════════════════════════════════════════
#  MAIN LISTEN LOOP
# ══════════════════════════════════════════════════════════════════

def _process_text(text: str, conf: float) -> bool:
    """
    Central routing after transcription.
    Returns True if program should stop.
    """
    if not text:
        return False

    if any(kw in text for kw in CFG["kill_words"]):
        log.info("🛑 Kill command received.")
        STATE.running = False
        return True

    if any(ww in text for ww in CFG["wake_words"]):
        STATE.active = True
        return False

    if not STATE.active:
        return False

    if not _confidence_ok(conf):
        log.warning(f"⚠  Low confidence ({conf:.2f}) — ignored: \"{text}\"")
        return False

    enqueue(text)
    return False


def listen_always():
    """Always-on VAD loop (no trigger button)."""
    log.info("🔁 Always-listen mode. Say a wake word to activate.")
    log.info(f"   Wake words : {CFG['wake_words']}")
    log.info(f"   Kill words : {CFG['kill_words']}")

    while STATE.running:
        # FEATURE 1: use streaming recorder in typing mode
        audio        = record_vad_streaming()
        text, conf   = transcribe(audio)
        if text:
            log.info(f"📝 \"{text}\" (conf={conf:.2f})")
        if _process_text(text, conf):
            break


def listen_triggered():
    """
    Space-bar / hotkey triggered recording.
    Falls back to always-listen if keyboard lib unavailable.
    """
    if not KEYBOARD_AVAILABLE:
        log.warning("keyboard lib unavailable → always-listen fallback.")
        listen_always()
        return

    TRIGGER_KEY = "f9"
    log.info(f"⌨  Hold {TRIGGER_KEY.upper()} to record.")

    while STATE.running:
        if keyboard.is_pressed(TRIGGER_KEY):
            time.sleep(0.15)
            # FEATURE 1: use streaming recorder in typing mode
            audio      = record_vad_streaming()
            text, conf = transcribe(audio)
            if text:
                log.info(f"📝 \"{text}\" (conf={conf:.2f})")
            if _process_text(text, conf):
                break
        else:
            time.sleep(0.05)


# ══════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    log.info("═" * 60)
    log.info("   SMART PEN v5.0  —  Voice → Action System")
    log.info("═" * 60)
    log.info(f"Config: {CONFIG_FILE.resolve()}")
    log.info(f"Log:    {CFG.get('log_file','stdout')}")

    log.info("")
    log.info("  NEW in v5.0:")
    log.info("  • [F1] Real-time live typing  (words appear while speaking)")
    log.info("  • [F2] Default AI search      (plain 'search X' → ChatGPT)")
    log.info("  • [F3] Context file search    (open <file> in <folder>)")
    log.info("  • [F4] Unit-aware drawing     (cm / mm / inches / feet)")
    log.info("  • [F5] In-slide pen drawing   (draw on live slideshow)")
    log.info("")
    log.info("  Carried forward from v4.0:")
    log.info("  • Slide management  (add slide after/begin/end/now)")
    log.info("  • Delete custom text (delete word/sentence/text <phrase>)")
    log.info("  • Selection commands (select word/sentence/paragraph/line)")
    log.info("  • Dynamic app launch (open <any installed app>)")
    log.info("  • AI search         (search X on gemini/chatgpt/copilot)")
    log.info("  • Slideshow search  (search without stopping presentation)")
    log.info("")

    capture_noise_profile()
    load_asr_model()

    _tray = _build_tray()

    worker = threading.Thread(target=_command_worker, daemon=True, name="CmdWorker")
    worker.start()

    use_trigger = KEYBOARD_AVAILABLE
    log.info(f"Listen mode: {'TRIGGERED (F9)' if use_trigger else 'ALWAYS-ON'}")
    log.info("─" * 60)

    try:
        if use_trigger:
            listen_triggered()
        else:
            listen_always()
    except KeyboardInterrupt:
        log.info("\n⚠  Keyboard interrupt.")
    finally:
        STATE.running = False
        _raw_queue.join()
        log.info("👋 Smart Pen shut down cleanly.")
        hist_file = Path("smart_pen_history.json")
        try:
            with open(hist_file, "w") as f:
                json.dump(STATE._cmd_history, f, indent=2)
            log.info(f"📄 History saved → {hist_file}")
        except Exception:
            pass


if __name__ == "__main__":
    main()