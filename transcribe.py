import json
import sys
import wave
import zipfile
from pathlib import Path

import requests
import soundfile as sf
from vosk import Model, KaldiRecognizer, SetLogLevel

# ── Настройки ──────────────────────────────────────────────
# Маленькая модель (~40 MB, быстро, decent качество)
MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
MODEL_NAME = "vosk-model-small-ru-0.22"

# Большая (~1.8 GB, точнее) — раскомментируй:
# MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-ru-0.42.zip"
# MODEL_NAME = "vosk-model-ru-0.42"

MODEL_DIR = Path(__file__).parent / "model"
SAMPLE_RATE = 16000

SetLogLevel(-1)  # тишина от Vosk


# ── Модель ─────────────────────────────────────────────────
def ensure_model() -> str:
    """Скачивает и распаковывает модель в ./model/, если её нет."""
    model_path = MODEL_DIR / MODEL_NAME
    zip_path = MODEL_DIR / "model.zip"
    
    # Если модель уже распакована — используем её
    if model_path.exists():
        return str(model_path)

    MODEL_DIR.mkdir(exist_ok=True)

    # Если zip уже есть, но не распакован — распаковываем
    if zip_path.exists():
        print(f"📂 Распаковываю существующий архив...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MODEL_DIR)
        zip_path.unlink()
        print("✅ Модель готова!\n")
        return str(model_path)

    # Иначе скачиваем
    print(f"📦 Модель не найдена. Скачиваю (~40 MB)...")
    resp = requests.get(MODEL_URL, stream=True)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with open(zip_path, "wb") as f:
        downloaded = 0
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                bar = downloaded * 30 // total
                pct = downloaded * 100 // total
                print(f"\r   [{'█' * bar}{'·' * (30 - bar)}] {pct}%", end="", flush=True)

    print("\n📂 Распаковываю...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(MODEL_DIR)
    zip_path.unlink()
    print("✅ Модель готова!\n")
    return str(model_path)

# ── Конвертация ────────────────────────────────────────────
def ogg_to_wav(ogg_path: str) -> str:
    """Telegram .ogg (Opus) → 16 kHz mono WAV для Vosk."""
    audio, sr = sf.read(ogg_path, dtype='int16')
    
    # Ресемплинг до 16 kHz если нужно
    if sr != SAMPLE_RATE:
        import numpy as np
        duration = len(audio) / sr
        new_length = int(duration * SAMPLE_RATE)
        audio = np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        ).astype('int16')
    
    # Если стерео — делаем моно
    if audio.ndim > 1:
        audio = audio.mean(axis=1).astype('int16')
    
    wav_path = str(Path(ogg_path).with_suffix(".wav"))
    sf.write(wav_path, audio, SAMPLE_RATE, subtype='PCM_16')
    return wav_path


# ── Распознавание ──────────────────────────────────────────
def transcribe(audio_path: str) -> str:
    """Принимает .ogg или .wav, возвращает распознанный текст."""
    model_path = ensure_model()
    model = Model(model_path)
    rec = KaldiRecognizer(model, SAMPLE_RATE)

    wav_path = None
    if audio_path.lower().endswith(".ogg"):
        wav_path = ogg_to_wav(audio_path)
        audio_path = wav_path

    try:
        wf = wave.open(audio_path, "rb")
        chunks = []

        while True:
            data = wf.readframes(4000)
            if not data:
                break
            if rec.AcceptWaveform(data):
                res = json.loads(rec.Result())
                if res.get("text"):
                    chunks.append(res["text"])

        final = json.loads(rec.FinalResult())
        if final.get("text"):
            chunks.append(final["text"])

        wf.close()
        return " ".join(chunks).strip()
    finally:
        if wav_path and Path(wav_path).exists():
            Path(wav_path).unlink()


# ── CLI ────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Использование:  python transcribe.py <файл.ogg>")
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"❌ Файл не найден: {path}")
        sys.exit(1)

    print("🎧 Распознаю...")
    text = transcribe(path)
    print(f"\n📝 Результат:\n{text}" if text else "\n🤷 Речь не распознана.")