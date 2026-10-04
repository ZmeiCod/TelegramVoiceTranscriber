import json
import sys
import wave
import zipfile
from pathlib import Path

import requests
import soundfile as sf
from vosk import Model, KaldiRecognizer, SetLogLevel

# ── Настройки ──────────────────────────────────────────────
# Папка, куда нужно просто скидывать голосовые сообщения
INPUT_DIR = Path(__file__).parent / "voices"

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
MODEL_NAME = "vosk-model-small-ru-0.22"

# Большая модель (~1.8 GB, точнее) — раскомментируй при необходимости:
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
    
    if model_path.exists():
        return str(model_path)

    MODEL_DIR.mkdir(exist_ok=True)

    if zip_path.exists():
        print(f"📂 Распаковываю существующий архив...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(MODEL_DIR)
        zip_path.unlink()
        print("✅ Модель готова!\n")
        return str(model_path)

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
    
    if sr != SAMPLE_RATE:
        import numpy as np
        duration = len(audio) / sr
        new_length = int(duration * SAMPLE_RATE)
        audio = np.interp(
            np.linspace(0, len(audio), new_length),
            np.arange(len(audio)),
            audio
        ).astype('int16')
    
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


# ── Вспомогательные функции ────────────────────────────────
def get_target_files(limit: int = None) -> list[Path]:
    """Возвращает список .ogg файлов из папки voices, отсортированных по имени."""
    INPUT_DIR.mkdir(exist_ok=True)
    files = list(INPUT_DIR.glob("*.ogg"))
    
    if not files:
        return []
    
    # Сортировка по имени (для audio_YYYY-MM-DD_HH-MM-SS.ogg это хронологический порядок)
    files.sort(key=lambda p: p.name)
    
    if limit is not None:
        return files[-limit:]  # Берем последние N файлов
    
    return files


def process_single_file(file_path: Path):
    """Обрабатывает один файл: транскрибирует и выводит результат в консоль."""
    print("-" * 60)
    print(f"📄 Файл: {file_path.name}")
    
    print("🎧 Распознаю...")
    text = transcribe(str(file_path))
    
    print(f"📝 Результат:\n{text if text else '🤷 Речь не распознана.'}\n")


# ── CLI (Командная строка) ─────────────────────────────────
if __name__ == "__main__":
    # Сразу проверяем модель, чтобы не прерывать процесс скачиванием посередине
    print("🔄 Проверка модели...")
    ensure_model()

    if len(sys.argv) == 1:
        # Режим 1: Обработать ВСЕ файлы в папке voices
        files = get_target_files()
        if not files:
            print(f"📁 Папка '{INPUT_DIR}' пуста или не содержит .ogg файлов.")
            print("💡 Скопируйте голосовые сообщения в эту папку и запустите скрипт снова.")
            sys.exit(0)
        
        print(f"🎯 Найдено файлов: {len(files)}. Начинаю обработку по очереди...\n")
        for f in files:
            process_single_file(f)

    elif len(sys.argv) == 2:
        arg = sys.argv[1]
        
        if arg.isdigit():
            # Режим 2: Обработать последние N файлов
            limit = int(arg)
            files = get_target_files(limit=limit)
            if not files:
                print(f"📁 В папке '{INPUT_DIR}' нет файлов для обработки.")
                sys.exit(0)
            
            print(f"🎯 Обрабатываю последние {limit} файл(ов)...\n")
            for f in files:
                process_single_file(f)
        else:
            # Режим 3: Обработать конкретный файл по имени или пути
            target_path = Path(arg)
            
            # Если файла нет в текущей папке, проверяем папку voices
            if not target_path.exists():
                alt_path = INPUT_DIR / arg
                if alt_path.exists():
                    target_path = alt_path
                else:
                    print(f"❌ Файл не найден: {arg}")
                    sys.exit(1)
            
            process_single_file(target_path)

    else:
        print("⚠️ Неверный формат команды.")
        print("\n📘 Использование:")
        print("  python transcribe.py          # обработать все .ogg в папке 'voices'")
        print("  python transcribe.py N        # обработать последние N файлов в папке 'voices'")
        print("  python transcribe.py file.ogg # обработать конкретный файл (из текущей папки или 'voices')")
        sys.exit(1)