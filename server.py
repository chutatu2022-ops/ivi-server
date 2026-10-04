from flask import Flask, request, Response
from openai import OpenAI
import io
import re
import os
import json
import threading

app = Flask(__name__)

# Считывание ключа из настроек сервера (Render Environment Variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

STT_MODEL = "gpt-4o-transcribe"
GPT_MODEL = "gpt-5-mini"
TTS_MODEL = "gpt-4o-mini-tts"

active_reminders = []

def on_reminder_triggered(reminder_text):
    print(f"[ТАЙМЕР СПРАЦЮВАВ] Нагадування: '{reminder_text}'")
    active_reminders.append(reminder_text)

def schedule_reminder(seconds, text):
    timer = threading.Timer(seconds, on_reminder_triggered, args=[text])
    timer.daemon = True
    timer.start()

def pop_due_reminders():
    global active_reminders
    if not active_reminders:
        return ""
    text = "Нагадую: " + ". ".join(active_reminders) + "."
    active_reminders = []
    return text

REMINDER_TOOL = {
    "type": "function",
    "name": "set_reminder",
    "description": "Встановлює нагадування для користувача через вказану кількість секунд",
    "parameters": {
        "type": "object",
        "properties": {
            "seconds": {"type": "integer", "description": "Кількість секунд"},
            "text": {"type": "string", "description": "Текст завдання"}
        },
        "required": ["seconds", "text"]
    }
}

HISTORY_FILE = "history.json"
MAX_MESSAGES = 20

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception:
            pass
    return []

def save_history(history):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

def clear_history():
    global conversation
    conversation = []
    save_history(conversation)

conversation = load_history()

SYSTEM_PROMPT = """
Ти — кишеньковий голосовий помічник на ім'я Іві.
Спілкуйся виключно українською мовою.

Твої відповіді озвучуються вголос через динамік робота:
- Відповідай лаконічно, коротко і зрозуміло (1-2 речення).
- Якщо просять щось нагадати — викликай set_reminder.
- Не використовуй списки, markdown, лапки чи зірочки.
- Будь живою, дотепною та дружньою.
"""

HALLUCINATIONS = [
    "це запис української мови", "це запис українською мовою",
    "дякую за перегляд", "дякую за увагу", "субтитри",
    "продовження буде", "до нових зустрічей", "підписуйтесь"
]

def text_to_speech(text):
    clean = (text or "").strip() or "Слухаю вас."
    print("TTS:", clean)
    response = client.audio.speech.create(
        model=TTS_MODEL,
        voice="shimmer",
        input=clean,
        response_format="wav"
    )
    return response.content

def speech_to_text(audio_bytes):
    try:
        f = io.BytesIO(audio_bytes)
        f.name = "audio.wav"
        res = client.audio.transcriptions.create(model=STT_MODEL, file=f, language="uk")
        text = res.text.strip()
        print("STT:", repr(text))
        return text
    except Exception as e:
        print("STT ERROR:", e)
        return ""

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text)

def is_clear_memory(text):
    norm = normalize_text(text)
    return any(p in norm for p in ["забудь все", "забудь усе", "очисти пам ять", "скинь історію"])

def is_sleep(text):
    norm = normalize_text(text)
    words = ["спати", "іди спати", "йди спати", "режим сну", "засинай", "відбій", "до побачення"]
    return any(w in norm for w in words)

def is_garbage(text):
    norm = normalize_text(text)
    if len(norm) < 2 or any(h in norm for h in HALLUCINATIONS):
        return True
    return False

def is_wake(text):
    norm = normalize_text(text)
    return any(v in norm for v in ["іві", "иві", "иви", "іва", "evi", "ivi"])

def ask_gpt(user_text):
    global conversation
    conversation.append({"role": "user", "content": user_text})
    if len(conversation) > MAX_MESSAGES:
        conversation = conversation[-MAX_MESSAGES:]

    try:
        response = client.responses.create(
            model=GPT_MODEL,
            instructions=SYSTEM_PROMPT,
            input=conversation,
            tools=[{"type": "web_search_preview"}, REMINDER_TOOL]
        )

        reminder_msg = ""
        for item in getattr(response, "output", []):
            if getattr(item, "type", "") == "function_call":
                if getattr(item, "name", "") == "set_reminder":
                    args = json.loads(item.arguments)
                    sec = int(args.get("seconds", 60))
                    reminder_msg = str(args.get("text", "Нагадування"))
                    schedule_reminder(sec, reminder_msg)

        answer = (response.output_text or "").strip()
        if not answer:
            answer = f"Добре, я нагадаю вам: {reminder_msg}." if reminder_msg else "Я вас слухаю."

        conversation.append({"role": "assistant", "content": answer})
        save_history(conversation)
        return answer
    except Exception as e:
        print("GPT ERROR:", e)
        return "Вибачте, виникла помилка під час запиту."

@app.route("/", methods=["GET"])
def index():
    return "IVI Cloud Server is running 24/7!", 200

@app.route("/check_reminders", methods=["GET"])
def check_reminders():
    due = pop_due_reminders()
    if not due:
        return Response("", status=204)
    audio = text_to_speech(due)
    resp = Response(audio, status=200, mimetype="audio/wav")
    resp.headers["X-Alarm"] = "1"
    return resp

@app.route("/wake", methods=["POST"])
def wake():
    audio = request.data
    if not audio:
        return Response("NO_AUDIO", status=400)
    text = speech_to_text(audio)
    if not text or not is_wake(text):
        return Response("", status=204)

    due = pop_due_reminders()
    msg = f"{due} Я слухаю, хазяїн." if due else "Я слухаю, хазяїн."
    audio_resp = text_to_speech(msg.strip())
    resp = Response(audio_resp, status=200, mimetype="audio/wav")
    resp.headers["X-Sleep"] = "0"
    return resp

@app.route("/upload", methods=["POST"])
def upload():
    audio = request.data
    if not audio:
        return Response("NO_AUDIO", status=400)

    text = speech_to_text(audio)
    if not text:
        resp = Response("", status=204)
        resp.headers["X-Silence"] = "1"
        return resp

    if is_clear_memory(text):
        clear_history()
        audio_resp = text_to_speech("Пам'ять очищено. Почнімо спочатку.")
        resp = Response(audio_resp, status=200, mimetype="audio/wav")
        resp.headers["X-Sleep"] = "0"
        return resp

    if is_sleep(text):
        audio_resp = text_to_speech("Переходжу в режим очікування.")
        resp = Response(audio_resp, status=200, mimetype="audio/wav")
        resp.headers["X-Sleep"] = "1"
        return resp

    if is_garbage(text):
        resp = Response("", status=204)
        resp.headers["X-Silence"] = "1"
        return resp

    due = pop_due_reminders()
    answer = ask_gpt(text)
    full = (due + " " + answer).strip()

    audio_resp = text_to_speech(full)
    resp = Response(audio_resp, status=200, mimetype="audio/wav")
    resp.headers["X-Sleep"] = "0"
    return resp

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
