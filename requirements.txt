import os, tempfile, subprocess
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    user_msg = data.get("message", "")
    system_msg = data.get("system", "You are a helpful assistant.")
    if not user_msg:
        return jsonify({"error": "message is required"}), 400
    r = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":system_msg},
                  {"role":"user","content":user_msg}]
    )
    return jsonify({"reply": r.choices[0].message.content})

@app.route("/transcribe", methods=["POST"])
def transcribe():
    data = request.get_json(force=True)
    url = data.get("url", "")
    if not url:
        return jsonify({"error":"url is required"}), 400
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, "audio.mp3")
        cmd = ["yt-dlp","-x","--audio-format","mp3","-o", out_path, url]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0 or not os.path.exists(out_path):
            return jsonify({"error": "yt-dlp failed"}), 500
        audio_file = open(out_path, "rb")
        r = client.audio.transcriptions.create(
            file=audio_file,
            model="gpt-4o-mini-transcribe"
        )
        return jsonify({"text": r.text})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000)
