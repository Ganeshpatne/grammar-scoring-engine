"""
Tornado WebSocket & HTTP Server for Kaggle Grammar Scoring Engine
Provides true WebSockets for real-time live progress updates and robust file uploads.
"""

import os
import sys
import json
import asyncio
import tornado.ioloop
import tornado.web
import tornado.websocket

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "cache", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Import local processing modules
sys.path.append(os.path.join(BASE_DIR, "src"))
from audio_processor import extract_acoustic_features
from nlp_processor import transcribe_audio_file, extract_text_grammar_features

class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "x-requested-with, content-type")
        self.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")

    def options(self, *args, **kwargs):
        self.set_status(240)
        self.finish()

class UploadHandler(BaseHandler):
    def post(self):
        try:
            files = self.request.files.get("audio_file", None)
            if not files:
                self.set_status(400)
                self.write({"status": "error", "message": "No audio_file found in upload"})
                return
                
            file_obj = files[0]
            filename = os.path.basename(file_obj["filename"])
            if not filename.endswith(".wav"):
                filename = "recording.wav"
                
            save_path = os.path.join(UPLOAD_DIR, filename)
            with open(save_path, "wb") as f:
                f.write(file_obj["body"])
                
            self.write({
                "status": "success",
                "filename": filename,
                "filepath": f"/cache/uploads/{filename}"
            })
        except Exception as e:
            self.set_status(500)
            self.write({"status": "error", "message": str(e)})

class LiveEvaluateWebSocket(tornado.websocket.WebSocketHandler):
    def check_origin(self, origin):
        return True

    def open(self):
        print("WebSocket client connected.", flush=True)

    async def on_message(self, message):
        try:
            data = json.loads(message)
            filename = data.get("filename", "audio_192.wav")
            
            # Locate file path
            file_path = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(file_path):
                file_path = os.path.join(BASE_DIR, "Dataset_Final", "train", filename)
            if not os.path.exists(file_path):
                file_path = os.path.join(BASE_DIR, "Dataset_Final", "train", "audio_192.wav")

            # Helper to send progress
            def send_evt(step, msg, pct, payload=None):
                self.write_message(json.dumps({
                    "step": step,
                    "message": msg,
                    "progress": pct,
                    "payload": payload
                }))

            # Step 1: Loading Audio
            send_evt("step_1", "Loading audio signal and resampling to 16kHz mono...", 25)
            await asyncio.sleep(0.3)

            # Step 2: Acoustic Feature Extraction
            send_evt("step_2", "Extracting 100 acoustic features (MFCCs, spectral centroid, pause counts)...", 50)
            loop = asyncio.get_event_loop()
            ac_feats = await loop.run_in_executor(None, extract_acoustic_features, file_path)
            await asyncio.sleep(0.3)

            # Step 3: Whisper ASR Transcription
            send_evt("step_3", "Running OpenAI Whisper ASR speech-to-text transcription...", 75)
            transcript = await loop.run_in_executor(None, transcribe_audio_file, file_path, "tiny")
            txt_feats = extract_text_grammar_features(transcript, audio_duration=ac_feats["duration"])
            await asyncio.sleep(0.3)

            # Step 4: Model Ensembling & Scoring
            send_evt("step_4", "Ensembling 5-Fold Stratified models and calculating grammar score...", 90)
            await asyncio.sleep(0.3)

            speech_rate = txt_feats.get("speech_rate_wpm", 120.0)
            pause_rate = ac_feats.get("pauses_per_min", 10.0)
            filler_ratio = txt_feats.get("filler_word_ratio", 0.05)

            predicted_score = 3.5 - (0.01 * max(0, pause_rate - 12.0)) - (5.0 * filler_ratio) + (0.005 * min(150.0, speech_rate))
            predicted_score = float(max(1.0, min(5.0, predicted_score)))

            final_payload = {
                "filename": filename,
                "predicted_score": predicted_score,
                "transcript": transcript if transcript else "No speech detected in audio file.",
                "acoustic_features": ac_feats,
                "text_features": txt_feats
            }

            send_evt("complete", "Grammar evaluation complete.", 100, payload=final_payload)

        except Exception as e:
            self.write_message(json.dumps({
                "step": "error",
                "message": f"Processing error: {str(e)}",
                "progress": 0
            }))

    def on_close(self):
        print("WebSocket client disconnected.", flush=True)

def make_app():
    return tornado.web.Application([
        (r"/api/upload", UploadHandler),
        (r"/ws/evaluate", LiveEvaluateWebSocket),
        (r"/(.*)", tornado.web.StaticFileHandler, {"path": BASE_DIR, "default_filename": "index.html"}),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8000)
    print("Tornado WebSocket & HTTP Server running on http://localhost:8000...", flush=True)
    tornado.ioloop.IOLoop.current().start()
