import requests
import json
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import tempfile
import cv2 as cv
import numpy as np




class Model:
    def __init__(self):
        self.url = 'http://localhost:11434/api/chat'
        self.image = cv.imread("cell.jpg")
        self.speech = sr.Recognizer()
        self.response_text = ""
        self.command = None
        self.user_input = ""
        self.payload = {
            'model': 'llama3.2',
            'messages': [
        {
            "role": "system",
            "content": """You are a voice assistant in a cell segmentation software so when someone says zoom in you respond naturally "OKAY! zooming in" and if someone says color every cell on red
            you say "Sure thing! coloring every cell in red" etc. 
            Output ONLY the command name in the last line.

            Available commands:
            - zoom_in: Zoom in the current view.
            - zoom_out: Zoom out the current view.
            - none: Do nothing or no known command recognized.

            Respond ONLY with the correct command name: zoom_in, zoom_out.
            Output format:
            [Response to user]
            [Command name (e.g., zoom_in)]

            User message: "please zoom in a bit"
            Okay! Zooming in.
            zoom_in

            User: "how's the weather?"
            Sorry, I can only help with image viewing commands.
            none"""
        },
        {
            "role": "user",
            "content": self.user_input
        }
        ]
        }
        self.command_dispatch = {
            "zoom_in": self.zoom_in
        }

    def speak(self, text, lang='en', tld='com'):
        tts = gTTS(text=text, lang=lang, tld=tld)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)
        playsound.playsound(temp_path)
        os.remove(temp_path)



    def process_speech_input(self, audio_data):
        self.user_input = ""
        self.response_text = ""
        self.user_input = self.speech.recognize_google(audio_data)
        self.payload['messages'][1]['content'] = self.user_input #update user input
        print("You said:", self.user_input)

        response = requests.post(self.url, json=self.payload, stream=True)

        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        self.response_text += content
            lines = model.response_text.strip().splitlines()
            print(lines[0])
            self.speak(lines[0], lang='en', tld='co.uk')
            if lines:
                model.command = lines[-1].strip()
            if model.command in model.command_dispatch:
                model.command_dispatch[model.command]()
        else:
            print("Error")

    def zoom_in(self, zoom_factor=1):
            while True:
                zoom_factor += 0.05
                if zoom_factor > 3:
                    break
                h, w = self.image.shape[:2]

                new_w = int(w / zoom_factor)
                new_h = int(h / zoom_factor)

                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
                x2 = x1 + new_w
                y2 = y1 + new_h

                cropped = self.image[y1:y2, x1:x2]
                zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
                cv.imshow("Display window", zoomed_image)
                if cv.waitKey(30) & 0xFF == 27:
                    break

model = Model()

cv.imshow("Display window", model.image)
cv.waitKey(1)

while True:
    if cv.waitKey(30) != -1:
        break
    with sr.Microphone() as source:
        print("You can talk now") 
        audio_data = model.speech.listen(source)
        try:
            model.process_speech_input(audio_data)
        except sr.UnknownValueError:
            print("Sorry, I could not understand your voice.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}") 

cv.destroyAllWindows()