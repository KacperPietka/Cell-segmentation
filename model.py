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
        self.current_zoom = (0, 0, self.image.shape[1], self.image.shape[0])
        self.zoom_factor = 1
        self.command = None
        self.zoom_position = 'center'
        self.user_input = ""
        self.payload = {
            'model': 'llama3.2',
            'messages': [
        {
            "role": "system",
            "content": """You are a voice assistant in a cell segmentation software When someone says commands like "zoom in" or "color every cell in red", respond naturally in the first lines like OKAY!, zooming in. If someone says zoom out you say sure! zooming out. etc.
            Then, on the LAST line of your output, output ONLY the command name and optionally the zoom position.


            Available commands:
            - zoom_in: Zoom in the current view.
            - zoom_out: Zoom out the current view.
            - none: Do nothing or no known command recognized.

            Available zoom positions:
            - center (default)
            - top left
            - top right
            - bottom left
            - bottom right

            Output format MUST be exactly:
            [Response to user text]
            [command_name] [position (optional, no extra spaces)]

            User: "please zoom in top left"
            Okay! Zooming in top left.
            zoom_in top_left

            User: "please zoom out"
            Okay! Zooming out.
            zoom_out

            User: "how's the weather?"
            Sorry, I can only help with software commands.
            none

            IMPORTANT:  
            - The LAST line must contain ONLY the command and optional position, nothing else.  
            - Do NOT output any extra blank lines or trailing spaces after the command line.
            - If no position is given or needed, output only the command."""
        },
        {
            "role": "user",
            "content": self.user_input
        }
        ]
        }
        self.command_dispatch = {
            "zoom_in": self.zoom_in,
            "zoom_out": self.zoom_out
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
        lines = None
        parts = []
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        self.response_text += content
            lines = self.response_text.strip().splitlines()
            print(lines[0])
            self.speak(lines[0], lang='en', tld='co.uk')
            if lines:
                parts = lines[-1].strip().split()
                if len(parts) == 2:
                    self.command, self.zoom_position = parts[0], parts[1]
                else:
                    self.command = parts[0]
            if self.command in self.command_dispatch:
                self.command_dispatch[self.command]()
        else:
            print("Error")

    def zoom_in(self):
        self.zoom_factor = 1.0
        zooming_threshold = self.zoom_factor*1.5

        x0,y0,w0,h0 = self.current_zoom
        while self.zoom_factor < zooming_threshold:
            self.zoom_factor += 0.05
            h, w = self.image.shape[:2]

            new_w = int(w0 / self.zoom_factor)
            new_h = int(h0 / self.zoom_factor)


            if self.zoom_position == 'center':
                x1 = x0 + (w0 - new_w) // 2
                y1 = y0 + (h0 - new_h) // 2
            elif self.zoom_position == 'top_left':
                x1 = x0
                y1 = y0
            elif self.zoom_position == 'top_right':
                x1 = x0 + w0 - new_w
                y1 = y0
            elif self.zoom_position == 'bottom_left':
                x1 = x0
                y1 = y0 + h0 - new_h
            elif self.zoom_position == 'bottom_right':
                x1 = x0 + w0 - new_w
                y1 = y0 + h0 - new_h

            x2 = x1 + new_w
            y2 = y1 + new_h
            cropped = self.image[y1:y2, x1:x2]
            zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
            cv.imshow("Display window", zoomed_image)
            if cv.waitKey(30) & 0xFF == 27:
                break
        self.current_zoom = (x1, y1, x2 - x1, y2 - y1)


    def zoom_out(self):
        while True:
            if self.zoom_factor <= 1:
                break
            self.zoom_factor -= 0.05

            h, w = self.image.shape[:2]

            new_w = int(w / self.zoom_factor)
            new_h = int(h / self.zoom_factor)

            if self.zoom_position == 'center':
                x1 = (w - new_w) // 2
                y1 = (h - new_h) // 2
            elif self.zoom_position == 'top_left':
                x1 = 0
                y1 = 0
            elif self.zoom_position == 'top_right':
                x1 = w - new_w
                y1 = 0
            elif self.zoom_position == 'bottom_left':
                x1 = 0
                y1 = h - new_h
            elif self.zoom_position == 'bottom_right':
                x1 = w - new_w
                y1 = h - new_h
            else:
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