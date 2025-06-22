import requests
import json
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import playsound
import os
import tempfile
import cv2 as cv
import numpy as np




class Model:
    def __init__(self):
        self.url = 'http://localhost:11434/api/chat'
        #self.voice = pyttsx3.init()
        self.speech = sr.Recognizer()
        self.response_text = ""

    def speak(self, text, lang='en', tld='com'):
        tts = gTTS(text=text, lang=lang, tld=tld)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_path = fp.name
            tts.save(temp_path)
        playsound.playsound(temp_path)
        os.remove(temp_path)

    def process_speech_input(self, audio_data):

        user_input = self.speech.recognize_google(audio_data)
        print("You said:", user_input)


        payload = {
            'model' : 'assistant',
            'messages' : [{"role": "user", "content": user_input}]}

        response = requests.post(self.url, json=payload, stream=True)
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        print(content, end='', flush=True)
                        self.response_text += content
            self.speak(self.response_text, lang='en', tld='co.uk')  # Change `tld` for accents
        else:
            print("Error")



def zoom_in(image, zoom_factor=1.5):

        h, w = image.shape[:2]

        new_w = int(w / zoom_factor)
        new_h = int(h / zoom_factor)

        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        x2 = x1 + new_w
        y2 = y1 + new_h

        cropped = image[y1:y2, x1:x2]
        zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)

        return zoomed_image

model = Model()
img = cv.imread("cell.jpg")
cv.imshow("Display window", img)
cv.waitKey(1)

with sr.Microphone() as source:
    print("You can talk now") 
    model.response_text = ""
    audio_data = model.speech.listen(source)
    try:
        model.process_speech_input(audio_data)
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")

    if "zoom" in model.response_text.lower():
        zoom_factor = 1
        while True:
            zoom_factor += 0.05
            if zoom_factor > 3:
                break
            zoomed_img = zoom_in(img, zoom_factor)
            cv.imshow("Display window", zoomed_img)
            if cv.waitKey(30) & 0xFF == 27:
                break
k = cv.waitKey(0)
cv.destroyAllWindows()