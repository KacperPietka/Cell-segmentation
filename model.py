import requests
import json
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import re
import tempfile
import cv2 as cv
import numpy as np
from cellpose import models
from cellpose.io import imread
from Variables import Variables
from Image import Image
import model_instruction

class Model(Image):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.url = Variables.URL
        self.speech = sr.Recognizer()
        self.response_text = ""
        self.command = None
        self.color = Variables.RED_BGR
        self.user_input = ""
        self.end = False
        self.payload = {
        'model': 'mistral',
        'messages': [
            {
                "role": "system",
                "content": model_instruction.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": self.user_input
            }
        ]
        }
        self.command_dispatch = {
            "zoom_in": self.zoom_in,
            "zoom_out": self.zoom_out,
            "cell_segmentation": self.cell_segmentation,
            "lighting_modification": self.lighting_modification,
            "quit": self.quit,
            "undo_all": self.undo_all,
            "change_image": self.change_image
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
                #print(parts)
                if len(parts) == 2:
                    if parts[1] in ['top_right', 'top_left', 'bottom_left', 'bottom_right', 'center']:
                        self.command, self.zoom_position = parts[0], parts[1]
                    #elif parts[1] in ['up', 'down', 'left', 'right']:
                        #self.command, self.move_position = parts[0], parts[1]
                    elif re.search("image", parts[1]):
                        self.reset_variables()
                        self.command, self.image_path = parts[0], './images/' + parts[1] + '.png'
                    elif parts[1] in ['50', '-50']:
                        self.command, self.brightness = parts[0], int(parts[1])
                elif len(parts) == 4: #colors
                    self.command, self.color = parts[0], parts[1:]
                else:
                    self.command = parts[0]
            if self.command in self.command_dispatch:
                self.command_dispatch[self.command]()
        else:
            print("Error")


    def reset_variables(self):
        self.url = 'http://localhost:11434/api/chat'
        self.speech = sr.Recognizer()
        self.response_text = ""
        self.command = None
        self.color = Variables.RED_BGR
        self.user_input = ""
        self.end = False
        self.current_zoom = (0, 0, self.image.shape[1], self.image.shape[0])
        self.zoom_factor = Variables.ORIGINAL_ZOOM_FACTOR
        self.brightness = Variables.BRIGHTNESS_STEP
        self.brightness_level = Variables.DEFAULT_BRIGHTNESS_LEVEL
        self.zoom_position = Variables.DEFAULT_ZOOM_POSITION
    

    def undo_all(self):
        self.image = self.original_image
        self.reset_variables()
        cv.imshow("Display window", self.image)


    def cell_segmentation(self):
        self.reset_variables()
        self.image = self.original_image.copy()

        model = models.CellposeModel(gpu=True)
        masks, flows, styles = model.eval(self.image, diameter=Variables.CELLPOSE_DIAMETER)

        image_BGR = self.image.copy()

        binary_mask = (masks > 0).astype(np.uint8)

        overlay_color = tuple(int(c) for c in self.color)

        # For each channel, blend overlay_color onto image where mask is 1 (Chatgpt)
        for c in range(3):
            image_BGR[:, :, c] = np.where(
                binary_mask == 1,
                (Variables.ALPHA * overlay_color[c] + (1 - Variables.ALPHA) * image_BGR[:, :, c]).astype(np.uint8),
                image_BGR[:, :, c]
            )

        self.image = image_BGR
        self.image = cv.convertScaleAbs(self.image, beta=self.brightness_level, alpha=1.0)
        cv.imshow("Display window", self.image)

    def quit(self):
        self.end = True

    """ TBD

    Available move positions:
        - up 
        - down
        - left
        - right

    - move: move current zoom

    User: "please move to the left"
    Okay! moving the zoom to the left.
    move left

    def move(self):
        h, w = self.image.shape[:2]
        x0,y0,w0,h0 = self.current_zoom

        if self.zoom_factor == 1: #cannot move the already zoomed out image
            return

        if self.move_position == 'top':
            x1 = x0
            y1 = y0 - h0
        elif self.move_position == 'down':
            x1 = x0
            y1 = y0 + h0
        elif self.move_position == 'left':
            x1 = x0 - w0
            y1 = y0
        elif self.move_position == 'right':
            x1 = x0 + w0 
            y1 = y0

        x1 = max(0, min(x1, w - w0))
        y1 = max(0, min(y1, h - h0))
        x2 = x1 + w0
        y2 = y1 + h0

        cropped = self.image[y1:y2, x1:x2]
        moved_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
        cv.imshow("Display window", moved_image)
        cv.waitKey(30)
        self.current_zoom = (x1, y1, w0, h0)
    """


image = Image("./images/image1.png")
model = Model("./images/image1.png")
cv.imshow("Display window", image.image)
cv.waitKey(1)

while model.end != True:
    with sr.Microphone() as source:
        print("You can talk now") 
        audio_data = model.speech.listen(source, phrase_time_limit=10, timeout=10)
        try:
            model.process_speech_input(audio_data)
        except sr.UnknownValueError:
            print("Sorry, I could not understand your voice.")
        except sr.RequestError as e:
            print(f"Speech recognition error: {e}") 
    if cv.waitKey(1) & 0xFF == 27: #press any key to end
        break
cv.destroyAllWindows()