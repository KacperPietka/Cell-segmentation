import requests
import json
import speech_recognition as sr
from gtts import gTTS
import playsound
import os
import tempfile
import cv2 as cv
import numpy as np
from cellpose import models
from cellpose.io import imread

class Config:
    IMAGE_SIZE = (1200, 1000)
    BRIGHTNESS_STEP = 50
    CELLPOSE_DIAMETER = 30
    ORIGINAL_ZOOM_FACTOR = 1
    DEFAULT_ZOOM_POSITION = 'center'
    DEFAULT_BRIGHTNESS_LEVEL = 0

class Image:
    def __init__(self, image_path):
        img = cv.imread(image_path)
        self.image = cv.resize(img, (1200, 1000))
        self.original_image = self.image.copy()
        self.current_zoom = (0, 0, self.image.shape[1], self.image.shape[0])
        self.zoom_factor = 1
        self.brightness = 50
        self.brightness_level = 0
        self.zoom_position = 'center'

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
            cv.waitKey(30)
        self.current_zoom = (x1, y1, x2 - x1, y2 - y1)


    def zoom_out(self):
        x0,y0,w0,h0 = self.current_zoom
        h, w = self.image.shape[:2]
        
        self.zoom_factor = w / w0
        center_x = x0 + w0 // 2
        center_y = y0 + h0 // 2

        if self.zoom_factor == 1:
            return

        while self.zoom_factor > 1:
            self.zoom_factor -= 0.05
            if self.zoom_factor < 1:
                self.zoom_factor = 1
            new_w = int(w / self.zoom_factor)
            new_h = int(h / self.zoom_factor)

            x1 = center_x - new_w // 2
            y1 = center_y - new_h // 2

            x1 = max(0, min(w - new_w, x1))
            y1 = max(0, min(h - new_h, y1))
            x2 = x1 + new_w
            y2 = y1 + new_h

            cropped = self.image[y1:y2, x1:x2]
            zoomed_image = cv.resize(cropped, (w, h), interpolation=cv.INTER_LINEAR)
            cv.imshow("Display window", zoomed_image)
            cv.waitKey(30)
        self.current_zoom = (x1, y1, x2 - x1, y2 - y1)

    def lighting_modification(self):
        temp_img = self.original_image.copy()
        
        self.brightness_level += self.brightness

        self.image = cv.convertScaleAbs(temp_img, beta=self.brightness_level, alpha=1.0)

        cv.imshow("Display window", self.image)

class Model(Image):
    def __init__(self, image_path):
        super().__init__(image_path)
        self.url = 'http://localhost:11434/api/chat'
        self.speech = sr.Recognizer()
        self.response_text = ""
        self.command = None
        self.color = [0, 0, 255]
        #self.move_position = None
        self.user_input = ""
        self.end = False
        self.payload = {
            'model': 'mistral',
            'messages': [
        {
            "role": "system",
            "content": """You are a voice assistant in a cell segmentation software When someone says commands respond naturally in the first lines like OKAY!, zooming in. If someone says zoom out you say sure! zooming out. etc.
            Then, on the LAST line of your output, output ONLY the command name and optionally the zoom position.


            Available commands:
            - zoom_in: Zoom in the current view.
            - zoom_out: Zoom out the current view.
            - cell_segmentation: Color every cell.
            - quit: Close everything and say goodbye.
            - lighting_modification: Modify the brightness
            - none: Do nothing or no known command recognized.

            IF you dont recognize a command respond the command none!

            Available zoom positions:
            - center (default)
            - top left
            - top right
            - bottom left
            - bottom right

            Color Name to BGR:
            red: 0 0 255
            green: 0 255 0
            blue: 255 0 0
            yellow: 0 255 255
            white: 255 255 255
            black: 0 0 0

            Available brigthness:
            50
            -50

            Output format MUST be exactly:
            [Response to user text]
            [command_name] [position (optional) OR color (optional)]
            
            User: "please zoom in top left"
            Response: Okay! Zooming in top left.
            zoom_in top_left

            User: "please zoom in the middle"
            Response: Okay! Zooming in the center.
            zoom_in center

            User: "Color every cell in blue"
            Response: Coloring cell segmentation in blue...
            cell_segmentation 255 0 0

            User: "Recolor every cell in red"
            Response: recoloring cell segmentation in red...
            cell_segmentation 0 0 255
            
            User: "please zoom out"
            Response: Okay! Zooming out.
            zoom_out

            User: "please brighten the image"
            Response: Sure! Making the image brighter.
            lighting_modification 50

            User: "please darken the image"
            Response: Sure! Making the image darker.
            lighting_modification -50

            User: "how's the weather?"
            Response: Sorry, I can only help with software commands.
            none

            User: "quit"
            Response: Thank you for using our software! Have a nice day.
            quit

            IMPORTANT:  
            - The LAST line must contain ONLY the command and optional position or optional color, nothing else.  
            - Do NOT output any extra blank lines or trailing spaces after the command line."""
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
            "quit": self.quit
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
                print(parts)
                if len(parts) == 2:
                    if parts[1] in ['top_right', 'top_left', 'bottom_left', 'bottom_right', 'center']:
                        self.command, self.zoom_position = parts[0], parts[1]
                    #elif parts[1] in ['up', 'down', 'left', 'right']:
                        #self.command, self.move_position = parts[0], parts[1]
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



    def cell_segmentation(self):
        
        self.image = self.original_image.copy()

        model = models.CellposeModel(gpu=True)
        masks, flows, styles = model.eval(self.image, diameter=30)

        image_BGR = self.image.copy()

        binary_mask = (masks > 0).astype(np.uint8)

        overlay_color = tuple(int(c) for c in self.color)

        alpha = 0.5

        # For each channel, blend overlay_color onto image where mask is 1 (Chatgpt)
        for c in range(3):
            image_BGR[:, :, c] = np.where(
                binary_mask == 1,
                (alpha * overlay_color[c] + (1 - alpha) * image_BGR[:, :, c]).astype(np.uint8),
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
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()