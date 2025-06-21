import requests
import json
import speech_recognition as sr
import pyttsx3
engine = pyttsx3.init()

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

url = 'http://localhost:11434/api/chat'
r = sr.Recognizer()

with sr.Microphone() as source:
    print("You can talk now")
    prompt = r.listen(source)


    try:
        user_input = r.recognize_google(prompt)
        print("You said:", user_input)


        payload = {
            'model' : 'llama3.2',
            'messages' : [{"role": "user", "content": user_input}]
        }

        response = requests.post(url, json=payload, stream=True)
        response_text = ""
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    json_data = json.loads(line)
                    if "message" in json_data and "content" in json_data["message"]:
                        content = json_data["message"]["content"]
                        print(content, end='', flush=True)
                        response_text += content
            engine.say(response_text)
            engine.runAndWait()
        else:
            print("Error")
    except sr.UnknownValueError:
        print("Sorry, I could not understand your voice.")
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
