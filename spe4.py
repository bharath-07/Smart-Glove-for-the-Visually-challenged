
import speech_recognition as sr
from ibm_watson import SpeechToTextV1
import json
r = sr.Recognizer()
speech = sr.Microphone()
speech_to_text = SpeechToTextV1(
    iam_apikey = "YOUR_API_KEY",
    url = "YOUR_URL"
)
with speech as source:
    print("say something!!…")
    audio_file = r.adjust_for_ambient_noise(source)
    audio_file = r.listen(source)
speech_recognition_results = speech_to_text.recognize(audio=audio_file.get_wav_data(), content_type='audio/wav').get_result()
print(json.dumps(speech_recognition_results, indent=2))
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
import speech_recognition as sr
from ibm_watson import SpeechToTextV1
import json
r = sr.Recognizer()
speech = sr.Microphone()
speech_to_text = SpeechToTextV1(
    iam_apikey = "Jk6yO5DQobWReM6wR24ouPWUEJXe0PwMQm4yaXAosNDP",
    url = "https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/0e3ee591-4cff-4ca4-bc86-c19c057acafd"
)
with speech as source:
    print("say something!!…")
    audio_file = r.adjust_for_ambient_noise(source)
    audio_file = r.listen(source)
speech_recognition_results = speech_to_text.recognize(audio=audio_file.get_wav_data(), content_type='audio/wav').get_result()
print(json.dumps(speech_recognition_results, indent=2))
