import torch
import whisper
import speech_recognition as sr
from tempfile import NamedTemporaryFile

from queue import Queue
import io

from hugchat import hugchat
from hugchat.login import Login

from elevenlabs import generate, stream, set_api_key

import os
from dotenv import load_dotenv

load_dotenv()
set_api_key(os.getenv('ELEVEN_API'))

def listen(energy_threshold: int = 1000, model: str = 'medium', record_timeout: float = 10) -> str:
    recorder = sr.Recognizer()
    recorder.energy_threshold = energy_threshold
    recorder.dynamic_energy_threshold = False

    data_queue = Queue()

    source = sr.Microphone(sample_rate=16000)

    print('[INFO] - listen : Audio setup complete.')

    audio_model = whisper.load_model(model)

    print('[INFO] - listen : Whisper model loaded.')

    temp_file = NamedTemporaryFile().name

    with source:
        recorder.adjust_for_ambient_noise(source)
    
    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        data_queue.put(data)
    
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    print('[INFO] - listen : Listening in background.')

    print('[INFO] - listen : Transcription started.')
    sample = bytes()
    while True:
        try:
            if not data_queue.empty():
                while not data_queue.empty():
                    data = data_queue.get()
                    sample += data
        except:
            break

    audio_data = sr.AudioData(sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
    wav_data = io.BytesIO(audio_data.get_wav_data())

    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())

    # Read the transcription.
    result = audio_model.transcribe(temp_file, fp16=torch.cuda.is_available())
    text = result['text'].strip()
    print('[INFO] - listen : Transcribed.')
    print(f'[DEBUG] - listen : TRANSCRIPTION\n\n"{text}"')

    return text

def chat(input: str, email: str, password: str):
    sign = Login(email, password)
    cookies = sign.login()

    print('[INFO] - chat : Signed into huggingface chat.')

    chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), default_llm=3) # Use falcon-180B

    print('[INFO] - chat : Prepared Falcon-180B chatbot.')
    
    query_result = chatbot.query(input)
    response = query_result['text'].strip()

    print('[INFO] - chat : Received response.')
    print(f'[DEBUG] - chat : RESPONSE\n\n"{response}"')

    return response


def talk(text: str):
    audio = generate(
        text=text,
        voice="Bella",
        model='eleven_monolingual_v1',
        stream=True
    )
    print('[INFO] - talk : Generated audio response.')

    stream(audio)
    print('[INFO] - talk : Playing audio response.')

if __name__ == "__main__":
    email = os.getenv('HUG_EMAIL')
    password = os.getenv('HUG_PASSWORD')

    input = listen(model='tiny')
    output = chat(input, email, password)
    talk(output)