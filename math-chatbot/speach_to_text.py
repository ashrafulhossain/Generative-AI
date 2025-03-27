import os
import time
import keyboard
import numpy as np
from dotenv import load_dotenv
from deepgram import Deepgram
from openai import OpenAI
import asyncio
import aiofiles
import sounddevice as sd
from scipy.io.wavfile import write

# ✅ Load API keys from .env
load_dotenv()
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ✅ Initialize APIs
deepgram = Deepgram(DEEPGRAM_API_KEY)
client = OpenAI(api_key=OPENAI_API_KEY)

# 🧠 Memory for context with Theta-wise format instruction
chat_history = [
    {
        "role": "system",
        "content": (
            "You are a helpful and expert math tutor. "
            "Always solve problems using a clear Theta-wise format like:\n"
            "Θ Step 1: ...\n"
            "Θ Step 2: ...\n"
            "End with ✅ Final Answer: ...\n"
            "Make each explanation simple, logical, and beginner-friendly."
        )
    }
]

# 📌 STEP 0: Record voice (start/stop with Enter key)
def record_voice_dynamic(filename='sample.wav', fs=44100):
    print("🎙️ Press [Enter] to START recording...")
    keyboard.wait('enter')

    print("🎤 Initializing microphone...")
    time.sleep(0.5)  # Allow mic to warm up

    stream = sd.InputStream(samplerate=fs, channels=1, dtype='int16')
    stream.start()
    print("🎤 Recording... Press [Enter] again to STOP.")

    frames = []

    while True:
        data, _ = stream.read(1024)
        frames.append(data)

        if keyboard.is_pressed('enter'):
            print("⏳ Waiting for key release...")
            while keyboard.is_pressed('enter'):
                time.sleep(0.1)
            break

    stream.stop()
    print(f"🛑 Recording stopped. Collected {len(frames)} frames.")

    if len(frames) < 10:
        print("⚠️ Not enough voice data captured. Try speaking for a bit longer.")
        return False

    audio_data = np.concatenate(frames, axis=0)
    write(filename, fs, audio_data)
    print(f"✅ Voice saved to {filename}")
    return True

# 📌 STEP 1: Transcribe with Deepgram
async def transcribe_audio(file_path):
    async with aiofiles.open(file_path, 'rb') as audio:
        source = {
            'buffer': await audio.read(),
            'mimetype': 'audio/wav'
        }
        response = await deepgram.transcription.prerecorded(source, {'model': 'nova'})
        return response['results']['channels'][0]['alternatives'][0]['transcript']

# 📌 STEP 2: Solve math with GPT-4 (with Theta-wise formatting)
def solve_math_with_gpt(prompt):
    # Format the user prompt to explicitly request Theta-wise step-by-step solution
    formatted_prompt = f"Please solve this math problem step by step using Theta-wise format:\n\n{prompt}"

    # Add user message to chat history
    chat_history.append({"role": "user", "content": formatted_prompt})

    # Keep memory within limit: system + 10 user-assistant pairs
    max_messages = 21
    if len(chat_history) > max_messages:
        chat_history[:] = [chat_history[0]] + chat_history[-(max_messages - 1):]

    # Call GPT-4 with memory
    response = client.chat.completions.create(
        model="gpt-4",
        messages=chat_history
    )

    answer = response.choices[0].message.content

    # Add assistant reply to chat history
    chat_history.append({"role": "assistant", "content": answer})

    return answer

# 📌 MAIN Chat Loop
async def main():
    print("🧠 Voice ChatGPT Math Solver (Theta-wise Format) Ready")
    print("🎙️ Press [Enter] to speak. Press [Ctrl+C] to exit.\n")

    while True:
        audio_file = 'sample.wav'
        success = record_voice_dynamic(audio_file)

        if not success:
            print("🔁 Recording failed or was too short. Try again.\n")
            continue

        print("🎧 Transcribing...")
        transcript = await transcribe_audio(audio_file)
        print(f"🗣️ You said: {transcript}")

        if not transcript.strip():
            print("⚠️ No speech detected. Please try again.\n")
            continue

        print("🧮 Solving with GPT-4...")
        solution = solve_math_with_gpt(transcript)
        print(f"✅ GPT-4 Answer:\n{solution}\n")

# ✅ Run the app
if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Chat Ended. See you next time!")









