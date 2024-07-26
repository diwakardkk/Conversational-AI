# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 09:20:37 2024

@author: win 10
"""



#================
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
import tempfile

import tkinter as tk
import openai
import speech_recognition as sr
from threading import Thread
from gtts import gTTS
from playsound import playsound
import tempfile
import os

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Flag to control the listening loop
listening = False

# Set your OpenAI API key directly
openai.api_key = 'YOUR-API-KEY'

# Define keywords for each class
class_keywords = {
    "Anxiety": [
        "worry", "fear", "nervous", "anxious", "panic", "anxiety", "stressed", "tense", "overwhelmed", "uneasy", "restless",
        "frightened", "scared", "dread", "tension", "phobia", "apprehensive", "worrying", "fearful", "panic attack",
        "anxiousness", "worried", "nervousness", "edgy", "jittery", "alarm", "concern", "distress", "agitation", "uneasiness",
        "fretful", "fear of", "afraid"
    ],
    "BPD": [
        "mood swings", "impulsive", "relationships", "abandonment", "empty", "unstable", "borderline", "self-image",
        "identity", "impulsivity", "intense", "fluctuating", "emotional instability", "fear of being alone", "splitting",
        "rapid changes", "self-harm", "suicidal", "efforts to avoid", "anger", "emptiness", "dissociation", "impulse control",
        "black and white thinking", "identity disturbance", "chronic feelings", "emotional dysregulation"
    ],
    "Depression": [
        "sad", "hopeless", "fatigue", "worthless", "interest lost", "depressed", "depression", "unhappy", "gloomy", "down",
        "melancholy", "despair", "low mood", "disinterest", "apathy", "despondent", "discouraged", "dismal", "suicidal thoughts",
        "sleep problems", "weight changes", "appetite changes", "energy loss", "self-loathing", "guilt", "concentration problems",
        "feelings of worthlessness", "hopelessness", "anhedonia", "sadness", "emptiness"
    ],
    "Bipolar": [
        "mood swings", "ups and downs", "rollercoaster emotions", "feeling on top of the world", "crashing lows",
        "unpredictable moods", "emotional rollercoaster", "highs and lows", "feeling wired", "feeling drained",
        "going through phases", "feeling like a different person", "riding the wave of emotions", "extreme mood changes",
        "having good days and bad days", "being on cloud nine", "feeling down in the dumps", "wild mood swings"
    ]
}

# Function to send request to GPT
def send_request_to_gpt(text):
    try:
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:bba-university:mental-disorder:8ebDCFla",
            messages=[{"role": "user", "content": text}],
            temperature=0.8,  # Adjust for randomness
            max_tokens=150,    # Maximum length of the response
            top_p=1.0,         # Nucleus sampling
            frequency_penalty=0.8  # Reduces repetition
        )
        return response.choices[0].message['content']
    except Exception as e:
        return str(e)

# Split the questions into initial and class-specific groups
initial_questions = [
    "May I ask your gender? and how old you are?",
    "Can you describe your emotional state over the past few weeks, including any specific moments or experiences that stood out to you?",
    "What things have been making you worried or stressed? How have you been dealing with these feelings?",
    "How have your relationships with friends, family, or partners been affecting your life and feelings recently?",
    "Are you seeking support to improve your well-being, understand yourself better, or navigate life challenges?"
]

# Define class-specific questions
class_specific_questions = {
    "Anxiety": [
        "Can you describe a time when you felt particularly worried or anxious? What was happening, and how did it make you feel?",
        "How does worrying affect your day-to-day life? Can you give examples of things you might avoid or struggle with because of anxiety?",
        "Have you ever experienced a panic attack or a moment of intense fear? What was that like for you?",
        "In what ways do you try to cope with restlessness or nervousness? What helps you feel more relaxed?",
        "Can you talk about how your anxiety has impacted your relationships or activities you used to enjoy?"
    ],
    "BPD": [
        "Could you share an experience where your mood changed very quickly? How did those mood swings affect your day or interactions with others?",
        "Have you ever felt a strong fear of being left alone or abandoned? How does this fear manifest in your relationships or behaviors?",
        "Can you describe a relationship that has been challenging for you? What patterns have you noticed in how you relate to others?",
        "Are there times when you've acted impulsively or done something you later regretted? What led to those actions, and how did you feel afterward?",
        "How do you deal with feelings of emptiness or boredom? Are there activities or thoughts that help fill that void?"
    ],
    "Depression": [
        "Can you tell me about a period when you felt particularly down or hopeless? What was going on in your life, and how did you cope with those feelings?",
        "Have your interests or hobbies changed due to feeling depressed? What activities have you stopped enjoying, and why?",
        "How has your appetite or sleep been affected by your mood? Can you describe any specific changes and how they impact your daily life?",
        "In moments of low self-esteem or guilt, what are the thoughts that go through your mind? Can you share an example?",
        "What strategies or support have you found helpful in managing your depression? Are there certain things that make you feel a bit better?"
    ],
    "Bipolar Disorder": [
        "During a high-energy phase, what kinds of activities do you find yourself drawn to? Can you describe a time when you felt on top of the world?",
        "How do your thoughts and speech change during these energetic periods? Can you give an example of an experience where this was noticeable?",
        "Have you had moments of making decisions that seemed great at the time but were problematic later? What happened, and how did you feel about it afterward?",
        "Can you share your experience with the lows as well? What does it feel like when you're in a depressive phase?",
        "How do the changes in your mood affect your life, work, or relationships? Can you describe how you navigate these ups and downs?"
    ]
}
# Initialize counters and sets for questions asked
sequential_questions_asked = 0
asked_questions = set()



def text_to_speech(text):
    """Converts text to speech and plays it using pydub and simpleaudio."""
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(temp_file.name)
        temp_file.close()  # Close the file to ensure it's written

        # Load the temporary mp3 file and play it
        audio = AudioSegment.from_mp3(temp_file.name)
        play(audio)

        # Cleanup
        os.unlink(temp_file.name)
    except Exception as e:
        print(f"Error converting text to speech: {e}")

# Function to handle user submission
def on_submit():
    global sequential_questions_asked, asked_questions

    user_input = user_input_box.get("1.0", "end-1c").strip()
    user_input_box.delete("1.0", tk.END)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "You: " + user_input + "\n")
    
    response = send_request_to_gpt(user_input)

    # Check if response doesn't end with '?' or '..'
    if not response.endswith('?') and not response.endswith('..'):
        # If initial five questions haven't been asked, ask them
        if sequential_questions_asked < len(initial_questions):
            question_to_ask = initial_questions[sequential_questions_asked]
            response += " " + question_to_ask
            sequential_questions_asked += 1
            asked_questions.add(question_to_ask)
        else:
            # Logic for handling responses after initial questions
            initial_responses = [question.split(': ')[1].strip() for question in chat_history.get("1.0", tk.END).split('You: ')[1:6]]

            # Determine the class based on keywords in the initial responses
            matched_class = None
            for class_name, keywords in class_keywords.items():
                for keyword in keywords:
                    for initial_response in initial_responses:
                        if keyword in initial_response:
                            matched_class = class_name
                            break
                    if matched_class:
                        break
                if matched_class:
                    break

            # Ask class-specific questions based on the matched class
            if matched_class:
                for question in class_specific_questions[matched_class]:
                    if question not in asked_questions:
                        response += " " + question
                        asked_questions.add(question)
                        break
            else:
                # If no class is matched, ask random questions from other classes
                for class_questions in class_specific_questions.values():
                    for question in class_questions:
                        if question not in asked_questions:
                            response += " " + question
                            asked_questions.add(question)
                            break
                    else:
                        continue
                    break

    chat_history.insert(tk.END, "Bot: " + response + "\n")
    text_to_speech(response)  # Convert bot response to speech

    chat_history.config(state=tk.DISABLED)

from tkinter import messagebox

def save_conversation():
    response = messagebox.askyesno("Save Conversation", "Do you want to save the conversation and exit?")
    if response:  # User clicked yes
        conversation_text = chat_history.get("1.0", tk.END)
        # Ensure we're not trying to save an empty conversation or just the newline at the end
        if conversation_text.strip() != "":
            try:
                with open("Chat_conversation.txt", "w", encoding='utf-8') as file:
                    file.write(conversation_text)
                print("Conversation saved successfully.")
            except Exception as e:
                print(f"Error saving conversation: {e}")
        else:
            print("No conversation to save.")
        root.after(0, lambda: [root.quit(), root.destroy()])

def recognize_speech_continuous():
    global listening
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        while listening:
            try:
                audio = recognizer.listen(source, timeout=5)
                recognized_text = recognizer.recognize_google(audio)
                print("Recognized: " + recognized_text)
                user_input_box.delete("1.0", tk.END)
                user_input_box.insert(tk.END, recognized_text)
                on_submit()
            except sr.WaitTimeoutError:
                print("Listening... (timeout)")
            except Exception as e:
                print("Error: ", str(e))

def toggle_listening():
    global listening
    listening = not listening
    if listening:
        audio_button.config(text="Stop Listening")
        Thread(target=recognize_speech_continuous, daemon=True).start()
    else:
        audio_button.config(text="Start Listening")

# Create the main window
root = tk.Tk()
root.title("ChatGPT Interface")

# Chat history textbox
chat_history = tk.Text(root, height=20)
chat_history.pack()
chat_history.config(state=tk.DISABLED)

# User input textbox
user_input_box = tk.Text(root, height=2)
user_input_box.pack()

# Submit button
submit_button = tk.Button(root, text="Send", command=on_submit)
submit_button.pack()

# Save conversation button
save_button = tk.Button(root, text="Submit Report", command=save_conversation)
save_button.pack()


# Button to start/stop audio input
audio_button = tk.Button(root, text="Start Listening", command=toggle_listening)
audio_button.pack()

# Run the application
root.mainloop()
