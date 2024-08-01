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
from tkinter import messagebox
import tensorflow as tf
from transformers import AutoTokenizer
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize the speech recognizer
recognizer = sr.Recognizer()

# Flag to control the listening loop
listening = False

# Set your OpenAI API key directly
openai.api_key = 'API KEY'

# Define the classes you're interested in
classes = ['BPD', 'bipolar', 'depression', 'Anxiety']

# Load the saved model and tokenizer
model_path = ".../save/path_to_save_model_directory"
tokenizer_path = ".../save/path_to_save_tokenizer_directory"


# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


# Function to load and preprocess the conversation text
def load_and_preprocess_conversation(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        conversation_text = file.read()
    return conversation_text

# Function to predict the class of the conversation
def predict_conversation_class(conversation_text):
    # Tokenize the input text
    inputs = tokenizer(conversation_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_index = torch.argmax(logits, dim=1).item()
    
    # Map the predicted index to the class name
    predicted_class = classes[predicted_index]
    
    return predicted_class

# Function to load and preprocess the conversation text
def load_and_preprocess_conversation(file_path):
    with open(file_path, "r", encoding='utf-8') as file:
        conversation_text = file.read()
    return conversation_text

# Function to predict the class of the conversation
def predict_conversation_class(conversation_text):
    # Tokenize the input text
    inputs = tokenizer(conversation_text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    
    logits = outputs.logits
    predicted_index = torch.argmax(logits, dim=1).item()
    
    # Map the predicted index to the class name
    predicted_class = classes[predicted_index]
    
    return predicted_class
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


# List of end conversation phrases
end_conversation_phrases = [
    "bye", "see you later", "goodbye", "au revoir", "sayonara", "ok bye",
    "bye then", "fare thee well", "see you later.", "have a nice day.",
    "bye! come back again.", "i'll see you soon."
]
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

def on_submit():
    global sequential_questions_asked, asked_questions

    user_input = user_input_box.get("1.0", "end-1c").strip()
    user_input_box.delete("1.0", tk.END)
    chat_history.config(state=tk.NORMAL)
    chat_history.insert(tk.END, "You: " + user_input + "\n")
    
    response = send_request_to_gpt(user_input).strip().lower()
    #chat_history.insert(tk.END, "Bot: " + response + "\n")
    #text_to_speech(response)  # Convert bot response to speech

    # Check if response indicates the end of the conversation
    if any(phrase in response for phrase in end_conversation_phrases):
        chat_history.config(state=tk.DISABLED)
        user_input_box.config(state=tk.DISABLED)  # Disable user input box
        submit_button.config(state=tk.DISABLED)  # Disable submit button
        audio_button.config(state=tk.DISABLED)  # Disable audio button

        
        detect_disease()  # to predict the disease
        return
    
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

def detect_disease():
    # Show the save prompt
    response = messagebox.askyesno("Check final report", "Do you want to submit the conversation?")
    if response:  # User clicked yes
        # Get conversation text directly from the chat history
        conversation_text = chat_history.get("1.0", tk.END).strip()
        #conversation_text = load_and_preprocess_conversation(conversation_text)
        # Ensure we're not trying to predict a empty conversation or just the newline at the end
        if len(conversation_text) > 0:
            try:
                              
                predicted_disease = predict_conversation_class(conversation_text)  # Replace with your actual logic
                messagebox.showinfo("Diagnosis Result", f"The predicted disease is: {predicted_disease}")
                
                thank_you_window = tk.Toplevel(root)
                thank_you_window.title("Thank You")
                tk.Label(thank_you_window, text="Thank you for using the application!").pack(pady=10)
                tk.Button(thank_you_window, text="Exit", command=lambda: [root.quit(), root.destroy()]).pack(pady=5)
            except Exception as e:
                messagebox.showerror("Error", f"Error processing conversation: {e}")
        else:
            messagebox.showwarning("Warning", "The conversation is empty. Nothing to submit.")

 

import tkinter as tk
from tkinter import ttk

# Function to clear the placeholder text on focus
def clear_placeholder(event):
    if user_input_box.get("1.0", tk.END).strip() == "Start conversation...":
        user_input_box.delete("1.0", tk.END)
        user_input_box.config(fg='black')

# Function to restore the placeholder text if the box is empty
def restore_placeholder(event):
    if not user_input_box.get("1.0", tk.END).strip():
        user_input_box.insert("1.0", "Start conversation...")
        user_input_box.config(fg='grey')

# Create the main window
root = tk.Tk()
root.title("Mental Health Assistant")
root.geometry("500x600")
root.configure(bg='white')  # Set background color to white

# Create a style object for ttk
style = ttk.Style()

# Define a style for the buttons with a green background and black text
style.configure(
    'TButton',
    background='green',
    foreground='black',
    font=('Arial', 10, 'bold'),
    padding=10,
    relief='flat'
)

# Style map for active and pressed states
style.map(
    'TButton',
    foreground=[('pressed', 'black'), ('active', 'black')],
    background=[('pressed', '!disabled', '#5cb85c'), ('active', '#5cb85c')],
    relief=[('pressed', 'sunken'), ('!pressed', 'flat')]
)

# Define custom layout for rounded corners (if desired)
style.layout("RoundedButton",
             [('Button.border', {'children': [('Button.padding', {'children': [('Button.label', {'sticky': 'nswe'})], 'sticky': 'nswe'})], 'sticky': 'nswe'})])
style.configure("RoundedButton", padding=6, relief="flat", borderwidth=0, background="green", foreground="black")
style.map("RoundedButton", background=[('active', '#5cb85c')])

# Chat history textbox
chat_history = tk.Text(root, height=20, state=tk.DISABLED, wrap=tk.WORD, bg='#ffffff', fg='#000000', font=('Arial', 10))
chat_history.pack(padx=10, pady=10)

# User input textbox
user_input_box = tk.Text(root, height=2, wrap=tk.WORD, fg='grey', font=('Arial', 10))
user_input_box.insert("1.0", "Start conversation...")
user_input_box.bind("<FocusIn>", clear_placeholder)
user_input_box.bind("<FocusOut>", restore_placeholder)
user_input_box.pack(padx=10, pady=10)

# Style buttons
button_style = {'bg': 'green', 'fg': 'black', 'font': ('Arial', 10, 'bold')}

# Submit button
submit_button = tk.Button(root, text="Send", command=on_submit, **button_style)
submit_button.pack(padx=10, pady=5)

# Report  button
save_button = tk.Button(root, text="Check Report", command=detect_disease, **button_style)
save_button.pack(padx=10, pady=5)

# Button to start/stop audio input
audio_button = tk.Button(root, text="Start Listening", command=toggle_listening, **button_style)
audio_button.pack(padx=10, pady=5)

# Run the application
root.mainloop()
