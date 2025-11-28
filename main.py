from keras.models import load_model
import cv2
import numpy as np
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Style, init
import random
import os

init(autoreset=True)


# model = load_model('./models/model_full.h5')
model = load_model('./models/emotion_model.keras')

emotions = ['Angry 😠', 'Disgust 🤢', 'Fear 😨', 'Happy 😊', 'Neutral 😐', 'Sad 😢', 'Surprise 😲']
emotion_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

emotion_colors = {
    0: Fore.RED,        # Angry
    1: Fore.GREEN,      # Disgust
    2: Fore.MAGENTA,    # Fear
    3: Fore.YELLOW,     # Happy
    4: Fore.CYAN,       # Neutral
    5: Fore.BLUE,       # Sad
    6: Fore.LIGHTMAGENTA_EX  # Surprise
}

quotes = {
    0: [  # Angry
        "Take a deep breath, it's not worth it.",
        "Anger is one letter short of danger.",
        "Maybe try counting to 10... or 100.",
    ],
    1: [  # Disgust
        "That bad, huh?",
        "We've all been there.",
        "Time to look away maybe?",
    ],
    2: [  # Fear
        "Everything's going to be okay.",
        "You're braver than you think.",
        "Face your fears, they're usually smaller up close.",
    ],
    3: [  # Happy
        "Keep that smile going!",
        "Happiness looks good on you.",
        "Now that's the energy we need!",
    ],
    4: [  # Neutral
        "Living life on autopilot today?",
        "The calm before the storm, or just calm?",
        "Keeping it cool, I see.",
    ],
    5: [  # Sad
        "This too shall pass.",
        "Better days are coming.",
        "It's okay to not be okay sometimes.",
    ],
    6: [  # Surprise
        "Didn't see that coming, did you?",
        "Life's full of surprises!",
        "Plot twist moment right there.",
    ]
}

def print_large(text, color):
    lines = [
        f" ███  {text}",
    ]
    for line in lines:
        print(color + line + Style.RESET_ALL)

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')
    print("\nEmotion Detection - Press SPACE to detect, C to clear, Q to quit\n")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

print("\nEmotion Detection - Press SPACE to detect, C to clear, Q to quit\n")

current_predictions = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    
    for idx, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        if idx in current_predictions:
            top3 = current_predictions[idx]
            y_offset = y - 15
            
            for i, (emo_idx, conf) in enumerate(top3):
                text = f"{emotion_names[emo_idx]}: {conf:.1f}%"
                cv2.putText(frame, text, (x, y_offset - i*25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == 32 and len(faces) > 0:
        timestamp = datetime.now().strftime('%H:%M:%S')
        current_predictions = {}
        
        for idx, (x, y, w, h) in enumerate(faces):
            face = gray[y:y+h, x:x+w]
            face = cv2.resize(face, (48, 48))
            face = face.astype('float32') / 255.0
            face = np.expand_dims(np.expand_dims(face, -1), 0)
            
            prediction = model.predict(face, verbose=0)[0]
            detected_emotion = np.argmax(prediction)
            
            top3_indices = np.argsort(prediction)[-3:][::-1]
            current_predictions[idx] = [(i, prediction[i]*100) for i in top3_indices]
            
            results = [[emotions[i], f"{prediction[i]*100:.2f}%"] for i in range(7)]
            results = sorted(results, key=lambda x: float(x[1].strip('%')), reverse=True)
            
            print(f"\n[{timestamp}] Detection Results:")
            print(tabulate(results, headers=['Emotion', 'Confidence'], tablefmt='grid'))
            
            quote = random.choice(quotes[detected_emotion])
            color = emotion_colors[detected_emotion]
            print()
            print_large(quote, color)
    
    if key == ord('c'):
        current_predictions = {}
        clear_terminal()
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()