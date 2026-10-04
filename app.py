import base64
import pickle
import cv2
import mediapipe as mp
import numpy as np
import warnings
from flask import Flask, render_template
from flask_socketio import SocketIO, emit

# Suppress specific warnings
warnings.filterwarnings("ignore", message="SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.")

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load model
try:
    model_dict = pickle.load(open('./model.p', 'rb'))
    model = model_dict['model']
except Exception as e:
    print("Error loading the model:", e)
    model = None

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Using static_image_mode=True is safer for web concurrency as it treats each frame independently without tracking state across frames.
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S',
    19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'Hello',
    27: 'Done', 28: 'Thank You', 29: 'I Love you', 30: 'Sorry', 31: 'Please',
    32: 'You are welcome.'
}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('image')
def handle_image(data_url):
    try:
        # Decode base64 image from client
        encoded_data = data_url.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        frame = cv2.flip(frame, 1)  # Flip the frame horizontally

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)
        
        # We need this try block to avoid failing if no hands are detected.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                x_ = []
                y_ = []
                data_aux = []

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                if model is not None:
                    try:
                        # Ensure data_aux has exactly the number of features the model expects
                        # Sometimes hands.process might give different shapes if multiple hands, but here we process each hand.
                        prediction = model.predict([np.asarray(data_aux)])
                        prediction_proba = model.predict_proba([np.asarray(data_aux)])
                        confidence = max(prediction_proba[0])
                        predicted_character = labels_dict[int(prediction[0])]
                        
                        emit('prediction', {'text': predicted_character, 'confidence': float(confidence)})
                        
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                        cv2.putText(frame, f"{predicted_character} ({confidence*100:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
                    except Exception as e:
                        print("Prediction Error:", e)

        # Encode back to base64 to send to client
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')
        emit('processed_image', 'data:image/jpeg;base64,' + frame_encoded)

    except Exception as e:
        print("Error processing image:", e)

if __name__ == '__main__':
    socketio.run(app, debug=True)
