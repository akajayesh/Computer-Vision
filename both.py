import cv2
import mediapipe as mp
import keyboard
import time
import numpy as np
import threading
import winsound

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

threshold = 30
cooldown = 0.4
last_action_time = [0, 0]
prev_coords = [None, None]
gesture_hold_frames = [0, 0]
hold_required = 4
current_gesture = [None, None]

# Left hand: WASD, Right hand: Arrow keys
HAND_KEY_MAP = [
    {"Left": "a", "Right": "d", "Up": "w", "Down": "s"},
    {"Left": "left", "Right": "right", "Up": "up", "Down": "down"}
]
HAND_COLOR = [
    (255, 0, 0),  # Left hand: blue
    (0, 255, 0)   # Right hand: green
]

def play_beep():
    threading.Thread(target=lambda: winsound.Beep(1200, 100)).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gestures = [None, None]
    if results.multi_hand_landmarks and results.multi_handedness:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            if prev_coords[idx] is not None:
                dx = x - prev_coords[idx][0]
                dy = y - prev_coords[idx][1]
                if abs(dx) > abs(dy):
                    if dx > threshold:
                        gestures[idx] = "Right"
                    elif dx < -threshold:
                        gestures[idx] = "Left"
                else:
                    if dy > threshold:
                        gestures[idx] = "Down"
                    elif dy < -threshold:
                        gestures[idx] = "Up"
            prev_coords[idx] = (x, y)
            # Gesture hold logic
            if gestures[idx] == current_gesture[idx] and gestures[idx] is not None:
                gesture_hold_frames[idx] += 1
            else:
                gesture_hold_frames[idx] = 1 if gestures[idx] else 0
                current_gesture[idx] = gestures[idx]
            # Only trigger if gesture is held for required frames
            if gestures[idx] and gesture_hold_frames[idx] >= hold_required and (time.time() - last_action_time[idx] > cooldown):
                key = HAND_KEY_MAP[idx].get(gestures[idx])
                if key:
                    keyboard.press_and_release(key)
                    play_beep()
                last_action_time[idx] = time.time()
                cv2.putText(frame, f"{gestures[idx]} (sent {key})", (10, 50 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, HAND_COLOR[idx], 2)
            elif gestures[idx]:
                cv2.putText(frame, f"{gestures[idx]}", (10, 50 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, HAND_COLOR[idx], 2)
            else:
                cv2.putText(frame, "No gesture", (10, 50 + idx*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (128,128,128), 2)
    else:
        prev_coords = [None, None]
        gesture_hold_frames = [0, 0]
        current_gesture = [None, None]
    cv2.imshow('Both Hands Gesture Control', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()