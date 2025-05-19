import cv2
import mediapipe as mp
import keyboard
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
prev_x, prev_y = None, None
threshold_x = 30
threshold_y = 15
cooldown = 0.25  # seconds
last_action_time = time.time()

no_hand_frames = 0
no_hand_reset = 5  # Number of frames to wait before resetting prev_x, prev_y

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    gesture = None
    if results.multi_hand_landmarks:
        no_hand_frames = 0
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            x = int(hand_landmarks.landmark[8].x * w)
            y = int(hand_landmarks.landmark[8].y * h)
            if prev_x is not None and prev_y is not None:
                dx = x - prev_x
                dy = y - prev_y
                if abs(dx) > abs(dy):
                    if dx > threshold_x:
                        gesture = "Right"
                    elif dx < -threshold_x:
                        gesture = "Left"
                else:
                    if dy > threshold_y:
                        gesture = "Down"
                    elif dy < -threshold_y:
                        gesture = "Up"
            prev_x, prev_y = x, y
    else:
        no_hand_frames += 1
        if no_hand_frames >= no_hand_reset:
            prev_x, prev_y = None, None
            no_hand_frames = 0
    if gesture and (time.time() - last_action_time > cooldown):
        print(f"Gesture detected: {gesture} (sending key)")
        cv2.putText(frame, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        if gesture == "Left":
            keyboard.press_and_release('left')
        elif gesture == "Right":
            keyboard.press_and_release('right')
        elif gesture == "Up":
            keyboard.press_and_release('up')
        elif gesture == "Down":
            keyboard.press_and_release('down')
        last_action_time = time.time()
    cv2.imshow('Gesture to Keyboard', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()