import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
drawing = False
prev_x, prev_y = None, None
canvas = None
color = (255, 0, 0)  # Start with blue
colors = [(255,0,0), (0,255,0), (0,0,255), (0,0,0)]  # Blue, Green, Red, Black
color_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, _ = frame.shape
            ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
            tx, ty = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)
            dist = np.hypot(ix - tx, iy - ty)
            # Draw when index and thumb are apart
            if dist > 40:
                if prev_x is not None and prev_y is not None:
                    cv2.line(canvas, (prev_x, prev_y), (ix, iy), color, 8)
                prev_x, prev_y = ix, iy
                drawing = True
            else:
                prev_x, prev_y = None, None
                drawing = False
            # Change color gesture (pinch index and thumb, then move up)
            if dist < 40 and iy < h//4:
                color_idx = (color_idx + 1) % len(colors)
                color = colors[color_idx]
                cv2.waitKey(300)  # Debounce
            # Clear canvas gesture (pinch index and thumb, then move down)
            if dist < 40 and iy > 3*h//4:
                canvas = np.zeros_like(frame)
                cv2.waitKey(300)  # Debounce
            cv2.circle(frame, (ix, iy), 10, color, -1)
    # Overlay the canvas
    frame = cv2.addWeighted(frame, 0.5, canvas, 0.5, 0)
    cv2.putText(frame, f'Color: {color}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow('Gesture Drawing', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
hands.close()
cv2.destroyAllWindows()