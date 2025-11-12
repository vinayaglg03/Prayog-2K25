"""
Gestures:
👍  Thumbs Up         → Scroll Up
👎  Thumbs Down       → Scroll Down
☝️  Index Finger Only → Select / Click
✌️  Index + Middle    → Move Cursor
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
from collections import deque

# Initialize
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

prev_x, prev_y = 0, 0
smoothening = 6  # Base smooth factor

# Store recent cursor points for averaging
pts_x, pts_y = deque(maxlen=5), deque(maxlen=5)

# ───── Hand Tracking ─────
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                lm = hand_landmarks.landmark

                # Get key points
                index_tip = (int(lm[8].x * w), int(lm[8].y * h))
                middle_tip = (int(lm[12].x * w), int(lm[12].y * h))
                thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
                wrist = (int(lm[0].x * w), int(lm[0].y * h))

                # Determine which fingers are up
                index_up = lm[8].y < lm[6].y
                middle_up = lm[12].y < lm[10].y
                ring_up = lm[16].y < lm[14].y
                pinky_up = lm[20].y < lm[18].y

                # Detect thumb up/down
                thumb_up = thumb_tip[1] < wrist[1] - 40
                thumb_down = thumb_tip[1] > wrist[1] + 40

                # ───── Gesture Logic ─────

                # 👍 Scroll Up
                if thumb_up and not index_up and not middle_up:
                    pyautogui.scroll(300)
                    cv2.putText(frame, "Scroll Up", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # 👎 Scroll Down
                elif thumb_down and not index_up and not middle_up:
                    pyautogui.scroll(-300)
                    cv2.putText(frame, "Scroll Down", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                # ☝️ Select (index only)
                elif index_up and not middle_up and not ring_up and not pinky_up:
                    pyautogui.click()
                    pyautogui.sleep(0.25)
                    cv2.putText(frame, "Select / Click", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

                # ✌️ Move Cursor (index + middle)
                elif index_up and middle_up and not ring_up and not pinky_up:
                    ix, iy = index_tip
                    target_x = np.interp(ix, [0, w], [0, screen_w])
                    target_y = np.interp(iy, [0, h], [0, screen_h])

                    # Adaptive smoothening (less lag when moving fast)
                    dx, dy = abs(target_x - prev_x), abs(target_y - prev_y)
                    adaptive_smooth = max(3, min(10, int(10 - min(8, (dx + dy) / 50))))

                    curr_x = prev_x + (target_x - prev_x) / adaptive_smooth
                    curr_y = prev_y + (target_y - prev_y) / adaptive_smooth

                    # Moving average smoothing
                    pts_x.append(curr_x)
                    pts_y.append(curr_y)
                    avg_x = int(np.mean(pts_x))
                    avg_y = int(np.mean(pts_y))

                    pyautogui.moveTo(avg_x, avg_y)
                    prev_x, prev_y = curr_x, curr_y

                    cv2.putText(frame, "Move Cursor️", (10, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.putText(frame, "Virtual Mouse (Press 'q' to quit)",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Virtual Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
