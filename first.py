import time
import urllib.request
import math
from pathlib import Path

import cv2
import mediapipe as mp

vision = mp.tasks.vision

MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
if not MODEL_PATH.exists():
    print("Downloading hand model...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=vision.RunningMode.VIDEO,   # video mode = tracks between frames (faster, smoother)
    num_hands=2,                             # set to 2 for two hands
    min_hand_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def pinch_calculate(hand):
    hand_size = dist(hand[0],hand[9])

    return dist(hand[4],hand[8]) / hand_size

def Isfist(hand):
    tips = [hand[8], hand[12], hand[16] , hand[20]]
    knucks = [hand[6], hand[10], hand[14], hand[18]]
    fullFist = True

    for i in range (len(tips)):
        if tips[i].y < knucks[i].y:
            fullFist = False
        else:
            continue

    return fullFist

def IsPeace(hand):
    tips = [hand[8], hand[12], hand[16] , hand[20]]
    knucks = [hand[6], hand[10], hand[14], hand[18]]
    IsPeace = False

    if tips[0].y < knucks[0].y and tips[1].y < knucks[1].y:
        if knucks[2].y < tips[2].y and knucks[3].y < tips[3].y:
            IsPeace = True
        else:
            IsPeace = False

    return IsPeace

def IsThUp(hand):
    tips = [hand[8], hand[12], hand[16] , hand[20]]
    knucks = [hand[7], hand[11], hand[15], hand[19]]
    IsThUp = False

    if hand[4].y < hand[3].y < hand[2].y and hand[4].y < hand[5].y:
        for i in range (len(tips)):
                if tips[i].x > knucks[i].x:
                    IsThUp = True
                else:
                    IsThUp = False
                    break
        

    return IsThUp
    
def draw_hand(frame, hand):
    """Draw the 21 dots and the connecting lines."""
    vision.drawing_utils.draw_landmarks(
        frame,
        hand,
        vision.HandLandmarksConnections.HAND_CONNECTIONS,
        vision.drawing_styles.get_default_hand_landmarks_style(),
        vision.drawing_styles.get_default_hand_connections_style(),
    )



def main():
    cap = cv2.VideoCapture(0)   # Windows: cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)    # smaller frames = higher FPS
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    last_ts, prev_time = 0, time.time()
    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Video mode requires strictly increasing timestamps in milliseconds.
            ts = max(last_ts + 1, int(time.monotonic() * 1000))
            last_ts = ts
            result = landmarker.detect_for_video(mp_image, ts)

            for hand in result.hand_landmarks:   # empty list if no hand is visible
                draw_hand(frame, hand)
                ratio = pinch_calculate(hand)

                if IsThUp(hand):
                    label = "Thumbs Up!!"
                elif IsPeace(hand):
                    label = "Peace Sign!!"
                elif Isfist(hand):
                    label = "Making a Fist!!"
                elif ratio < 0.20:
                    label = "Pinching!!"
                else:
                    label = None

                if label:
                    cv2.putText(frame, label, (600, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)               


            now = time.time()
            fps = 1 / max(now - prev_time, 1e-6)
            prev_time = now
            cv2.putText(frame, f"{fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Hand Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

