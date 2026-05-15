import cv2
import mediapipe as mp
import keyboard
import json
import math
import os
from collections import deque

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

GESTURES_FILE = "gestures.json"
CONTROL_FILE  = "control_gesture.json"   # single gesture, not a role dict

# ── Helpers ───────────────────────────────────────────────────────────────────
def pt_dist(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def lm_dist(a, b):
    return math.sqrt((a["x"]-b["x"])**2 + (a["y"]-b["y"])**2 + (a["z"]-b["z"])**2)

def hand_to_relative(landmarks):
    wrist = landmarks[0]
    return [{"x": lm.x - wrist.x, "y": lm.y - wrist.y, "z": lm.z - wrist.z}
            for lm in landmarks]

def get_hand_by_side(result, side):
    """Return landmarks for the given MediaPipe side label.
    Unflipped camera: 'Left' = user's right hand (ASL), 'Right' = user's left hand (control)."""
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == side:
            return lms
    return None

def score_static(live_rel, saved_landmarks):
    return sum(lm_dist(live_rel[i], saved_landmarks[i]) for i in range(21)) / 21

def wrist_spread(history):
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return (max(xs) - min(xs)) + (max(ys) - min(ys))

def average_relative_landmarks(frame_list):
    sums = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
    for frame in frame_list:
        for i, lm in enumerate(frame):
            sums[i]["x"] += lm["x"]
            sums[i]["y"] += lm["y"]
            sums[i]["z"] += lm["z"]
    n = len(frame_list)
    return [{"x": s["x"]/n, "y": s["y"]/n, "z": s["z"]/n} for s in sums]

# ── DTW ───────────────────────────────────────────────────────────────────────
def resample_path(points, n=32):
    if len(points) < 2:
        return points
    total = sum(pt_dist(points[i], points[i+1]) for i in range(len(points)-1))
    if total == 0:
        return [points[0]] * n
    step = total / (n - 1)
    resampled = [points[0]]
    accum = 0.0
    i = 0
    while len(resampled) < n and i < len(points) - 1:
        seg = pt_dist(points[i], points[i+1])
        if accum + seg >= step:
            t = (step - accum) / seg
            nx = points[i][0] + t * (points[i+1][0] - points[i][0])
            ny = points[i][1] + t * (points[i+1][1] - points[i][1])
            resampled.append((nx, ny))
            accum = accum + seg - step
        else:
            accum += seg
            i += 1
    while len(resampled) < n:
        resampled.append(points[-1])
    return resampled

def dtw_distance(a, b):
    n, m = len(a), len(b)
    cost = [[float("inf")] * m for _ in range(n)]
    cost[0][0] = pt_dist(a[0], b[0])
    for i in range(1, n):
        cost[i][0] = cost[i-1][0] + pt_dist(a[i], b[0])
    for j in range(1, m):
        cost[0][j] = cost[0][j-1] + pt_dist(a[0], b[j])
    for i in range(1, n):
        for j in range(1, m):
            cost[i][j] = pt_dist(a[i], b[j]) + min(
                cost[i-1][j], cost[i][j-1], cost[i-1][j-1])
    return cost[n-1][m-1]

# ── Gesture file I/O ──────────────────────────────────────────────────────────
def load_gestures():
    if not os.path.exists(GESTURES_FILE):
        return []
    with open(GESTURES_FILE) as f:
        data = json.load(f)
    for g in data:
        if "letter" in g and "label" not in g:
            g["label"] = g.pop("letter")
    return data

def save_gesture(entry):
    gestures = load_gestures()
    gestures = [g for g in gestures if g["label"] != entry["label"]]
    gestures.append(entry)
    with open(GESTURES_FILE, "w") as f:
        json.dump(gestures, f, indent=2)
    print(f"Saved '{entry['label']}' as {entry['type']} ({len(gestures)} total)")

def load_control_gesture():
    """Load the single control gesture landmark list, or None if not recorded yet."""
    if not os.path.exists(CONTROL_FILE):
        return None
    with open(CONTROL_FILE) as f:
        return json.load(f)

def save_control_gesture(landmarks):
    with open(CONTROL_FILE, "w") as f:
        json.dump(landmarks, f, indent=2)
    print("Saved control gesture")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    gestures        = load_gestures()
    control_gesture = load_control_gesture()

    print(f"Loaded {len(gestures)} gesture(s): {[g['label'] for g in gestures]}")
    print(f"Control gesture: {'loaded' if control_gesture else 'not recorded'}")
    print()
    print("E = record ASL gesture (right hand, hold)")
    print("R = record control gesture (left hand, press once then get in position)")
    print("Q = quit")
    print()
    print("Hold the control gesture with your LEFT hand to enable detection.")

    # ── Tuning ────────────────────────────────────────────────────────────────
    STILL_WINDOW          = 12
    STILL_THRESHOLD       = 0.025
    STATIC_THRESHOLD      = 0.10
    CONFIRM_FRAMES        = 10
    DTW_THRESHOLD         = 15.0
    MIN_TRAJ_POINTS       = 10
    DISPLAY_FRAMES        = 90
    CONTROL_THRESHOLD     = 0.10
    CONTROL_RECORD_FRAMES = 90   # frames to record after countdown
    CONTROL_COUNTDOWN     = 90   # 3 second countdown at ~30fps

    # ── State ─────────────────────────────────────────────────────────────────
    wrist_history   = deque(maxlen=STILL_WINDOW)
    confirm_buffer  = deque(maxlen=CONFIRM_FRAMES)
    control_confirm = deque(maxlen=CONFIRM_FRAMES)
    pinky_buffer    = []
    was_moving      = False

    # Detection is driven entirely by the control hand — starts off
    detecting      = False

    display_label  = None
    display_timer  = 0
    match_cooldown = 0   # frames to suppress static after a trajectory match

    # Control gesture flash
    control_flash_timer = 0

    # ASL recording
    recorded_frames = []
    pinky_record    = []
    was_pressing_e  = False

    # Control gesture recording
    control_recording        = False
    control_record_countdown = 0
    control_recorded_frames  = []
    was_pressing_r           = False

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ts     = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = landmarker.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

            is_pressing_e = keyboard.is_pressed("e")
            is_pressing_r = keyboard.is_pressed("r")

            # Resolve hands by side (stable regardless of detection order)
            # Unflipped camera: MediaPipe "Left" = user's right, "Right" = user's left
            asl_hand     = get_hand_by_side(result, "Left")   # user's right — ASL
            control_hand = get_hand_by_side(result, "Right")  # user's left  — control
            hand_visible = asl_hand is not None

            # ── Draw all visible hand landmarks ───────────────────────────────
            for lms, handedness in zip(result.hand_landmarks, result.handedness):
                side  = handedness[0].category_name
                color = (0, 255, 0) if side == "Left" else (255, 0, 0)  # green=ASL, blue=control
                for lm in lms:
                    cv2.circle(frame,
                               (int(lm.x * frame.shape[1]),
                                int(lm.y * frame.shape[0])),
                               4, color, -1)

            # ── Control hand: drive detecting on/off each frame ───────────────
            if control_gesture and control_hand is not None and not control_recording:
                off_rel = hand_to_relative(control_hand)
                score   = score_static(off_rel, control_gesture)

                if score < CONTROL_THRESHOLD:
                    control_confirm.append(True)
                else:
                    control_confirm.clear()

                # Confirmed: start detecting
                if len(control_confirm) == CONFIRM_FRAMES and all(control_confirm):
                    if not detecting:
                        detecting = True
                        wrist_history.clear()
                        confirm_buffer.clear()
                        pinky_buffer.clear()
                        was_moving     = False
                        match_cooldown = 0
                        control_flash_timer = 20
                        print("Detection STARTED")
            else:
                # Control hand gone or gesture not matched — stop detecting
                control_confirm.clear()
                if detecting:
                    detecting = False
                    control_flash_timer = 20
                    print("Detection STOPPED")

            # ── Record ASL gesture (E held, user's right hand) ───────────────
            if is_pressing_e and hand_visible and not control_recording:
                rel = hand_to_relative(asl_hand)
                recorded_frames.append(rel)
                pinky = asl_hand[20]
                pinky_record.append((pinky.x, pinky.y))
                cv2.putText(frame, f"RECORDING ASL ({len(recorded_frames)} frames)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if was_pressing_e and not is_pressing_e and recorded_frames:
                label = input("Label (e.g. A): ").strip().upper()
                gtype = input("Static or trajectory? (s/t): ").strip().lower()
                if label and gtype in ("s", "t"):
                    if gtype == "s":
                        entry = {
                            "label": label,
                            "type": "static",
                            "landmarks": average_relative_landmarks(recorded_frames),
                        }
                    else:
                        sx, sy = pinky_record[0]
                        norm   = [(x - sx, y - sy) for x, y in pinky_record]
                        entry  = {
                            "label": label,
                            "type": "trajectory",
                            "path": resample_path(norm),
                        }
                    save_gesture(entry)
                    gestures = load_gestures()
                recorded_frames.clear()
                pinky_record.clear()
                wrist_history.clear()
                confirm_buffer.clear()
                pinky_buffer.clear()
                was_moving = False

            # ── Trigger control recording (R press, single tap) ───────────────
            if was_pressing_r and not is_pressing_r and not control_recording:
                control_recording        = True
                control_record_countdown = CONTROL_COUNTDOWN
                control_recorded_frames.clear()
                print("Get left hand in frame... recording starts in 3 seconds")

            # ── Control recording: countdown then capture ─────────────────────
            if control_recording:
                if control_record_countdown > 0:
                    control_record_countdown -= 1
                    seconds_left = math.ceil(control_record_countdown / 30)
                    cv2.putText(frame,
                                f"Control recording starts in {seconds_left}s...",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 255), 2)
                else:
                    if control_hand is not None:
                        rel = hand_to_relative(control_hand)
                        control_recorded_frames.append(rel)

                    frames_so_far = len(control_recorded_frames)
                    cv2.putText(frame,
                                f"RECORDING CONTROL ({frames_so_far}/{CONTROL_RECORD_FRAMES})",
                                (10, 65), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (255, 0, 255), 2)

                    if frames_so_far >= CONTROL_RECORD_FRAMES:
                        if control_recorded_frames:
                            averaged = average_relative_landmarks(control_recorded_frames)
                            save_control_gesture(averaged)
                            control_gesture = load_control_gesture()
                        control_recorded_frames.clear()
                        control_recording = False

            # ── Detection (user's right hand, only when detecting=True) ──────
            if (not is_pressing_e and not control_recording
                    and hand_visible and gestures and detecting):

                wrist = asl_hand[0]
                pinky = asl_hand[20]
                wrist_history.append((wrist.x, wrist.y))

                if len(wrist_history) == STILL_WINDOW:
                    spread   = wrist_spread(wrist_history)
                    is_still = spread < STILL_THRESHOLD

                    if not is_still:
                        pinky_buffer.append((pinky.x, pinky.y))
                        if len(pinky_buffer) > 120:
                            pinky_buffer.pop(0)
                        confirm_buffer.clear()
                        was_moving = True

                    else:
                        # Transition moving → still: try trajectory match
                        if was_moving and len(pinky_buffer) >= MIN_TRAJ_POINTS:
                            traj_gestures = [g for g in gestures
                                             if g.get("type") == "trajectory"]
                            if traj_gestures:
                                sx, sy    = pinky_buffer[0]
                                norm      = [(x - sx, y - sy) for x, y in pinky_buffer]
                                live_path = resample_path(norm)

                                best_label = None
                                best_score = float("inf")
                                for g in traj_gestures:
                                    saved_path = resample_path(g["path"])
                                    score = dtw_distance(live_path, saved_path)
                                    if score < best_score:
                                        best_score = score
                                        best_label = g["label"]

                                if best_score < DTW_THRESHOLD:
                                    display_label  = best_label
                                    display_timer  = DISPLAY_FRAMES
                                    match_cooldown = CONFIRM_FRAMES + 5
                                    print(f"Trajectory MATCHED: {best_label} "
                                          f"(score={best_score:.2f})")

                            pinky_buffer.clear()

                        was_moving = False

                        # Static match with confirmation buffer
                        if match_cooldown > 0:
                            match_cooldown -= 1
                            confirm_buffer.clear()
                        else:
                            static_gestures = [g for g in gestures
                                               if g.get("type") == "static"]
                            if static_gestures:
                                live_rel   = hand_to_relative(asl_hand)
                                best_label = None
                                best_score = float("inf")
                                for g in static_gestures:
                                    score = score_static(live_rel, g["landmarks"])
                                    if score < best_score:
                                        best_score = score
                                        best_label = g["label"]

                                if best_score < STATIC_THRESHOLD:
                                    confirm_buffer.append(best_label)
                                else:
                                    confirm_buffer.clear()

                                if (len(confirm_buffer) == CONFIRM_FRAMES and
                                        len(set(confirm_buffer)) == 1):
                                    detected = confirm_buffer[0]
                                    if detected != display_label or display_timer == 0:
                                        display_label = detected
                                        display_timer = DISPLAY_FRAMES
                                        print(f"Static MATCHED: {detected} "
                                              f"(score={best_score:.4f})")
                                    confirm_buffer.clear()

            # ── Display detected label ────────────────────────────────────────
            if display_timer > 0:
                display_timer -= 1
                cv2.putText(frame, display_label,
                            (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 220, 0), 6)

            # ── Control gesture flash border ──────────────────────────────────
            if control_flash_timer > 0:
                control_flash_timer -= 1
                border_color = (0, 220, 0) if detecting else (0, 0, 255)
                cv2.rectangle(frame, (0, 0),
                              (frame.shape[1] - 1, frame.shape[0] - 1),
                              border_color, 8)

            # ── HUD ───────────────────────────────────────────────────────────
            state_color = (0, 220, 0) if detecting else (0, 0, 255)
            state_text  = "DETECTING" if detecting else "PAUSED"
            cv2.putText(frame, state_text,
                        (frame.shape[1] - 170, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, state_color, 2)

            # ── Control gesture confirmation progress bar ─────────────────────
            if len(control_confirm) > 0 and not detecting:
                fill_ratio   = len(control_confirm) / CONFIRM_FRAMES
                bar_width    = 200
                filled       = int(bar_width * fill_ratio)
                bar_x, bar_y = frame.shape[1] - 220, 55
                cv2.rectangle(frame,
                              (bar_x, bar_y),
                              (bar_x + bar_width, bar_y + 18),
                              (80, 80, 80), -1)
                cv2.rectangle(frame,
                              (bar_x, bar_y),
                              (bar_x + filled, bar_y + 18),
                              (255, 180, 0), -1)
                cv2.putText(frame, "CTRL: activating...",
                            (bar_x, bar_y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 180, 0), 1)

            cv2.putText(frame, "E: ASL gesture  R: control gesture  Q: quit",
                        (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("ASL Detector", frame)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

            was_pressing_e = is_pressing_e
            was_pressing_r = is_pressing_r

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()