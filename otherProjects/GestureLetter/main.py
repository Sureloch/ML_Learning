import cv2
import mediapipe as mp
import keyboard
import json
import math
import os
import numpy as np
from collections import deque

BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

GESTURES_FILE = "gestures.json"
CONTROL_FILE  = "control_gesture.json"

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
    for lms, handedness in zip(result.hand_landmarks, result.handedness):
        if handedness[0].category_name == side:
            return lms
    return None

def procrustes_score(live_rel, saved_landmarks):
    """Align live points to saved template via translation, rotation, scale.
       Returns RMSD after alignment."""
    live_pts = np.array([[lm["x"], lm["y"]] for lm in live_rel])
    saved_pts = np.array([[lm["x"], lm["y"]] for lm in saved_landmarks])
    
    live_cent = live_pts - np.mean(live_pts, axis=0)
    saved_cent = saved_pts - np.mean(saved_pts, axis=0)
    
    scale_live = np.sqrt(np.sum(live_cent**2))
    scale_saved = np.sqrt(np.sum(saved_cent**2))
    if scale_live == 0 or scale_saved == 0:
        return float('inf')
    live_norm = live_cent / scale_live
    saved_norm = saved_cent / scale_saved
    
    H = live_norm.T @ saved_norm
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    aligned = live_norm @ R
    aligned = aligned * scale_saved
    rmsd = np.sqrt(np.mean(np.sum((aligned - saved_cent)**2, axis=1)))
    return rmsd

def point_velocity(history):
    """Average Euclidean distance per step over a deque of (x,y) points."""
    if len(history) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(history)):
        total += math.hypot(history[i][0] - history[i-1][0],
                            history[i][1] - history[i-1][1])
    return total / (len(history) - 1)

def average_relative_landmarks(frame_list):
    sums = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(21)]
    for frame in frame_list:
        for i, lm in enumerate(frame):
            sums[i]["x"] += lm["x"]
            sums[i]["y"] += lm["y"]
            sums[i]["z"] += lm["z"]
    n = len(frame_list)
    return [{"x": s["x"]/n, "y": s["y"]/n, "z": s["z"]/n} for s in sums]

# ── Path helpers ──────────────────────────────────────────────────────────────
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

def normalize_path(points):
    """Resample to 32 pts, then translate to origin and scale separately in X and Y,
       preserving aspect ratio (each coordinate mapped to 0..1 independently)."""
    pts = resample_path(points, n=32)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = max_x - min_x
    range_y = max_y - min_y
    # Avoid division by zero
    range_x = range_x if range_x > 1e-6 else 1.0
    range_y = range_y if range_y > 1e-6 else 1.0
    return [((x - min_x) / range_x, (y - min_y) / range_y) for x, y in pts]

# ── DTW with angle features ───────────────────────────────────────────────────
def path_to_angles(pts):
    angles = []
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        angles.append(math.atan2(dy, dx))
    return angles

def dtw_distance(a, b):
    n, m = len(a), len(b)
    ang_a = path_to_angles(a)
    ang_b = path_to_angles(b)

    def point_cost(i, j):
        pos_d = pt_dist(a[i], b[j])
        if i > 0 and j > 0:
            da = ang_a[i-1] - ang_b[j-1]
            da = (da + math.pi) % (2 * math.pi) - math.pi
            ang_d = abs(da) / math.pi   # 0..1
        else:
            ang_d = 0.0
        return pos_d + 0.8 * ang_d   # increased weight from 0.4 to 0.8

    cost = [[float("inf")] * m for _ in range(n)]
    cost[0][0] = point_cost(0, 0)
    for i in range(1, n):
        cost[i][0] = cost[i-1][0] + point_cost(i, 0)
    for j in range(1, m):
        cost[0][j] = cost[0][j-1] + point_cost(0, j)
    for i in range(1, n):
        for j in range(1, m):
            cost[i][j] = point_cost(i, j) + min(
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
    if entry["type"] == "trajectory":
        existing = next((g for g in gestures if g["label"] == entry["label"]), None)
        if existing and existing.get("type") == "trajectory":
            existing.setdefault("takes", [existing["path"]])
            existing["takes"].append(entry["path"])
            existing["path"] = entry["path"]
            take_count = len(existing["takes"])
        else:
            gestures = [g for g in gestures if g["label"] != entry["label"]]
            entry["takes"] = [entry["path"]]
            take_count = 1
            gestures.append(entry)
    else:
        gestures = [g for g in gestures if g["label"] != entry["label"]]
        gestures.append(entry)
        take_count = 1
    with open(GESTURES_FILE, "w") as f:
        json.dump(gestures, f, indent=2)
    print(f"Saved '{entry['label']}' as {entry['type']} "
          f"({take_count} take(s), {len(gestures)} gesture(s) total)")

def load_control_gesture():
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
    STILL_VELOCITY_THRESHOLD      = 0.012   # slightly increased for sensitivity
    STATIC_THRESHOLD              = 0.12
    CONFIRM_FRAMES                = 15
    DTW_THRESHOLD                 = 30.0
    MIN_TRAJ_POINTS               = 15
    DISPLAY_FRAMES                = 90
    CONTROL_THRESHOLD             = 0.10
    CONTROL_RECORD_FRAMES         = 90
    CONTROL_COUNTDOWN             = 90
    MOTION_DISPLACEMENT_THRESHOLD = 0.08
    REJECTION_RATIO               = 0.75

    # ── State ─────────────────────────────────────────────────────────────────
    # Use index fingertip for stillness detection (more sensitive)
    index_history     = deque(maxlen=12)
    confirm_buffer    = deque(maxlen=CONFIRM_FRAMES)
    control_confirm   = deque(maxlen=CONFIRM_FRAMES)
    traj_buffer       = []                 # stores index fingertip positions during motion
    was_moving        = False
    fingertip_motion_start = None

    detecting      = False

    display_label  = None
    display_timer  = 0
    match_cooldown = 0

    control_flash_timer = 0

    recorded_frames = []
    traj_record     = []                   # for recording (index fingertip)
    was_pressing_e  = False

    control_recording        = False
    control_record_countdown = 0
    control_recorded_frames  = []
    was_pressing_r           = False

    with HandLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(0)
        # Increase resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

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

            asl_hand     = get_hand_by_side(result, "Left")   # user's right — ASL
            control_hand = get_hand_by_side(result, "Right")  # user's left  — control
            hand_visible = asl_hand is not None

            # ── Draw all visible hand landmarks with side labels ─────────────
            for lms, handedness in zip(result.hand_landmarks, result.handedness):
                side  = handedness[0].category_name
                color = (0, 255, 0) if side == "Left" else (255, 0, 0)
                for lm in lms:
                    cv2.circle(frame,
                               (int(lm.x * frame.shape[1]),
                                int(lm.y * frame.shape[0])),
                               4, color, -1)
                # Draw side label above wrist
                wrist = lms[0]
                cx, cy = int(wrist.x * frame.shape[1]), int(wrist.y * frame.shape[0])
                cv2.putText(frame, side, (cx-20, cy-20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ── Warning if ASL hand lost during detection ────────────────────
            if not hand_visible and detecting and not is_pressing_e:
                cv2.putText(frame, "!!! RIGHT HAND NOT DETECTED !!!", (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # ── Control hand: drive detecting on/off each frame ──────────────
            if control_gesture and control_hand is not None and not control_recording:
                off_rel = hand_to_relative(control_hand)
                score   = procrustes_score(off_rel, control_gesture)

                if score < CONTROL_THRESHOLD:
                    control_confirm.append(True)
                else:
                    control_confirm.clear()

                if len(control_confirm) == CONFIRM_FRAMES and all(control_confirm):
                    if not detecting:
                        detecting          = True
                        fingertip_motion_start = None
                        index_history.clear()
                        confirm_buffer.clear()
                        traj_buffer.clear()
                        was_moving     = False
                        match_cooldown = 0
                        control_flash_timer = 20
                        print("Detection STARTED")
            else:
                control_confirm.clear()
                if detecting:
                    detecting = False
                    control_flash_timer = 20
                    print("Detection STOPPED")

            # ── Record ASL gesture (E held, user's right hand) ───────────────
            if is_pressing_e and hand_visible and not control_recording:
                rel = hand_to_relative(asl_hand)
                recorded_frames.append(rel)
                # Use index fingertip (landmark 8) for trajectory recording
                index_tip = asl_hand[8]
                traj_record.append((index_tip.x, index_tip.y))
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
                        sx, sy = traj_record[0]
                        norm  = [(x - sx, y - sy) for x, y in traj_record]
                        entry = {
                            "label": label,
                            "type": "trajectory",
                            "path": norm,
                        }
                    save_gesture(entry)
                    gestures = load_gestures()
                recorded_frames.clear()
                traj_record.clear()
                index_history.clear()
                confirm_buffer.clear()
                traj_buffer.clear()
                fingertip_motion_start = None
                was_moving = False

            # ── Trigger control recording (R press, single tap) ──────────────
            if was_pressing_r and not is_pressing_r and not control_recording:
                control_recording        = True
                control_record_countdown = CONTROL_COUNTDOWN
                control_recorded_frames.clear()
                print("Get left hand in frame... recording starts in 3 seconds")

            # ── Control recording: countdown then capture ────────────────────
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

                index_tip = asl_hand[8]
                index_history.append((index_tip.x, index_tip.y))

                if len(index_history) == index_history.maxlen:
                    vel = point_velocity(index_history)
                    is_still = vel < STILL_VELOCITY_THRESHOLD

                    if not is_still:
                        if fingertip_motion_start is None:
                            fingertip_motion_start = (index_tip.x, index_tip.y)
                        # Append index fingertip for trajectory matching
                        traj_buffer.append((index_tip.x, index_tip.y))
                        if len(traj_buffer) > 120:
                            traj_buffer.pop(0)
                        confirm_buffer.clear()
                        was_moving = True
                    else:
                        # Transition moving → still: try trajectory match
                        if was_moving and len(traj_buffer) >= MIN_TRAJ_POINTS:
                            traj_gestures = [g for g in gestures
                                             if g.get("type") == "trajectory"]
                            traj_matched = False
                            if traj_gestures:
                                sx, sy    = traj_buffer[0]
                                norm      = [(x - sx, y - sy) for x, y in traj_buffer]
                                live_path = normalize_path(norm)

                                scores = []
                                for g in traj_gestures:
                                    takes = g.get("takes", [g["path"]])
                                    best_take_score = float("inf")
                                    for take in takes:
                                        saved_path = normalize_path(take)
                                        score = dtw_distance(live_path, saved_path)
                                        if score < best_take_score:
                                            best_take_score = score
                                    scores.append((best_take_score, g["label"]))
                                scores.sort()
                                if scores:
                                    best_score, best_label = scores[0]
                                    if len(scores) > 1:
                                        second_best = scores[1][0]
                                        if (best_score < DTW_THRESHOLD and
                                            best_score < REJECTION_RATIO * second_best):
                                            traj_matched = True
                                            display_label = best_label
                                            display_timer = DISPLAY_FRAMES
                                            match_cooldown = CONFIRM_FRAMES + 5
                                            print(f"Trajectory MATCHED: {best_label} "
                                                  f"(score={best_score:.2f})")
                                        else:
                                            print(f"Trajectory rejected: best={best_label} "
                                                  f"score={best_score:.2f}, second={second_best:.2f}")
                                    else:
                                        if best_score < DTW_THRESHOLD:
                                            traj_matched = True
                                            display_label = best_label
                                            display_timer = DISPLAY_FRAMES
                                            match_cooldown = CONFIRM_FRAMES + 5
                                            print(f"Trajectory MATCHED: {best_label} "
                                                  f"(score={best_score:.2f})")
                                        else:
                                            print(f"Trajectory NO MATCH: {best_label} "
                                                  f"score={best_score:.2f}")

                            # Suppress static if significant motion occurred
                            if (not traj_matched and fingertip_motion_start is not None):
                                dx = index_tip.x - fingertip_motion_start[0]
                                dy = index_tip.y - fingertip_motion_start[1]
                                displacement = math.sqrt(dx*dx + dy*dy)
                                if displacement > MOTION_DISPLACEMENT_THRESHOLD:
                                    match_cooldown = max(match_cooldown, CONFIRM_FRAMES + 5)
                                    print(f"Motion displacement {displacement:.3f} > threshold "
                                          f"— static suppressed")

                            fingertip_motion_start = None
                            traj_buffer.clear()

                        was_moving = False

                        # Static match with confirmation buffer
                        if match_cooldown > 0:
                            match_cooldown -= 1
                            confirm_buffer.clear()
                        else:
                            static_gestures = [g for g in gestures
                                               if g.get("type") == "static"]
                            if static_gestures:
                                live_rel = hand_to_relative(asl_hand)
                                best_label = None
                                best_score = float("inf")
                                second_best_score = float("inf")
                                for g in static_gestures:
                                    score = procrustes_score(live_rel, g["landmarks"])
                                    if score < best_score:
                                        second_best_score = best_score
                                        best_score = score
                                        best_label = g["label"]
                                    elif score < second_best_score:
                                        second_best_score = score

                                if (best_score < STATIC_THRESHOLD and
                                    (second_best_score == float("inf") or
                                     best_score < REJECTION_RATIO * second_best_score)):
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

            # ── Display trajectory buffer size while moving ──────────────────
            if was_moving and detecting:
                cv2.putText(frame, f"Traj points: {len(traj_buffer)}", (10, 280),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

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