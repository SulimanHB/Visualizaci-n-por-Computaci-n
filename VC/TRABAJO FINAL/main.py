import cv2
import numpy as np
import mediapipe as mp
import math
import time
import improve_IA  

# ==========================================
# 1. Device and ORB Configuration
# ==========================================
# Change index to 0 if using the default webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

orb = cv2.ORB_create(nfeatures=2000)
index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# ==========================================
# 2. Persistence and Canvas Variables
# ==========================================
ar_library = []
selected_points = []
L_SIZE = 1000
current_canvas = np.zeros((L_SIZE, L_SIZE, 3), dtype=np.uint8)
src_pts = np.array([[0, 0], [L_SIZE, 0], [L_SIZE, L_SIZE], [0, L_SIZE]], dtype=np.float32)

DRAW_COLORS = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
color_names = ["RED", "GREEN", "BLUE", "YELLOW"]
color_index = 0

thickness_list = [5, 15, 45]
thickness_names = ["THIN", "MEDIUM", "THICK"]
thick_index = 1

prev_point_ar = None
prev_index_x, prev_index_y = None, None
last_action_time = 0
erase_start_time = None
status_msg = "Waiting for 4 points selection..."

SWIPE_THRESHOLD = 60
COOLDOWN = 0.6

def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def select_points(event, x, y, flags, param):
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append([x, y])

cv2.namedWindow("Virtual Graffiti AR", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Virtual Graffiti AR", select_points)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_f, w_f, _ = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_time = time.time()

    # --- A. RECOGNITION ---
    kp_frame, des_frame = orb.detectAndCompute(gray, None)

    if des_frame is not None and len(kp_frame) >= 2 and len(ar_library) > 0:
        for item in ar_library:
            try:
                matches = flann.knnMatch(item['des'], des_frame, k=2)
                good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

                if len(good_matches) > 30:
                    src_p = np.float32([item['kp'][m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_p = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    M_track, _ = cv2.findHomography(src_p, dst_p, cv2.RANSAC, 5.0)
                    
                    if M_track is not None:
                        M_final = M_track @ item['M_orig']
                        warped = cv2.warpPerspective(item['canvas'], M_final, (w_f, h_f))
                        mask_ar = cv2.threshold(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
                        frame[mask_ar > 0] = warped[mask_ar > 0]
            except: continue

    # --- B. HAND AND GESTURE PROCESSING ---
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            ix, iy = int(lm[8].x * w_f), int(lm[8].y * h_f)
            tx, ty = int(lm[4].x * w_f), int(lm[4].y * h_f)
            dist_pinch = math.hypot(ix - tx, iy - ty)
            i_up, m_up, r_up, p_up = finger_up(lm, 8, 6), finger_up(lm, 12, 10), finger_up(lm, 16, 14), finger_up(lm, 20, 18)

            # 1. FULL ERASE
            if i_up and m_up and r_up and p_up:
                if erase_start_time is None: erase_start_time = current_time
                remaining_time = 2.1 - (current_time - erase_start_time)
                status_msg = f"ERASING IN... {remaining_time:.1f}s"
                if current_time - erase_start_time > 2.0:
                    current_canvas = np.zeros_like(current_canvas)
                    erase_start_time = None
            else: erase_start_time = None

            # 2. CHANGE THICKNESS (Index + Middle + Vertical Swipe)
            if i_up and m_up and not r_up:
                if prev_index_y is not None:
                    dy = iy - prev_index_y
                    if abs(dy) > SWIPE_THRESHOLD and current_time - last_action_time > COOLDOWN:
                        thick_index = (thick_index + (1 if dy < 0 else -1)) % len(thickness_list)
                        last_action_time = current_time
                prev_index_y = iy
            else: prev_index_y = None

            # 3. CHANGE COLOR (Index Only + Horizontal Swipe)
            if i_up and not m_up:
                if prev_index_x is not None:
                    dx = ix - prev_index_x
                    if abs(dx) > SWIPE_THRESHOLD and current_time - last_action_time > COOLDOWN:
                        color_index = (color_index + (1 if dx > 0 else -1)) % len(DRAW_COLORS)
                        last_action_time = current_time
                prev_index_x = ix
            else: prev_index_x = None

            # 4. LOCAL DRAW AND ERASE
            if len(selected_points) == 4:
                M_draw, _ = cv2.findHomography(np.array(selected_points, dtype=np.float32), src_pts)
                ar_finger = cv2.perspectiveTransform(np.array([[[ix, iy]]], dtype=np.float32), M_draw)
                gx, gy = int(ar_finger[0][0][0]), int(ar_finger[0][0][1])

                if i_up and m_up and not r_up and dist_pinch > 60: # Eraser
                    if 0 <= gx < L_SIZE and 0 <= gy < L_SIZE:
                        cv2.circle(current_canvas, (gx, gy), 30, (0, 0, 0), -1)
                        cv2.circle(frame, (ix, iy), 35, (255, 255, 255), 2)
                elif dist_pinch < 40: # Draw
                    if 0 <= gx < L_SIZE and 0 <= gy < L_SIZE:
                        if prev_point_ar is not None:
                            cv2.line(current_canvas, prev_point_ar, (gx, gy), DRAW_COLORS[color_index], thickness_list[thick_index])
                        prev_point_ar = (gx, gy)
                else: prev_point_ar = None

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- C. HUD INTERFACE ---
    cv2.rectangle(frame, (0, 0), (w_f, 80), (30, 30, 30), -1)
    cv2.circle(frame, (40, 40), 20, DRAW_COLORS[color_index], -1)
    cv2.putText(frame, f"COLOR: {color_names[color_index]}", (70, 50), 1, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, f"SIZE: {thickness_names[thick_index]}", (w_f // 2 - 100, 50), 1, 1.5, (255, 255, 255), 2)
    cv2.putText(frame, status_msg, (10, h_f - 20), 1, 1.2, (0, 255, 0), 2)

    if len(selected_points) == 4:
        M_warp_act, _ = cv2.findHomography(src_pts, np.array(selected_points, dtype=np.float32))
        warped_act = cv2.warpPerspective(current_canvas, M_warp_act, (w_f, h_f))
        mask_act = cv2.threshold(cv2.cvtColor(warped_act, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)[1]
        frame[mask_act > 0] = warped_act[mask_act > 0]
        cv2.polylines(frame, [np.array(selected_points, np.int32)], True, (0, 255, 0), 2)
    else:
        for p in selected_points: cv2.circle(frame, tuple(p), 6, (0, 0, 255), -1)

    cv2.imshow("Virtual Graffiti AR", frame)
    key = cv2.waitKey(1)
    if key == 27: break # ESC to exit
    
    # Save/Seal Graffiti
    if key == ord('s') and len(selected_points) == 4 and des_frame is not None:
        if len(kp_frame) >= 10:
            M_orig, _ = cv2.findHomography(src_pts, np.array(selected_points, dtype=np.float32))
            ar_library.append({'kp': kp_frame, 'des': des_frame, 'canvas': current_canvas.copy(), 'M_orig': M_orig})
            current_canvas = np.zeros_like(current_canvas)
            selected_points = []
            status_msg = "GRAFFITI SEALED"
            
    # Reset Points
    if key == ord('r'): selected_points = []
    
    # AI Generation
    if key == ord('i'):
        if np.any(current_canvas):
            status_msg = "AI PROCESSING... PLEASE WAIT"
            cv2.imshow("Virtual Graffiti AR", frame)
            cv2.waitKey(1) # Force UI update

            try:
                result = improve_IA.improve_draw_ia(current_canvas)
                if result is not None:
                    current_canvas = cv2.resize(result, (L_SIZE, L_SIZE))
                    status_msg = "AI: DRAWING OPTIMIZED"
                else:
                    status_msg = "AI CONNECTION ERROR"
            except Exception as e:
                print(f"AI Error: {e}")
                status_msg = "AI ERROR"
        else:
            status_msg = "DRAW SOMETHING FIRST"

cap.release()
cv2.destroyAllWindows()