"""
Air Canvas — Draw in the Air with Hand Gestures
================================================
A real-time computer vision application that lets you draw on a virtual canvas
using hand gestures captured through your webcam.

Tech Stack: Python + OpenCV + MediaPipe Tasks API
Author: Air Canvas Project

Gestures:
    - Index finger up only          → Draw mode (trace lines)
    - Index + Middle fingers up     → Select / Navigate (move without drawing)
    - All five fingers up           → Erase mode (circular eraser around palm)
    - Fist (no fingers up)          → Idle (do nothing)
    - Thumb + Pinky up              → Clear entire canvas

Controls:
    - Hover index finger over top color bar to pick a color
    - Press 'q' to quit
    - Press 'c' to clear canvas
    - Press '+' / '=' to increase brush size
    - Press '-' to decrease brush size
    - Press 's' to save current canvas as PNG
    - Press 'z' to undo last stroke
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import time
import os
import threading
from collections import deque


# Hand Detector (MediaPipe Tasks API)
class HandDetector:
    """Detects hand landmarks using MediaPipe Tasks API and classifies gestures."""

    # MediaPipe landmark indices (same as before)
    WRIST = 0
    THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
    INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
    MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
    RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
    PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

    def __init__(self, model_path="hand_landmarker.task", num_hands=1,
                 detection_conf=0.5, tracking_conf=0.5):
        self.landmarks = []
        self.hand_detected = False
        self.frame_width = 0
        self.frame_height = 0

        # Thread-safe storage for results from async callback
        self._lock = threading.Lock()
        self._latest_result = None

        # Configure the HandLandmarker with LIVE_STREAM mode
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.LIVE_STREAM,
            num_hands=num_hands,
            min_hand_detection_confidence=detection_conf,
            min_hand_presence_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
            result_callback=self._result_callback,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def _result_callback(self, result, output_image, timestamp_ms):
        """Async callback invoked by MediaPipe when detection is done."""
        with self._lock:
            self._latest_result = result

    def detect(self, frame):
        """
        Send a BGR frame for async detection and return the latest results.
        Returns True if a hand was detected in the latest available result.
        """
        h, w, _ = frame.shape
        self.frame_width = w
        self.frame_height = h

        # Convert BGR → RGB and wrap in mp.Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Send frame with monotonically increasing timestamp
        self._timestamp_ms += 33  # ~30fps
        self.landmarker.detect_async(mp_image, self._timestamp_ms)

        # Read latest result (from previous frame's callback)
        with self._lock:
            result = self._latest_result

        self.landmarks = []
        self.hand_detected = False

        if result and result.hand_landmarks:
            hand = result.hand_landmarks[0]  # first detected hand
            self.landmarks = [
                (int(lm.x * w), int(lm.y * h)) for lm in hand
            ]
            self.hand_detected = True

            # Draw hand skeleton on frame for visual feedback
            self._draw_landmarks(frame, hand)

        return self.hand_detected

    def _draw_landmarks(self, frame, hand_landmarks):
        """Draw hand landmarks and connections on the frame."""
        h, w = frame.shape[:2]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]

        # Define hand connections (same as MediaPipe's HAND_CONNECTIONS)
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),      # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16), # Ring
            (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
            (5, 9), (9, 13), (13, 17),             # Palm
        ]

        # Draw connections
        for start_idx, end_idx in connections:
            cv2.line(frame, points[start_idx], points[end_idx], (0, 230, 180), 2, cv2.LINE_AA)

        # Draw landmarks
        for i, pt in enumerate(points):
            color = (0, 0, 255) if i in (4, 8, 12, 16, 20) else (0, 200, 0)
            cv2.circle(frame, pt, 5, color, -1)
            cv2.circle(frame, pt, 5, (255, 255, 255), 1)

    def get_landmark(self, idx):
        """Return (x, y) for a specific landmark index."""
        if self.landmarks and 0 <= idx < len(self.landmarks):
            return self.landmarks[idx]
        return None

    def fingers_up(self):
        """
        Returns a list of 5 booleans indicating which fingers are raised.
        Order: [Thumb, Index, Middle, Ring, Pinky]
        """
        if not self.landmarks:
            return [False] * 5

        fingers = []

        # Thumb — use x-axis comparison
        thumb_tip = self.landmarks[self.THUMB_TIP]
        thumb_ip = self.landmarks[self.THUMB_IP]

        # Check if left or right hand by comparing wrist to middle MCP
        wrist_x = self.landmarks[self.WRIST][0]
        middle_mcp_x = self.landmarks[self.MIDDLE_MCP][0]

        if wrist_x < middle_mcp_x:
            # Right hand (after flip) → thumb is up if tip is to the LEFT of IP
            fingers.append(thumb_tip[0] < thumb_ip[0])
        else:
            # Left hand (after flip) → thumb is up if tip is to the RIGHT of IP
            fingers.append(thumb_tip[0] > thumb_ip[0])

        # Other four fingers — tip above PIP means finger is up (lower y = higher)
        for tip_id, pip_id in [
            (self.INDEX_TIP, self.INDEX_PIP),
            (self.MIDDLE_TIP, self.MIDDLE_PIP),
            (self.RING_TIP, self.RING_PIP),
            (self.PINKY_TIP, self.PINKY_PIP),
        ]:
            tip_y = self.landmarks[tip_id][1]
            pip_y = self.landmarks[pip_id][1]
            fingers.append(tip_y < pip_y)

        return fingers

    def classify_gesture(self):
        """
        Classify the current hand gesture based on raised fingers.
        Returns one of: 'DRAW', 'SELECT', 'ERASE', 'CLEAR', 'IDLE'
        """
        if not self.hand_detected:
            return "IDLE"

        fingers = self.fingers_up()
        thumb, index, middle, ring, pinky = fingers
        count = sum(fingers)

        # All five fingers up → Erase mode
        if count == 5:
            return "ERASE"

        # Thumb + Pinky only → Clear canvas
        if thumb and pinky and not index and not middle and not ring:
            return "CLEAR"

        # Index + Middle up, others down → Selection/Navigate mode
        if index and middle and not ring and not pinky:
            return "SELECT"

        # Only index finger up → Draw mode
        if index and not middle and not ring and not pinky:
            return "DRAW"

        # Default → Idle
        return "IDLE"

    def get_palm_center(self):
        """Get approximate center of the palm."""
        if not self.landmarks:
            return None
        palm_ids = [
            self.WRIST,
            self.INDEX_MCP,
            self.MIDDLE_MCP,
            self.RING_MCP,
            self.PINKY_MCP,
        ]
        xs = [self.landmarks[i][0] for i in palm_ids]
        ys = [self.landmarks[i][1] for i in palm_ids]
        return (int(np.mean(xs)), int(np.mean(ys)))

    def close(self):
        """Release the landmarker resources."""
        self.landmarker.close()


# Virtual Canvas (drawing buffer)
class Canvas:
    """Manages the drawing buffer — a black image where strokes are drawn."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.buffer = np.zeros((height, width, 3), dtype=np.uint8)
        # Undo history
        self.history = deque(maxlen=30)
        self._save_snapshot()

    def draw_line(self, pt1, pt2, color, thickness):
        """Draw a line between two points on the canvas."""
        if pt1 and pt2:
            cv2.line(self.buffer, pt1, pt2, color, thickness, cv2.LINE_AA)

    def erase_circle(self, center, radius):
        """Erase (paint black) in a circular region."""
        if center:
            cv2.circle(self.buffer, center, radius, (0, 0, 0), -1)

    def clear(self):
        """Clear the entire canvas."""
        self._save_snapshot()
        self.buffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)

    def undo(self):
        """Restore the last saved snapshot."""
        if self.history:
            self.buffer = self.history.pop()

    def save_checkpoint(self):
        """Manually save a checkpoint for undo."""
        self._save_snapshot()

    def _save_snapshot(self):
        self.history.append(self.buffer.copy())

    def overlay_on(self, frame):
        """
        Composite the canvas onto a camera frame.
        Only the drawn parts (non-black pixels) appear over the video.
        """
        gray = cv2.cvtColor(self.buffer, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        frame_bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
        canvas_fg = cv2.bitwise_and(self.buffer, self.buffer, mask=mask)

        return cv2.add(frame_bg, canvas_fg)

    def save_image(self, path="air_canvas_drawing.png"):
        """Save the canvas buffer to a file."""
        cv2.imwrite(path, self.buffer)
        return path


# Color Palette (header bar UI)
class ColorPalette:
    """Renders and manages the color selection bar at the top of the screen."""

    def __init__(self, frame_width):
        self.frame_width = frame_width
        self.bar_height = 80
        self.button_margin = 8

        # Define colors (BGR format for OpenCV)
        self.colors = {
            "Red":       (0, 0, 255),
            "Green":     (0, 220, 0),
            "Blue":      (255, 100, 0),
            "Yellow":    (0, 255, 255),
            "Purple":    (200, 50, 200),
            "Cyan":      (255, 255, 0),
            "Orange":    (0, 140, 255),
            "White":     (255, 255, 255),
            "Pink":      (180, 105, 255),
        }
        self.color_names = list(self.colors.keys())
        self.current_color = "Red"
        self.eraser_active = False

        # Pre-compute button positions
        num_buttons = len(self.colors) + 2  # colors + ERASER + CLEAR
        self.btn_width = (frame_width - self.button_margin * (num_buttons + 1)) // num_buttons
        self.btn_height = self.bar_height - 2 * self.button_margin
        self.buttons = self._compute_buttons()

    def _compute_buttons(self):
        """Compute (x1, y1, x2, y2, label, color) for each button."""
        buttons = []
        x = self.button_margin

        # Color buttons
        for name in self.color_names:
            x1 = x
            y1 = self.button_margin
            x2 = x1 + self.btn_width
            y2 = y1 + self.btn_height
            buttons.append((x1, y1, x2, y2, name, self.colors[name]))
            x = x2 + self.button_margin

        # Eraser button
        x1 = x
        y1 = self.button_margin
        x2 = x1 + self.btn_width
        y2 = y1 + self.btn_height
        buttons.append((x1, y1, x2, y2, "ERASER", (128, 128, 128)))
        x = x2 + self.button_margin

        # Clear button
        x1 = x
        y1 = self.button_margin
        x2 = x1 + self.btn_width
        y2 = y1 + self.btn_height
        buttons.append((x1, y1, x2, y2, "CLEAR", (50, 50, 200)))

        return buttons

    def draw(self, frame, brush_size):
        """Render the palette bar onto the frame."""
        # Semi-transparent dark background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (self.frame_width, self.bar_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        for x1, y1, x2, y2, label, color in self.buttons:
            is_selected = (
                (label == self.current_color and not self.eraser_active)
                or (label == "ERASER" and self.eraser_active)
            )

            # Draw button background
            if label == "CLEAR":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 50, 200), -1)
            elif label == "ERASER":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), -1)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                sz = min(self.btn_width, self.btn_height) // 4
                cv2.line(frame, (cx - sz, cy - sz), (cx + sz, cy + sz), (200, 200, 200), 2)
                cv2.line(frame, (cx + sz, cy - sz), (cx - sz, cy + sz), (200, 200, 200), 2)
            else:
                cv2.rectangle(frame, (x1 + 4, y1 + 4), (x2 - 4, y2 - 4), color, -1)

            # Selection highlight — glowing border
            if is_selected:
                cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 255), 3)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 100), 1)

            # Label text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = label[:5]
            text_size = cv2.getTextSize(text, font, 0.35, 1)[0]
            text_x = x1 + (self.btn_width - text_size[0]) // 2
            text_y = y2 + 1
            if text_y < self.bar_height:
                cv2.putText(frame, text, (text_x, text_y - 4), font, 0.35,
                            (200, 200, 200), 1, cv2.LINE_AA)

        # Brush size indicator
        brush_text = f"Brush: {brush_size}px"
        cv2.putText(frame, brush_text, (self.frame_width - 160, self.bar_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def check_hover(self, x, y):
        """Check if (x, y) is hovering over any palette button."""
        if y > self.bar_height:
            return None
        for x1, y1, x2, y2, label, color in self.buttons:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return label
        return None

    def select(self, label):
        """Select a color or toggle eraser."""
        if label == "ERASER":
            self.eraser_active = True
        elif label == "CLEAR":
            pass  # handled by caller
        elif label in self.colors:
            self.current_color = label
            self.eraser_active = False

    def get_current_bgr(self):
        """Get the currently selected color as BGR tuple."""
        if self.eraser_active:
            return None  # signal for eraser
        return self.colors.get(self.current_color, (255, 255, 255))


# Main Application
class AirCanvasApp:
    """Main application that ties everything together."""

    def __init__(self, camera_id=0, width=1280, height=720):
        self.camera_id = camera_id
        self.target_width = width
        self.target_height = height

        # Resolve model path (look for hand_landmarker.task in same directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "hand_landmarker.task")

        if not os.path.exists(model_path):
            print(f"[ERROR] Model file not found: {model_path}")
            print("[INFO]  Download it from:")
            print("        https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
            raise FileNotFoundError(f"Missing model: {model_path}")

        # Core components
        self.detector = HandDetector(
            model_path=model_path,
            num_hands=1,
            detection_conf=0.5,
            tracking_conf=0.5,
        )
        self.canvas = None
        self.palette = None

        # Drawing state
        self.prev_point = None
        self.brush_size = 8
        self.eraser_radius = 40
        self.min_brush = 2
        self.max_brush = 30

        # Gesture state
        self.current_gesture = "IDLE"
        self.gesture_start_time = 0
        self.clear_gesture_duration = 0.6

        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()

        # Cooldowns
        self.last_color_select_time = 0
        self.color_select_cooldown = 0.4

        # Drawing started flag (for undo checkpoints)
        self.is_drawing_stroke = False

    def run(self):
        """Main application loop."""
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        if not cap.isOpened():
            print("[ERROR] Cannot open camera. Check your webcam connection.")
            return

        # Warm up the camera — read multiple frames to let auto-exposure settle
        print("[INFO] Warming up camera...")
        frame = None
        for i in range(30):
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            if i % 10 == 0:
                print(f"[INFO] Warm-up frame {i}...")

        if frame is None:
            print("[ERROR] Cannot read from camera after retries.")
            cap.release()
            return

        h, w = frame.shape[:2]
        print(f"[INFO] Camera resolution: {w}x{h}")

        # Initialize canvas and palette with actual camera dimensions
        self.canvas = Canvas(w, h)
        self.palette = ColorPalette(w)

        print("+==================================================+")
        print("|           Air Canvas - Hand Drawing              |")
        print("+==================================================+")
        print("|  Gestures:                                       |")
        print("|    Index finger    -> Draw                       |")
        print("|    Index + Middle  -> Navigate (no draw)         |")
        print("|    Open palm       -> Erase                      |")
        print("|    Fist            -> Idle                       |")
        print("|    Thumb + Pinky   -> Clear canvas               |")
        print("|                                                  |")
        print("|  Keys: q=quit  c=clear  +/-=brush  s=save        |")
        print("|        z=undo                                    |")
        print("+==================================================+")

        # Create the display window explicitly and bring to foreground
        window_name = "Air Canvas - Hand Drawing"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, w, h)
        cv2.moveWindow(window_name, 50, 50)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        # Show initial frame to make window visible immediately
        cv2.imshow(window_name, frame)
        cv2.waitKey(100)
        # Remove always-on-top after initial show
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 0)

        print("[INFO] Window created. Starting main loop...")

        try:
            frame_num = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Mirror the frame for intuitive interaction
                frame = cv2.flip(frame, 1)

                # ── Hand detection ──
                hand_found = self.detector.detect(frame)

                if hand_found:
                    gesture = self.detector.classify_gesture()
                    self._handle_gesture(gesture, frame)
                else:
                    self.current_gesture = "IDLE"
                    self.prev_point = None
                    if self.is_drawing_stroke:
                        self.is_drawing_stroke = False

                # ── Composite canvas onto frame ──
                frame = self.canvas.overlay_on(frame)

                # ── Draw UI elements ──
                self.palette.draw(frame, self.brush_size)
                self._draw_hud(frame)
                self._draw_cursor(frame)

                # ── FPS calculation ──
                self.frame_count += 1
                frame_num += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.fps = self.frame_count / elapsed
                    self.frame_count = 0
                    self.fps_start_time = time.time()
                    print(f"[DEBUG] Frame #{frame_num} | FPS: {int(self.fps)} | Gesture: {self.current_gesture}")

                # ── Display ──
                cv2.imshow(window_name, frame)

                # ── Keyboard input ──
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    self.canvas.clear()
                    print("[INFO] Canvas cleared!")
                elif key in (ord("+"), ord("=")):
                    self.brush_size = min(self.brush_size + 2, self.max_brush)
                elif key == ord("-"):
                    self.brush_size = max(self.brush_size - 2, self.min_brush)
                elif key == ord("s"):
                    path = self.canvas.save_image()
                    print(f"[INFO] Canvas saved to: {os.path.abspath(path)}")
                elif key == ord("z"):
                    self.canvas.undo()
                    print("[INFO] Undo!")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.detector.close()
            print("[INFO] Air Canvas closed. Goodbye!")

    def _handle_gesture(self, gesture, frame):
        """Process the detected gesture and act accordingly."""
        now = time.time()
        index_tip = self.detector.get_landmark(HandDetector.INDEX_TIP)

        # ── Color selection (fingertip in palette area) ──
        if index_tip and gesture in ("DRAW", "SELECT"):
            hovered = self.palette.check_hover(index_tip[0], index_tip[1])
            if hovered and (now - self.last_color_select_time) > self.color_select_cooldown:
                if hovered == "CLEAR":
                    self.canvas.clear()
                    print("[INFO] Canvas cleared via palette!")
                else:
                    self.palette.select(hovered)
                    mode = "Eraser" if self.palette.eraser_active else hovered
                    print(f"[INFO] Selected: {mode}")
                self.last_color_select_time = now
                self.prev_point = None
                self.current_gesture = "SELECT"
                return

        # ── Gesture handling ──
        if gesture == "DRAW":
            self.current_gesture = "DRAW"
            color = self.palette.get_current_bgr()

            if color is None:
                # Eraser is active — erase instead of draw
                if index_tip:
                    self.canvas.erase_circle(index_tip, self.eraser_radius)
            else:
                # Normal drawing
                if not self.is_drawing_stroke:
                    self.canvas.save_checkpoint()
                    self.is_drawing_stroke = True

                if index_tip and index_tip[1] > self.palette.bar_height:
                    if self.prev_point and self.prev_point[1] > self.palette.bar_height:
                        dist = np.sqrt(
                            (index_tip[0] - self.prev_point[0]) ** 2
                            + (index_tip[1] - self.prev_point[1]) ** 2
                        )
                        if dist < 150:
                            self.canvas.draw_line(
                                self.prev_point, index_tip, color, self.brush_size
                            )
                    self.prev_point = index_tip
                else:
                    self.prev_point = None

        elif gesture == "SELECT":
            self.current_gesture = "SELECT"
            self.prev_point = None
            if self.is_drawing_stroke:
                self.is_drawing_stroke = False

        elif gesture == "ERASE":
            self.current_gesture = "ERASE"
            self.prev_point = None
            palm = self.detector.get_palm_center()
            if palm:
                self.canvas.erase_circle(palm, self.eraser_radius)
            if self.is_drawing_stroke:
                self.is_drawing_stroke = False

        elif gesture == "CLEAR":
            if self.current_gesture != "CLEAR":
                self.gesture_start_time = now
            self.current_gesture = "CLEAR"
            self.prev_point = None
            if (now - self.gesture_start_time) >= self.clear_gesture_duration:
                self.canvas.clear()
                self.gesture_start_time = now
                print("[INFO] Canvas cleared via gesture!")

        else:
            self.current_gesture = "IDLE"
            self.prev_point = None
            if self.is_drawing_stroke:
                self.is_drawing_stroke = False

    def _draw_hud(self, frame):
        """Draw heads-up display info on the frame."""
        h, w = frame.shape[:2]

        gesture_colors = {
            "DRAW": (0, 255, 0),
            "SELECT": (255, 200, 0),
            "ERASE": (0, 100, 255),
            "CLEAR": (0, 0, 255),
            "IDLE": (150, 150, 150),
        }
        gesture_icons = {
            "DRAW": "DRAW",
            "SELECT": "NAVIGATE",
            "ERASE": "ERASE",
            "CLEAR": "CLEARING...",
            "IDLE": "IDLE",
        }

        color = gesture_colors.get(self.current_gesture, (150, 150, 150))
        text = gesture_icons.get(self.current_gesture, "IDLE")

        # Mode badge background
        badge_w = 200
        badge_h = 40
        badge_x = 10
        badge_y = h - 55
        overlay = frame.copy()
        cv2.rectangle(overlay, (badge_x, badge_y),
                      (badge_x + badge_w, badge_y + badge_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Mode indicator circle
        cv2.circle(frame, (badge_x + 20, badge_y + badge_h // 2), 8, color, -1)
        cv2.circle(frame, (badge_x + 20, badge_y + badge_h // 2), 8, (255, 255, 255), 1)

        # Mode text
        cv2.putText(
            frame, f"Mode: {text}",
            (badge_x + 35, badge_y + badge_h // 2 + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
        )

        # FPS counter
        fps_text = f"FPS: {int(self.fps)}"
        cv2.putText(
            frame, fps_text, (w - 120, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (0, 255, 0) if self.fps > 20 else (0, 165, 255),
            2, cv2.LINE_AA,
        )

        # Clear progress bar
        if self.current_gesture == "CLEAR":
            elapsed = time.time() - self.gesture_start_time
            progress = min(elapsed / self.clear_gesture_duration, 1.0)
            bar_w = 300
            bar_x = (w - bar_w) // 2
            bar_y = h // 2 - 20

            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 30),
                          (50, 50, 50), -1)
            cv2.rectangle(frame, (bar_x, bar_y),
                          (bar_x + int(bar_w * progress), bar_y + 30),
                          (0, 0, 255), -1)
            cv2.putText(frame, "CLEARING CANVAS...",
                        (bar_x + 50, bar_y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _draw_cursor(self, frame):
        """Draw a cursor indicator at the index fingertip."""
        if not self.detector.hand_detected:
            return

        index_tip = self.detector.get_landmark(HandDetector.INDEX_TIP)
        if not index_tip:
            return

        if self.current_gesture == "DRAW":
            color = self.palette.get_current_bgr()
            if color is None:
                cv2.circle(frame, index_tip, self.eraser_radius, (255, 255, 255), 2)
                cv2.line(frame, (index_tip[0] - 10, index_tip[1]),
                         (index_tip[0] + 10, index_tip[1]), (255, 255, 255), 1)
                cv2.line(frame, (index_tip[0], index_tip[1] - 10),
                         (index_tip[0], index_tip[1] + 10), (255, 255, 255), 1)
            else:
                cv2.circle(frame, index_tip, self.brush_size // 2 + 4, color, 2)
                cv2.circle(frame, index_tip, 3, color, -1)

        elif self.current_gesture == "SELECT":
            cv2.circle(frame, index_tip, 12, (255, 200, 0), 2)
            cv2.line(frame, (index_tip[0] - 15, index_tip[1]),
                     (index_tip[0] + 15, index_tip[1]), (255, 200, 0), 1)
            cv2.line(frame, (index_tip[0], index_tip[1] - 15),
                     (index_tip[0], index_tip[1] + 15), (255, 200, 0), 1)

        elif self.current_gesture == "ERASE":
            palm = self.detector.get_palm_center()
            if palm:
                cv2.circle(frame, palm, self.eraser_radius, (0, 0, 255), 2)
                cv2.putText(frame, "ERASE",
                            (palm[0] - 25, palm[1] - self.eraser_radius - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)


# Entry Point
if __name__ == "__main__":
    app = AirCanvasApp(camera_id=0, width=1280, height=720)
    app.run()
