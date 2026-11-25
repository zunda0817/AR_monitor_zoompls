"""Smart Monitor Office Live AR (Windows).

Capture an Office window live, warp it onto the detected physical monitor in
webcam preview, and control the Office window using a fingertip.

Install (Python 3.10-3.12):
    python -m ensurepip --upgrade
    python -m pip install opencv-python numpy mediapipe==0.10.21 pyautogui pywin32 mss

Run: python smart_monitor_office_native_zoom_two_hands_low_sensitivity.py
The camera should view the SAME physical display that hosts the Office window.
Use a second monitor for the preview if possible to avoid capture recursion.
"""
import ctypes
import os
import time
import threading
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
import mss
import win32gui
import win32ui
import win32con

INTERVAL = 0.48
DETECTION_MODE = "monitor"  # "monitor" or "paper"
PAPER_MIN_AREA_RATIO = 0.035
PAPER_MIN_BRIGHTNESS = 155
PAPER_MAX_SATURATION = 80
MIN_AREA_RATIO = 0.08
MAX_JUMP = 0.10
HOLD_SECONDS = 0.38  # Faster click after holding a single index finger
STILL_RADIUS = 38.0    # Larger tolerance for finger jitter, including after zoom
SMOOTHING = 0.30

# Gesture settings (one index = pointer/swipe; two tips = native Office/Windows zoom)
GESTURE_SWIPE_DISTANCE = 0.25  # Fraction of the recognized rectangle width
GESTURE_SWIPE_MAX_SECONDS = 0.85
GESTURE_SWIPE_MAX_VERTICAL = 0.17
GESTURE_COOLDOWN = 1.2
GESTURE_TWO_FINGER_CONFIRM = 0.30
GESTURE_ZOOM_GAIN = 1.00  # Slightly more responsive, bidirectional
GESTURE_ZOOM_SMOOTHING = 0.38
GESTURE_MIN_ZOOM = 0.30
GESTURE_MAX_ZOOM = 5.0
# Office mode sends Ctrl+mouse wheel to the actual selected Office window.
# Desktop mode sends Win+Plus/Minus to Windows Magnifier (whole desktop).
ZOOM_MODE_DEFAULT = 'office'
PINCH_STEP_RATIO = 1.12  # 12% from last issued step, same for IN and OUT
PINCH_STEP_COOLDOWN = 0.24
MAX_ZOOM_STEPS_PER_FRAME = 1
GESTURE_MIN_FINGER_DISTANCE = 0.025  # Only a start guard; closing fingers must still zoom out
GESTURE_ZOOM_DEADBAND = 0.025  # Ignore small changes in hand/finger separation
CLICK_TIP_SMOOTHING = 0.30  # Less reset from fingertip jitter
CLICK_LOST_GRACE = 0.24  # seconds; tolerate a brief landmark dropout
# Zoom AMOUNT per recognized step, not detection sensitivity.
# In Office mode, one gesture zoom step now sends 3 Ctrl+wheel ticks.
OFFICE_WHEEL_TICKS_PER_STEP = 5
# In Desktop mode, one gesture step presses Win+Plus/Minus twice.
DESKTOP_MAGNIFIER_KEYS_PER_STEP = 2
# First try the external camera (commonly index 1), then the built-in camera (commonly 0).
# OpenCV indices are not guaranteed to identify the physical device. Set
# SMART_CAMERA_ORDER=0,1,2,3 in PowerShell if your external camera is index 0.
CAMERA_ORDER = (1, 0, 2, 3)
# High-quality preview settings. If camera cannot provide frames, revert to defaults.
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30
CAPTURE_FPS = 18         # Office capture is asynchronous
OFFICE_MAX_WIDTH = 1600  # Keep text sharp without extreme scaling
DETECT_SCALE = 0.56      # Detect shape on smaller frames
HAND_SCALE = 0.62        # MediaPipe inference on smaller frames
DISPLAY_MAX_WIDTH = 1440 # Limit costly preview scaling

AR_OPACITY = 1.0  # 1.0 = fully sharp Office overlay, no camera bleed-through
AR_INTERPOLATION = cv2.INTER_LINEAR  # Fast and usually best for text
WINDOW_TITLE_KEYWORDS = ('powerpoint', 'excel', 'word', 'wordpad', 'onenote', 'office')

def order_points(p):
    p = np.asarray(p, np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)],
                     p[np.argmax(s)], p[np.argmin(d)]], np.float32)

def detect(frame, previous=None):
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.GaussianBlur(enhanced, (5, 5), 0)
    candidates = []
    debug = np.zeros_like(gray)

    # Use both adaptive contrast and raw luminance; monitor need not be black.
    for source in (enhanced, cv2.GaussianBlur(gray, (5, 5), 0)):
        for low, high in ((35, 100), (65, 170)):
            edges = cv2.Canny(source, low, high)
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                     cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
            debug = cv2.bitwise_or(debug, edges)
            contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if not (MIN_AREA_RATIO * w * h < area < .94 * w * h):
                    continue
                peri = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, .025 * peri, True)
                if len(approx) == 4 and cv2.isContourConvex(approx):
                    quad = order_points(approx[:, 0, :])
                else:
                    # Only use minAreaRect if the contour is already quite rectangular.
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    rect_area = max(rect[1][0] * rect[1][1], 1)
                    if area / rect_area < .78:
                        continue
                    quad = order_points(box)
                side = [np.linalg.norm(quad[(i+1)%4] - quad[i]) for i in range(4)]
                if min(side) < 55:
                    continue
                # Reject long, narrow artifacts from glare.
                aspect = max(side) / max(min(side), 1)
                if aspect > 4.5:
                    continue
                # Prefer near-previous detections to avoid jumping to reflections.
                displacement = (np.mean(np.linalg.norm(quad - previous, axis=1))
                                if previous is not None else 0)
                score = area / (w*h) - 2.0 * displacement / np.hypot(w, h)
                candidates.append((score, quad))

    if not candidates:
        return None, debug
    candidates.sort(key=lambda v: v[0], reverse=True)
    best = candidates[0][1]
    if previous is not None:
        delta = np.mean(np.linalg.norm(best - previous, axis=1)) / np.hypot(w, h)
        if delta > MAX_JUMP:
            return None, debug
    return best, debug

def detect_paper(frame, previous=None):
    """Find a large, nearly-white rectangular sheet, robust to uneven light.

    Returns four corners in TL, TR, BR, BL order, and a diagnostic mask.
    """
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # A white sheet normally has low saturation and high brightness. Include
    # well-lit light gray areas to tolerate shadows across the paper.
    saturation = hsv[:, :, 1]
    brightness = hsv[:, :, 2]
    white = np.uint8((saturation < PAPER_MAX_SATURATION) &
                     (brightness > PAPER_MIN_BRIGHTNESS)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, kernel, iterations=2)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, kernel, iterations=1)

    # Detect the outer boundary rather than inner text / graphics.
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    frame_area = float(w * h)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (PAPER_MIN_AREA_RATIO * frame_area <= area < 0.94 * frame_area):
            continue
        perimeter = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * perimeter, True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            # A convex hull can recover corners when the white region has
            # minor notches, e.g. from shadows on the edge.
            hull = cv2.convexHull(cnt)
            approx = cv2.approxPolyDP(hull, 0.028 * cv2.arcLength(hull, True), True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
        quad = order_points(approx[:, 0, :])
        sides = [np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)]
        if min(sides) < 45 or max(sides) / max(min(sides), 1) > 4.0:
            continue
        # Check the whiteness of the candidate's interior, not just its edges.
        mask = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(mask, np.round(quad).astype(np.int32), 255)
        filled = cv2.countNonZero(mask)
        whiteness = cv2.mean(white, mask=mask)[0] / 255.0
        if filled == 0 or whiteness < 0.68:
            continue
        # Prefer large white rectangles and continuity across frames.
        displacement = (np.mean(np.linalg.norm(quad - previous, axis=1))
                        if previous is not None else 0.0)
        score = area / frame_area * (0.55 + 0.45 * whiteness) - \
                2.0 * displacement / np.hypot(w, h)
        candidates.append((score, quad))

    if not candidates:
        return None, white
    best = max(candidates, key=lambda item: item[0])[1]
    if previous is not None:
        jump = np.mean(np.linalg.norm(best - previous, axis=1)) / np.hypot(w, h)
        if jump > MAX_JUMP:
            return None, white
    return best, white


def choose_window():
    """Ask user to choose a visible Office window, by HWND."""
    windows = []
    def enum(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd) or win32gui.IsIconic(hwnd):
            return
        title = win32gui.GetWindowText(hwnd).strip()
        if title and (any(word in title.lower() for word in WINDOW_TITLE_KEYWORDS)
                      or 'slide show' in title.lower()
                      or 'スライドショー' in title.lower()
                      or win32gui.GetClassName(hwnd).lower() == 'screenclass'):
            l, t, r, b = win32gui.GetWindowRect(hwnd)
            if r-l >= 180 and b-t >= 130:
                windows.append((hwnd, title))
    win32gui.EnumWindows(enum, None)
    if not windows:
        raise RuntimeError('Office window not found. Open a PowerPoint, Excel, or Word file first.')
    print('\nAvailable Office windows:')
    for i, (_, title) in enumerate(windows, 1):
        print(f'  {i}: {title}')
    while True:
        value = input('Select number [1]: ').strip() or '1'
        if value.isdigit() and 1 <= int(value) <= len(windows):
            return windows[int(value)-1][0]
        print('Invalid number.')


def window_rect(hwnd):
    if not win32gui.IsWindow(hwnd) or win32gui.IsIconic(hwnd):
        return None
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    if r <= l or b <= t:
        return None
    return (l, t, r, b)



def print_window(hwnd, rect):
    """Win32 capture. GDI objects created/deleted inside the capture worker."""
    l, t, r, b = rect
    width, height = r-l, b-t
    if width < 2 or height < 2 or width > 5000 or height > 4000:
        return None
    hwnd_dc = mem_dc = bmp = source_dc = None
    try:
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        source_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        mem_dc = source_dc.CreateCompatibleDC()
        bmp = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(source_dc, width, height)
        mem_dc.SelectObject(bmp)
        ok = ctypes.windll.user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), 2)
        if not ok:
            ok = ctypes.windll.user32.PrintWindow(hwnd, mem_dc.GetSafeHdc(), 0)
        if not ok:
            return None
        data = bmp.GetBitmapBits(True)
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
        bgr = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if float(np.std(bgr)) < 3.0:
            return None
        return bgr
    except Exception:
        return None
    finally:
        if bmp is not None:
            try: win32gui.DeleteObject(bmp.GetHandle())
            except Exception: pass
        if mem_dc is not None:
            try: mem_dc.DeleteDC()
            except Exception: pass
        # GetWindowDC is borrowed; release it with ReleaseDC, not DeleteDC.
        if hwnd_dc is not None:
            try: win32gui.ReleaseDC(hwnd, hwnd_dc)
            except Exception: pass


class OfficeCaptureWorker:
    """Keep slow PrintWindow/MSS calls off the camera/hand tracking thread."""
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = (None, None, 'waiting')
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def get(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1.5)

    def _run(self):
        try:
            # MSS objects must be created and used in the SAME thread.
            with mss.mss() as sct:
                period = 1.0 / max(1, CAPTURE_FPS)
                while not self._stop.is_set():
                    start = time.monotonic()
                    rect = window_rect(self.hwnd)
                    image = None
                    mode = 'unavailable'
                    if rect is not None:
                        image = print_window(self.hwnd, rect)
                        mode = 'PrintWindow'
                        if image is None:
                            try:
                                l, t, r, b = rect
                                shot = np.asarray(sct.grab({
                                    'left': l, 'top': t,
                                    'width': r-l, 'height': b-t
                                }))
                                image = cv2.cvtColor(shot, cv2.COLOR_BGRA2BGR)
                                mode = 'MSS visible'
                            except Exception:
                                mode = 'capture unavailable'
                        if image is not None and image.shape[1] > OFFICE_MAX_WIDTH:
                            scale = OFFICE_MAX_WIDTH / image.shape[1]
                            image = cv2.resize(image, (OFFICE_MAX_WIDTH,
                                max(2, round(image.shape[0]*scale))),
                                interpolation=cv2.INTER_AREA)
                    # Reuse previous capture if a single cycle fails; keep its rect
                    # paired with the image to preserve coordinate correspondence.
                    if image is not None:
                        with self._lock:
                            self._latest = (image, rect, mode)
                    elif rect is None:
                        with self._lock:
                            self._latest = (None, None, mode)
                    self._stop.wait(max(0.001, period-(time.monotonic()-start)))
        except Exception as exc:
            with self._lock:
                self._latest = (None, None, f'capture error: {exc}')


def target_units(point, corners):
    """Camera point -> normalized position on the tracked target plane."""
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    transform = cv2.getPerspectiveTransform(corners.astype(np.float32), unit)
    return cv2.perspectiveTransform(
        np.array([[point]], dtype=np.float32), transform
    )[0, 0]


class GestureController:
    """Interpret either one hand's index+middle or two hands' index fingertips.

    Priority: two detected hands with extended index fingers > one-hand two fingers.
    One extended index (middle folded) provides pointer, dwell click and slide swipe.
    The two recognized points are returned for dot drawing only (no connecting line).
    """

    def __init__(self):
        self.zoom = 1.0
        self.mode = 'none'
        self.swipe_start = None
        self.swipe_started = 0.0
        self.last_swipe = -1e10
        self.two_started = None
        self.zoom_base = None
        self.zoom_distance = None
        self.zoom_source = None
        self.swipe_suppressed_until = 0.0

    def reset(self, keep_zoom=False):
        if not keep_zoom:
            self.zoom = 1.0
        self.mode = 'none'
        self.swipe_start = None
        self.two_started = None
        self.zoom_base = None
        self.zoom_distance = None
        self.zoom_source = None
        self.swipe_suppressed_until = 0.0

    @staticmethod
    def extended(points, tip_id, pip_id):
        wrist = points[0]
        tip = points[tip_id]
        pip = points[pip_id]
        return float(np.linalg.norm(tip - wrist)) > float(np.linalg.norm(pip - wrist)) * 1.12

    @staticmethod
    def middle_clearly_extended(points):
        """Stricter middle-finger test to avoid accidental zoom while clicking.

        A curled middle finger often passes a wrist-distance-only test.
        Require both length extension and an approximately straight PIP joint.
        """
        wrist = points[0]
        tip, pip, mcp = points[12], points[10], points[9]
        length_extended = (float(np.linalg.norm(tip - wrist)) >
                           float(np.linalg.norm(pip - wrist)) * 1.18)
        a, b = mcp - pip, tip - pip
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom < 1e-5:
            return length_extended
        cosine = float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))
        # Straight finger = ~180 deg, curled finger = much smaller angle.
        return length_extended and cosine < -0.72

    @staticmethod
    def within_target(uv):
        return bool(-0.03 <= uv[0] <= 1.03 and -0.03 <= uv[1] <= 1.03)

    def _begin_or_update_zoom(self, pt1, pt2, corners, now, source):
        """Accumulate low-sensitivity zoom without jumping across gesture types."""
        uv1 = target_units(pt1, corners)
        uv2 = target_units(pt2, corners)
        if not (self.within_target(uv1) and self.within_target(uv2)):
            self.reset(keep_zoom=True)
            return 'none', None, None, None

        self.swipe_start = None
        self.swipe_suppressed_until = now + 0.6
        if self.zoom_source != source:
            # A change from one-hand to two-hand must NOT inherit old spacing.
            self.zoom_source = source
            self.two_started = now
            self.zoom_distance = None
            self.zoom_base = None
        if self.two_started is None:
            self.two_started = now
        if now - self.two_started < GESTURE_TWO_FINGER_CONFIRM:
            self.mode = 'two_pending'
            return self.mode, pt1, pt2, None

        distance = float(np.linalg.norm(uv1 - uv2))
        if self.zoom_distance is None:
            if distance < GESTURE_MIN_FINGER_DISTANCE:
                self.mode = 'two_pending'
                return self.mode, pt1, pt2, None
            self.zoom_distance = distance
            self.zoom_base = self.zoom

        # IMPORTANT: do not re-baseline at short distances. Otherwise a user
        # can zoom IN but closing their fingers never reaches a zoom OUT step.
        ratio = max(distance, 0.006) / max(self.zoom_distance, 0.001)
        # Dead zone absorbs small landmark jitter / natural hand trembling.
        if abs(ratio - 1.0) <= GESTURE_ZOOM_DEADBAND:
            proposed = self.zoom_base
        else:
            # Subtract the deadband rather than introducing a sudden value jump.
            adjusted = 1.0 + np.sign(ratio - 1.0) * (abs(ratio - 1.0) - GESTURE_ZOOM_DEADBAND)
            proposed = self.zoom_base * max(0.05, adjusted) ** GESTURE_ZOOM_GAIN
        proposed = float(np.clip(proposed, GESTURE_MIN_ZOOM, GESTURE_MAX_ZOOM))
        self.zoom = ((1.0 - GESTURE_ZOOM_SMOOTHING) * self.zoom
                     + GESTURE_ZOOM_SMOOTHING * proposed)
        self.mode = 'zoom'
        return self.mode, pt1, pt2, None

    def update(self, hands_landmarks, width, height, corners, now):
        """Return (mode, first_tip, second_tip, slide_action).

        hands_landmarks is the full list from MediaPipe, up to two detected hands.
        """
        if not hands_landmarks or corners is None:
            self.reset(keep_zoom=True)
            return 'none', None, None, None

        hands = []
        for lm in hands_landmarks[:2]:
            points = np.array([(p.x * width, p.y * height)
                               for p in lm.landmark], dtype=np.float32)
            hands.append({
                'index': tuple(map(int, points[8])),
                'middle': tuple(map(int, points[12])),
                'index_up': self.extended(points, 8, 6),
                'middle_up': self.middle_clearly_extended(points),
            })

        if len(hands) == 2:
            # Two extended index fingers have priority for pinch zoom.
            left, right = sorted(hands, key=lambda h: h['index'][0])
            if left['index_up'] and right['index_up']:
                return self._begin_or_update_zoom(
                    left['index'], right['index'], corners, now, 'two_hands')
            # An idle second hand must not disable one-finger clicks.
            eligible = [h for h in hands if h['index_up']]
            if len(eligible) != 1:
                self.reset(keep_zoom=True)
                return 'none', None, None, None
            one = eligible[0]
        else:
            one = hands[0]
        index, middle = one['index'], one['middle']
        if not one['index_up']:
            self.reset(keep_zoom=True)
            return 'none', None, None, None

        if one['middle_up']:
            return self._begin_or_update_zoom(index, middle, corners, now, 'one_hand')

        # Returning from pinch should never create an accidental swipe/click.
        if self.zoom_source is not None:
            self.swipe_suppressed_until = now + 0.6
        self.zoom_source = None
        self.two_started = None
        self.zoom_distance = None
        self.zoom_base = None
        index_uv = target_units(index, corners)
        if not self.within_target(index_uv):
            self.swipe_start = None
            self.mode = 'none'
            return self.mode, None, None, None
        if now < self.swipe_suppressed_until:
            self.swipe_start = None
            self.mode = 'one'
            return self.mode, index, None, None

        if self.swipe_start is None or now - self.swipe_started > GESTURE_SWIPE_MAX_SECONDS:
            self.swipe_start = index_uv.copy()
            self.swipe_started = now
        delta = index_uv - self.swipe_start
        action = None
        if (now - self.last_swipe >= GESTURE_COOLDOWN and
                abs(float(delta[0])) >= GESTURE_SWIPE_DISTANCE and
                abs(float(delta[1])) <= GESTURE_SWIPE_MAX_VERTICAL and
                now - self.swipe_started <= GESTURE_SWIPE_MAX_SECONDS):
            action = 'next' if delta[0] < 0 else 'previous'
            self.last_swipe = now
            self.swipe_start = None
            self.swipe_suppressed_until = now + 0.55
        self.mode = 'one'
        return self.mode, index, None, action


class NativeZoomDispatcher:
    """Translate relative two-finger pinch motion to application/desktop zoom.

    The internal pinch value is *not* the Office application's actual zoom level.
    It is only a threshold accumulator; the app/OS owns the real zoom state.
    """

    def __init__(self, mode=ZOOM_MODE_DEFAULT):
        self.mode = mode
        self.anchor = 1.0
        self.last_sent = -1e10

    def reset(self, relative=1.0):
        self.anchor = max(0.01, float(relative))
        self.last_sent = -1e10

    def toggle_mode(self, relative=1.0):
        self.mode = 'desktop' if self.mode == 'office' else 'office'
        self.reset(relative)
        return self.mode

    def dispatch(self, relative_zoom, now, hwnd, screen_point):
        if now - self.last_sent < PINCH_STEP_COOLDOWN:
            return None
        relative_zoom = max(.01, float(relative_zoom))
        ratio = relative_zoom / max(.01, self.anchor)
        direction = 1 if ratio >= PINCH_STEP_RATIO else (-1 if ratio <= 1 / PINCH_STEP_RATIO else 0)
        if not direction:
            return None

        # Avoid a large burst of scroll events when tracking briefly disappears.
        steps = min(MAX_ZOOM_STEPS_PER_FRAME, max(1,
            int(abs(np.log(ratio)) / np.log(PINCH_STEP_RATIO))))
        try:
            if self.mode == 'office':
                if not win32gui.IsWindow(hwnd) or win32gui.IsIconic(hwnd):
                    return 'OFFICE WINDOW UNAVAILABLE'
                # Ctrl+wheel is a native zoom command in Office editing views.
                # Focus and pointer positioning matter: otherwise events may land
                # on the OpenCV preview, VS Code, or another app.
                try:
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
                if win32gui.GetForegroundWindow() != hwnd:
                    return 'CLICK OFFICE WINDOW FIRST'
                if screen_point is not None:
                    x, y = map(int, screen_point)
                    pyautogui.moveTo(x, y, duration=0)
                pyautogui.keyDown('ctrl')
                try:
                    pyautogui.scroll(direction * steps * OFFICE_WHEEL_TICKS_PER_STEP)
                finally:
                    pyautogui.keyUp('ctrl')
                message = 'OFFICE ZOOM IN' if direction > 0 else 'OFFICE ZOOM OUT'
            else:
                # Win + Plus/Minus changes the actual Windows Magnifier level.
                # It is not Office's document zoom and may disrupt target tracking.
                for _ in range(steps * DESKTOP_MAGNIFIER_KEYS_PER_STEP):
                    pyautogui.hotkey('winleft', 'add' if direction > 0 else 'subtract')
                message = 'DESKTOP MAGNIFY +' if direction > 0 else 'DESKTOP MAGNIFY -'
        except pyautogui.FailSafeException:
            return 'FAILSAFE: MOUSE CONTROL STOPPED'
        except Exception as exc:
            print('Native zoom send error:', exc)
            return 'ZOOM SEND FAILED'

        self.last_sent = now
        # Set baseline to the just-observed value, NOT an idealized target.
        # This gives equal travel thresholds for zooming IN and reversing OUT.
        self.anchor = relative_zoom
        return message


def change_slide(direction, hwnd):
    """Control slideshow via PowerPoint COM; fallback to window key message.

    Must be called from the main GUI thread. The fallback may not be honored
    by every PowerPoint version, so return success only for the COM path.
    """
    try:
        import win32com.client
        powerpoint = win32com.client.GetActiveObject('PowerPoint.Application')
        if powerpoint.SlideShowWindows.Count > 0:
            slide_view = powerpoint.SlideShowWindows(1).View
            if direction == 'next':
                slide_view.Next()
            else:
                slide_view.Previous()
            return True, 'PowerPoint slideshow'
    except Exception as exc:
        print('PowerPoint COM slideshow navigation unavailable:', exc)
    try:
        vk = win32con.VK_RIGHT if direction == 'next' else win32con.VK_LEFT
        win32gui.PostMessage(hwnd, win32con.WM_KEYDOWN, vk, 0)
        win32gui.PostMessage(hwnd, win32con.WM_KEYUP, vk, 0xC0000001)
        return False, 'Window message sent (not confirmed)'
    except Exception as exc:
        print('Slide control failed:', exc)
        return False, 'Slide control unavailable'


def compose_ar(frame, office, corners):
    """Warp *only* the bounding box of the target, not the whole camera image."""
    h, w = frame.shape[:2]
    oh, ow = office.shape[:2]
    polygon = corners.astype(np.float32)
    x0 = max(0, int(np.floor(polygon[:, 0].min())))
    y0 = max(0, int(np.floor(polygon[:, 1].min())))
    x1 = min(w, int(np.ceil(polygon[:, 0].max()))+1)
    y1 = min(h, int(np.ceil(polygon[:, 1].max()))+1)
    if x1 <= x0 or y1 <= y0:
        return frame
    src = np.float32([[0, 0], [ow-1, 0], [ow-1, oh-1], [0, oh-1]])
    dst = polygon - np.float32([x0, y0])
    H = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(office, H, (x1-x0, y1-y0),
                                 flags=AR_INTERPOLATION)
    mask = np.zeros((y1-y0, x1-x0), np.uint8)
    cv2.fillConvexPoly(mask, np.round(dst).astype(np.int32), 255)
    roi = frame[y0:y1, x0:x1]
    if AR_OPACITY >= 0.999:
        cv2.copyTo(warped, mask, roi)
    else:
        blended = cv2.addWeighted(roi, 1-AR_OPACITY, warped, AR_OPACITY, 0)
        cv2.copyTo(blended, mask, roi)
    return frame


def camera_tip_to_screen(tip, corners, rect):
    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    H = cv2.getPerspectiveTransform(corners.astype(np.float32), unit)
    u, v = cv2.perspectiveTransform(np.array([[tip]], np.float32), H)[0, 0]
    if not (-0.02 <= u <= 1.02 and -0.02 <= v <= 1.02):
        return None
    l, t, r, b = rect
    return (int(l + np.clip(u, .005, .995) * (r-l)),
            int(t + np.clip(v, .005, .995) * (b-t)))

def camera_candidates(exclude=None):
    """Camera indices, overridable with SMART_CAMERA_ORDER=1,0,2,3."""
    raw = os.environ.get('SMART_CAMERA_ORDER', '').strip()
    try:
        order = [int(part.strip()) for part in raw.split(',')] if raw else list(CAMERA_ORDER)
        if any(i < 0 for i in order):
            raise ValueError('negative camera index')
    except ValueError:
        print('Invalid SMART_CAMERA_ORDER; using default camera order.')
        order = list(CAMERA_ORDER)
    order = list(dict.fromkeys(order))
    if exclude is not None and exclude in order:
        order.remove(exclude)
        order.append(exclude)
    return order


def open_camera(exclude=None):
    """Try connected cameras; success requires a real decoded video frame."""
    backends = [('DirectShow', cv2.CAP_DSHOW),
                ('Media Foundation', cv2.CAP_MSMF),
                ('Automatic', cv2.CAP_ANY)]
    for idx in camera_candidates(exclude):
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(idx, backend)
            if not cap.isOpened():
                cap.release()
                continue
            # Do not force MJPG; start with device defaults, which work on more cameras.
            ok = False
            frame = None
            for _ in range(3):
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    break
            if not ok or frame is None or frame.size == 0:
                cap.release()
                continue
            # Request high resolution only AFTER the device has provided a frame.
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            new_ok, new_frame = cap.read()
            if not new_ok or new_frame is None or new_frame.size == 0:
                print(f'Camera {idx}: high-resolution setting failed; reopening defaults.')
                cap.release()
                cap = cv2.VideoCapture(idx, backend)
                if not cap.isOpened():
                    cap.release()
                    continue
                new_ok, new_frame = cap.read()
                if not new_ok or new_frame is None or new_frame.size == 0:
                    cap.release()
                    continue
            height, width = new_frame.shape[:2]
            print(f'Using camera {idx} ({backend_name}): {width}x{height}')
            return cap, idx
    raise RuntimeError(
        'No usable camera found. Close Teams/Zoom/Camera apps; check Windows camera permissions. '
        'Try SMART_CAMERA_ORDER=0,1,2,3 if your camera uses a different index.'
    )


def main():
    if not hasattr(mp, 'solutions'):
        raise RuntimeError('Use mediapipe==0.10.21 with Python 3.10-3.12.')
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try: ctypes.windll.user32.SetProcessDPIAware()
        except Exception: pass

    cv2.setNumThreads(2)
    hwnd = choose_window()
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0
    cap, camera_index = open_camera()
    # DSHOW/MSMF do not always honor this, but it can reduce frame latency.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    office_worker = OfficeCaptureWorker(hwnd)
    office_worker.start()

    corners = None
    detection_mode = DETECTION_MODE
    manual_mode = manual_locked = show_edges = control_enabled = False
    manual_points = []
    stable_candidate = None
    stable_count = 0
    last_scan = -1e10
    last_debug = None
    tip_start = hover_start = None
    clicked = False
    smooth_xy = None
    click_filtered_tip = None
    click_last_seen = -1e10
    gestures = GestureController()
    native_zoom = NativeZoomDispatcher()
    gesture_feedback = ''
    feedback_until = 0.0
    fps_avg = None
    last_fps_time = time.monotonic()
    window_name = 'Smart Monitor Office Live AR - HQ Smooth'
    displayed_size = (0, 0)

    def on_mouse(event, x, y, flags, param):
        nonlocal corners, manual_mode, manual_locked, manual_points
        if not (manual_mode and event == cv2.EVENT_LBUTTONDOWN):
            return
        # Scale displayed preview coords back to full-resolution camera coords.
        fw, fh = displayed_size
        if fw <= 0 or fh <= 0:
            return
        sx, sy = current_frame_size[0]/fw, current_frame_size[1]/fh
        manual_points.append((x*sx, y*sy))
        if len(manual_points) == 4:
            corners = order_points(manual_points)
            manual_points = []
            manual_mode = False
            manual_locked = True
            print('Manual four corners locked; a=auto detect.')

    current_frame_size = [0, 0]
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window_name, on_mouse)
    print('1=monitor 2=white paper e=mouse ON/OFF m=manual a=auto')
    print('d=debug edges s=screenshot g=zoom mode q/ESC=quit')
    print('GESTURES: one index=mouse/swipe; index+middle OR both indexes=real zoom')
    print('Swipe LEFT=next slide, RIGHT=previous; PowerPoint slideshow required')
    print('ZOOM: Office Ctrl+Wheel (3 ticks/step), or G=Windows Magnifier')
    print('CLICK: return to ONE index finger after pinch, hold ~0.38 seconds')
    print('Note: capture FPS and camera preview FPS are separate.')
    try:
        with mp.solutions.hands.Hands(
            static_image_mode=False, max_num_hands=2, model_complexity=0,
            min_detection_confidence=0.60, min_tracking_confidence=0.60
        ) as hands:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None or frame.size == 0:
                    print('Camera frames stopped; trying other camera...')
                    cap.release()
                    try:
                        cap, camera_index = open_camera(exclude=camera_index)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    except RuntimeError as exc:
                        print(exc)
                        break
                    corners = None
                    stable_candidate = None
                    stable_count = 0
                    last_scan = -1e10
                    control_enabled = False
                    manual_mode = manual_locked = False
                    manual_points.clear()
                    gestures.reset()
                    continue
                now = time.monotonic()
                h, w = frame.shape[:2]
                current_frame_size[:] = [w, h]

                # Heavy detection runs on a reduced frame only occasionally.
                if not manual_mode and not manual_locked and now-last_scan >= INTERVAL:
                    last_scan = now
                    ds = min(1.0, DETECT_SCALE)
                    small = cv2.resize(frame, (max(1, round(w*ds)),
                                                      max(1, round(h*ds))),
                                       interpolation=cv2.INTER_AREA)
                    prev_small = corners*ds if corners is not None else None
                    if detection_mode == 'monitor':
                        found, last_debug = detect(small, prev_small)
                    else:
                        found, last_debug = detect_paper(small, prev_small)
                    if found is not None:
                        found = found/ds
                        if corners is None:
                            if stable_candidate is not None and np.mean(
                                np.linalg.norm(found-stable_candidate, axis=1)) < 22:
                                stable_count += 1
                            else:
                                stable_count = 1
                            stable_candidate = found
                            if stable_count >= 2:
                                corners = found
                        else:
                            corners = (.80*corners+.20*found).astype(np.float32)

                office, rect, capture_mode = office_worker.get()
                view = frame.copy()
                if corners is not None and office is not None:
                    compose_ar(view, office, corners)
                    cv2.polylines(view, [np.round(corners).astype(np.int32)],
                                  True, (0, 255, 0), 2)

                # MediaPipe sees a smaller frame; project landmarks back into camera coordinates.
                hs = min(1.0, HAND_SCALE)
                hand_frame = cv2.resize(frame, (max(1, round(w*hs)),
                                                       max(1, round(h*hs))),
                                        interpolation=cv2.INTER_AREA)
                rgb = cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                result = hands.process(rgb)
                detected_hands = result.multi_hand_landmarks or []
                gesture_mode, index_tip, middle_tip, slide_action = gestures.update(
                    detected_hands if control_enabled else [],
                    w, h, corners if not manual_mode else None, now)

                # A recognized gesture needs a tracked target and mouse/gesture ON.
                if not control_enabled or rect is None:
                    slide_action = None
                if slide_action is not None:
                    confirmed, message = change_slide(slide_action, hwnd)
                    gesture_feedback = (
                        ('NEXT' if slide_action == 'next' else 'PREVIOUS')
                        + (' SLIDE' if confirmed else ' (sent, unconfirmed)'))
                    feedback_until = now + 2.0
                    print(gesture_feedback, '-', message)

                # OS/application zoom uses the midpoint of either recognized finger pair.
                # This does not crop, warp, or digitally magnify the AR overlay.
                if gesture_mode == 'zoom' and control_enabled and rect is not None and not manual_mode:
                    mid = ((index_tip[0] + middle_tip[0]) / 2,
                           (index_tip[1] + middle_tip[1]) / 2)
                    zoom_point = camera_tip_to_screen(mid, corners, rect)
                    feedback = native_zoom.dispatch(gestures.zoom, now, hwnd, zoom_point)
                    if feedback:
                        gesture_feedback = feedback
                        feedback_until = now + 1.2
                        if feedback == 'FAILSAFE: MOUSE CONTROL STOPPED':
                            control_enabled = False
                elif gesture_mode != 'zoom':
                    # Allow fresh pinch sequences without initial jumps.
                    native_zoom.reset(gestures.zoom)

                # Zoom and swipes never move the pointer or trigger dwell click.
                tip = index_tip if gesture_mode == 'one' else None
                target = None
                if (tip is not None and rect is not None and corners is not None
                        and not manual_mode and slide_action is None):
                    target = camera_tip_to_screen(tip, corners, rect)
                if target is None:
                    # Keep dwell state over very brief landmark dropouts, but
                    # never click when no valid one-finger target is present.
                    if gesture_mode in ('zoom', 'two_pending') or now - click_last_seen > CLICK_LOST_GRACE:
                        tip_start = hover_start = None
                        clicked = False
                        click_filtered_tip = None
                        smooth_xy = None
                elif control_enabled:
                    click_last_seen = now
                    xy = np.array(target, dtype=np.float32)
                    smooth_xy = xy if smooth_xy is None else (
                        (1-SMOOTHING)*smooth_xy+SMOOTHING*xy)
                    can_click = True
                    try:
                        pyautogui.moveTo(int(smooth_xy[0]), int(smooth_xy[1]), duration=0)
                    except pyautogui.FailSafeException:
                        control_enabled = False
                        can_click = False
                        print('Emergency stop: mouse control OFF')
                    raw_tip = np.array(tip, dtype=np.float32)
                    click_filtered_tip = raw_tip if click_filtered_tip is None else (
                        (1-CLICK_TIP_SMOOTHING)*click_filtered_tip
                        + CLICK_TIP_SMOOTHING*raw_tip)
                    # Compare filtered coordinates to reduce reset due to jitter.
                    if tip_start is None or np.linalg.norm(click_filtered_tip - tip_start) > STILL_RADIUS:
                        tip_start = click_filtered_tip.copy()
                        hover_start = now
                        clicked = False
                    elif (can_click and not clicked and hover_start is not None
                          and now-hover_start >= HOLD_SECONDS and control_enabled):
                        try:
                            pyautogui.click()
                            clicked = True
                        except pyautogui.FailSafeException:
                            control_enabled = False
                else:
                    tip_start = hover_start = None
                    clicked = False
                    click_filtered_tip = None
                    smooth_xy = None

                if index_tip is not None:
                    cv2.circle(view, index_tip, 8, (0, 0, 255), -1)
                    cv2.circle(view, index_tip, 13, (255, 255, 255), 2)
                if (middle_tip is not None and
                        gesture_mode in ('zoom', 'two_pending')):
                    cv2.circle(view, middle_tip, 8, (255, 140, 0), -1)
                    cv2.circle(view, middle_tip, 13, (255, 255, 255), 2)
                if gesture_mode == 'zoom':
                    cv2.putText(view, f'PINCH {gestures.zoom_source.upper()} / {native_zoom.mode.upper()} ({gestures.zoom:.2f})',
                                (12, 112), cv2.FONT_HERSHEY_SIMPLEX,
                                .7, (0, 255, 255), 2)
                elif now < feedback_until:
                    cv2.putText(view, gesture_feedback, (12, 112),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                .65, (0, 255, 255), 2)
                elapsed = now-last_fps_time
                if elapsed > 0:
                    fps = min(200.0, 1/elapsed)
                    fps_avg = fps if fps_avg is None else .88*fps_avg+.12*fps
                last_fps_time = now
                mode_name = 'MONITOR' if detection_mode == 'monitor' else 'PAPER'
                state = 'ON' if control_enabled else 'OFF'
                cv2.putText(view, f'{mode_name} [1/2]  MOUSE {state} [e]  FPS {fps_avg:.1f}',
                            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 255, 255), 2)
                cv2.putText(view, f'Office: {capture_mode}', (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 220, 0), 2)
                if corners is None:
                    cv2.putText(view, f'Searching {mode_name}...', (12, 85),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 180, 255), 2)
                if manual_mode:
                    cv2.putText(view, f'Click TL TR BR BL: {len(manual_points)}/4',
                                (12, 85), cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 180, 255), 2)
                    for px, py in manual_points:
                        cv2.circle(view, (round(px), round(py)), 6, (0, 180, 255), -1)
                if control_enabled and hover_start is not None and not clicked:
                    pct = min(100, int((now-hover_start)/HOLD_SECONDS*100))
                    cv2.putText(view, f'CLICK HOLD {pct}%', (12, h-20),
                                cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 220, 255), 2)

                # Display at native camera resolution up to DISPLAY_MAX_WIDTH.
                disp_scale = min(1.0, DISPLAY_MAX_WIDTH/w)
                if disp_scale < 1.0:
                    display = cv2.resize(view, (round(w*disp_scale),
                                                        round(h*disp_scale)),
                                         interpolation=cv2.INTER_AREA)
                else:
                    display = view
                displayed_size = (display.shape[1], display.shape[0])
                cv2.imshow(window_name, display)
                if show_edges and last_debug is not None:
                    cv2.imshow('Detected Edges (low-res)', last_debug)
                key = cv2.waitKey(1) & 0xff
                if key in (ord('q'), 27):
                    break
                elif key in (ord('1'), ord('2')):
                    new_mode = 'monitor' if key == ord('1') else 'paper'
                    if detection_mode != new_mode:
                        detection_mode = new_mode
                        corners = stable_candidate = None
                        stable_count = 0
                        last_scan = -1e10
                        manual_mode = manual_locked = False
                        manual_points.clear()
                        last_debug = None
                        control_enabled = False
                        smooth_xy = None
                        tip_start = hover_start = None
                        clicked = False
                        gestures.reset()
                        print('Mode:', detection_mode, '(mouse OFF)')
                elif key == ord('e'):
                    if corners is None:
                        print('Detect target before enabling mouse.')
                    else:
                        control_enabled = not control_enabled
                        smooth_xy = None
                        tip_start = hover_start = None
                        clicked = False
                        gestures.reset(keep_zoom=True)
                        native_zoom.reset(gestures.zoom)
                        print('Mouse/gestures:', 'ON' if control_enabled else 'OFF')
                elif key == ord('m'):
                    control_enabled = False
                    manual_mode, manual_locked = True, False
                    manual_points.clear()
                    gestures.reset(keep_zoom=True)
                elif key == ord('a'):
                    control_enabled = False
                    manual_mode = manual_locked = False
                    corners = stable_candidate = None
                    stable_count = 0
                    last_scan = -1e10
                    gestures.reset()
                elif key == ord('g'):
                    new_zoom_mode = native_zoom.toggle_mode(gestures.zoom)
                    print('Actual zoom destination:', new_zoom_mode.upper())
                    print('DESKTOP mode: press Windows+ESC to close Magnifier')
                elif key == ord('d'):
                    show_edges = not show_edges
                    if not show_edges:
                        try: cv2.destroyWindow('Detected Edges (low-res)')
                        except cv2.error: pass
                elif key == ord('s'):
                    cv2.imwrite('office_live_ar_hq_smooth.png', view)
                    print('Saved office_live_ar_hq_smooth.png')
    finally:
        office_worker.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
