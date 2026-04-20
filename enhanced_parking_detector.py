import heapq
import logging
import math
import os
import pickle
import sqlite3
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import cvzone
import numpy as np
import pandas as pd
import requests

import config
from car_detector import CarDetector
from mjpeg_server import MJPEGServer
from setup_directories import setup_directories

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class EnhancedParkingDetector:
    def __init__(
        self,
        video_path: Optional[str] = None,
        image_path: Optional[str] = None,
        camera_url: Optional[str] = None,
    ) -> None:
        """Initialise the detector.

        Parameters
        ----------
        video_path  : path to a local video file
        image_path  : path to a static parking-lot image (used for space selection)
        camera_url  : URL of the live camera stream (RTSP / HTTP / MJPEG).
                      When provided, ``process_camera()`` will stream from this URL.
        """
        # Setup directories first
        setup_directories()

        self.video_path = video_path
        self.image_path = image_path
        self.camera_url = camera_url
        self.width, self.height = config.PARKING_WIDTH, config.PARKING_HEIGHT
        self.posList = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.temp_rectangles = []
        self.car_detector = CarDetector()
        self.load_parking_positions()
        self.show_help = True
        self.history = []  # undo history
        self.is_reset = False
        self.original_image = None
        self.current_image = None
        self.boundary_mode = False
        self.lot_boundary = None

        # Temporal smoothing — per-slot deque of recent occupied booleans
        self.slot_history: Dict[int, deque] = {}

        # Add zoom functionality
        self.zoom_scale = 1.0
        self.zoom_step = 0.1
        self.min_zoom = 0.5
        self.max_zoom = 3.0
        self.pan_x = 0
        self.pan_y = 0
        self.pan_step = 50
        self.is_panning = False
        self.last_mouse_pos = None

        # ── Parking session / reservation state ───────────────────────────
        # slot index the system is currently guiding a car towards
        self._recommended_slot: Optional[int] = None
        # tracking when the recommended slot first became occupied
        self._recommended_occupied_start: Optional[float] = None
        # confirmation banner display until time (using physical time)
        self._parked_confirm_until: Optional[float] = None
        self._tracking_car_pos: Optional[Tuple[int, int]] = None
        self._tracking_loss_frames: int = 0
        self._tracking_last_step: float = 0.0
        self._tracking_expected_area: Optional[float] = None
        self._db_saved_banner_until: Optional[float] = None
        self._active_session_by_slot: Dict[int, str] = {}
        self._car_seen_in_recommended_until: Optional[float] = None
        self._session_active_slot_idx: Optional[int] = None
        # ── Gate-confirmed guidance flag ──────────────────────────────────
        # Guidance path + session claim are BLOCKED until the frontend
        # creates a session (POST /start).  Toggled True by _check_pending_session().
        self._guidance_enabled: bool = False
        self._pending_session_checked: bool = False   # True after first successful poll
        self._last_pending_poll: float = 0.0          # rate-limit backend polls
        # Redirect cooldown: don't fire redirect more than once per N seconds
        self._last_redirect_time: float = 0.0
        self._redirect_cooldown: float = 3.0  # seconds between redirects
        self._slots_waiting_exit: set = set()
        self._exiting_cars_tracking_pos: Dict[int, Tuple[int, int]] = {}
        self._exiting_cars_loss_frames: Dict[int, int] = {}
        self._exiting_cars_expected_area: Dict[int, float] = {}
        self._reserved_empty_since: Dict[int, float] = {}
        self._supabase_enabled = bool(config.SUPABASE_URL and config.SUPABASE_KEY)
        self._supabase_headers = {
            "apikey": config.SUPABASE_KEY,
            "Authorization": f"Bearer {config.SUPABASE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

        # Initialise the SQLite reservation database
        self._init_db()
        # Clear out old reservations on startup so the demo always starts fresh
        self._clear_all_reservations()

    def load_parking_positions(self) -> None:
        """Load previously saved parking positions from file"""
        try:
            with open(config.POSITION_FILE, "rb") as f:
                self.posList = pickle.load(f)
            logger.info(f"Loaded {len(self.posList)} parking positions")
        except FileNotFoundError:
            logger.warning(
                f"No saved positions found at '{config.POSITION_FILE}'. "
                "Run with --setup to define parking spaces from your camera."
            )
            self.posList = []
        except Exception as e:
            logger.error(
                f"Corrupt or invalid positions file ('{config.POSITION_FILE}'): {e}. "
                "Deleting it and starting fresh. Run with --setup to redefine slots."
            )
            # Remove the bad file so it doesn't cause the same error next run
            try:
                import os as _os
                _os.remove(config.POSITION_FILE)
            except OSError:
                pass
            self.posList = []

        try:
            with open(config.BOUNDARY_FILE, "rb") as f:
                self.lot_boundary = pickle.load(f)
            logger.info("Loaded parking lot boundary")
        except FileNotFoundError:
            self.lot_boundary = None
        except Exception as e:
            logger.error(f"Failed to load boundary: {e}")
            self.lot_boundary = None

    def save_parking_positions(self) -> None:
        """Persist current parking positions to disk (pure file write, no history side-effects)."""
        try:
            with open(config.POSITION_FILE, "wb") as f:
                pickle.dump(self.posList, f)
            logger.info(f"Saved {len(self.posList)} parking positions")
        except Exception as e:
            logger.error(f"Error saving positions: {e}")

        try:
            with open(config.BOUNDARY_FILE, "wb") as f:
                pickle.dump(self.lot_boundary, f)
            logger.info("Saved parking lot boundary")
        except Exception as e:
            logger.error(f"Error saving boundary: {e}")

        # Optional cloud sync for setup/layout data.
        self._supabase_sync_layout()

    def add_help_text(self, img: np.ndarray) -> np.ndarray:
        """Add help text overlay to the image showing keyboard shortcuts"""
        # Get image dimensions
        height, width = img.shape[:2]

        # Add numbered help text in the right corner
        help_text = [
            "=== Available Shortcuts ===",
            "1. R: Reset all selections",
            "2. Z: Undo last selection",
            "3. D: Detect & generate report",
            "4. S: Save current layout",
            "",
            "=== Mouse Controls ===",
            "• Left-click + Drag: Draw ONE slot",
            "• Right-click: Remove slot",
            "• B: Toggle boundary mode (Blue Box)",
        ]

        # Calculate starting position (right corner)
        start_x = width - 300  # 300 pixels from right edge
        y_offset = 30

        # Add text with background for better visibility
        for text in help_text:
            if text.startswith("==="):
                # Make section headers more prominent
                cvzone.putTextRect(
                    img,
                    text,
                    (start_x, y_offset),
                    scale=1.5,
                    thickness=2,
                    offset=5,
                    colorR=(0, 0, 0),
                )
            else:
                cvzone.putTextRect(
                    img,
                    text,
                    (start_x, y_offset),
                    scale=1.2,
                    thickness=2,
                    offset=5,
                    colorR=(0, 0, 0),
                )
            y_offset += 25  # Spacing between lines

    def apply_zoom_and_pan(self, img: np.ndarray) -> np.ndarray:
        # Get image dimensions
        h, w = img.shape[:2]

        # Calculate zoomed dimensions
        zoomed_w = int(w * self.zoom_scale)
        zoomed_h = int(h * self.zoom_scale)

        # Resize image
        zoomed_img = cv2.resize(img, (zoomed_w, zoomed_h))

        # Calculate pan boundaries
        max_pan_x = max(0, zoomed_w - w)
        max_pan_y = max(0, zoomed_h - h)

        # Clamp pan values
        self.pan_x = max(0, min(self.pan_x, max_pan_x))
        self.pan_y = max(0, min(self.pan_y, max_pan_y))

        # Crop image based on pan
        if zoomed_w > w or zoomed_h > h:
            x1 = self.pan_x
            y1 = self.pan_y
            x2 = min(x1 + w, zoomed_w)
            y2 = min(y1 + h, zoomed_h)
            zoomed_img = zoomed_img[y1:y2, x1:x2]

            # If the cropped image is smaller than the window, pad it
            if zoomed_img.shape[0] < h or zoomed_img.shape[1] < w:
                padded_img = np.zeros((h, w, 3), dtype=np.uint8)
                padded_img[: zoomed_img.shape[0], : zoomed_img.shape[1]] = zoomed_img
                zoomed_img = padded_img

        return zoomed_img

    def mouse_callback(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        if self.is_panning:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.last_mouse_pos is not None:
                dx = x - self.last_mouse_pos[0]
                dy = y - self.last_mouse_pos[1]
                self.pan_x = max(0, self.pan_x - dx)
                self.pan_y = max(0, self.pan_y - dy)
                self.last_mouse_pos = (x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.last_mouse_pos = None
            return

        # Adjust coordinates for border
        border_size = 50
        x -= border_size
        y -= border_size

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                x1, y1 = self.start_point
                x2, y2 = self.end_point

                # Normalise so top-left is always (x1, y1)
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                slot_w = x2 - x1
                slot_h = y2 - y1

                # Only add if the drawn area is large enough to be a real slot
                if slot_w >= 10 and slot_h >= 10:
                    if self.boundary_mode:
                        self.lot_boundary = (x1, y1, slot_w, slot_h)
                        self.save_parking_positions()
                        self.boundary_mode = False # Auto-exit boundary mode after drawing
                    else:
                        self.history.append(self.posList.copy())
                        self.posList.append((x1, y1, slot_w, slot_h))
                        self.save_parking_positions()

                self.start_point = None
                self.end_point = None
        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.boundary_mode and self.lot_boundary:
                bx, by, bw, bh = self.lot_boundary
                if bx < x < bx + bw and by < y < by + bh:
                    self.lot_boundary = None
                    self.save_parking_positions()
            else:
                # Remove the first slot whose bounding box contains the click point
                for pos in self.posList[:]:
                    px, py, pw, ph = pos
                    if px < x < px + pw and py < y < py + ph:
                        self.history.append(self.posList.copy())
                        self.posList.remove(pos)
                        self.save_parking_positions()
                        break

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame using computer vision techniques for parking space detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (3, 3), 1)
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 16
        )
        # Apply median blur to remove noise
        median = cv2.medianBlur(thresh, 5)
        # Dilate to connect components
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(median, kernel, iterations=1)
        return dilated

    def check_parking_space(
        self, img_pro: np.ndarray, img: np.ndarray
    ) -> Tuple[int, int, List[int]]:
        """Check each parking space using temporally-smoothed pixel counting.

        Returns
        -------
        space_counter  : number of free (non-gate) slots
        occupied_slots : number of occupied (non-gate) slots
        empty_indices  : 0-indexed list of currently-free slot positions
        """
        space_counter = 0
        occupied_slots = 0
        empty_indices: List[int] = []
        gate_idx = config.GATE_SLOT_INDEX

        for idx, pos in enumerate(self.posList):
            x, y, w, h = pos
            if w <= 0 or h <= 0:
                continue
            img_crop = img_pro[y: y + h, x: x + w]
            if img_crop.size == 0:
                continue

            count = cv2.countNonZero(img_crop)

            # Scale threshold proportionally to slot area
            slot_area = w * h
            ref_area = config.PARKING_WIDTH * config.PARKING_HEIGHT
            scaled_threshold = max(
                1,
                int(config.OCCUPANCY_THRESHOLD * slot_area / ref_area)
            )

            raw_occupied = count >= scaled_threshold

            # ── Temporal smoothing ────────────────────────────────────────────
            # Only flip a slot's displayed status when it has been consistently
            # the same state for TEMPORAL_SMOOTH_FRAMES frames in a row.
            if idx not in self.slot_history:
                self.slot_history[idx] = deque(maxlen=config.TEMPORAL_SMOOTH_FRAMES)
            self.slot_history[idx].append(1 if raw_occupied else 0)
            history = self.slot_history[idx]
            occupied = (sum(history) / len(history)) >= 0.5
            # ─────────────────────────────────────────────────────────────────

            # Gate slot: always draw with orange, skip from free/occ counts
            if idx == gate_idx:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 4)
                cvzone.putTextRect(
                    img, "GATE",
                    (x, y - 8),
                    scale=1.2, thickness=2, offset=4,
                    colorR=(0, 140, 255),
                )
                continue

            if not occupied:
                color = (0, 255, 0)    # green — empty
                thickness = 5
                space_counter += 1
                empty_indices.append(idx)
            else:
                color = (0, 0, 255)    # red — occupied
                thickness = 2
                occupied_slots += 1

            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            cvzone.putTextRect(
                img,
                str(count),
                (x, y + h - 3),
                scale=1,
                thickness=2,
                offset=0,
                colorR=color,
            )

        # Display overall statistics (excluding gate slot)
        non_gate_total = len(self.posList) - (1 if 0 <= gate_idx < len(self.posList) else 0)
        cvzone.putTextRect(
            img,
            f"Free: {space_counter}/{non_gate_total}",
            (100, 50),
            scale=3,
            thickness=5,
            offset=20,
            colorR=(0, 200, 0),
        )

        return space_counter, occupied_slots, empty_indices

    def check_parking_space_diff(
        self, img: np.ndarray, bg_gray: Optional[np.ndarray]
    ) -> Tuple[int, int, List[int], bool, bool, Optional[np.ndarray]]:
        """Check each parking space using background subtraction and smooth it temporally."""
        space_counter = 0
        occupied_slots = 0
        empty_indices: List[int] = []
        gate_idx = config.GATE_SLOT_INDEX
        exit_gate_idx = config.EXIT_GATE_SLOT_INDEX
        gate_occupied = False
        exit_gate_occupied = False

        if bg_gray is not None:
            curr_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
            
            diff = cv2.absdiff(bg_gray, curr_gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=1)
        else:
            thresh = None

        for idx, pos in enumerate(self.posList):
            x, y, w, h = pos
            if w <= 0 or h <= 0:
                continue

            count = 0
            if thresh is not None:
                img_crop = thresh[y: y + h, x: x + w]
                # Filter noise by ignoring very edge pixels if necessary
                count = cv2.countNonZero(img_crop)

            # Scale threshold proportionally to slot area
            slot_area = w * h
            ref_area = config.PARKING_WIDTH * config.PARKING_HEIGHT
            scaled_threshold = max(
                1,
                int(config.OCCUPANCY_THRESHOLD * slot_area / ref_area)
            )

            raw_occupied = count >= scaled_threshold

            # ── Temporal smoothing ────────────────────────────────────────────
            if idx not in self.slot_history:
                self.slot_history[idx] = deque(maxlen=config.TEMPORAL_SMOOTH_FRAMES)
            self.slot_history[idx].append(1 if raw_occupied else 0)
            history = self.slot_history[idx]
            occupied = (sum(history) / len(history)) >= 0.5
            # ─────────────────────────────────────────────────────────────────

            # Gate slot: always draw with orange, skip from free/occ counts
            if idx == gate_idx:
                gate_occupied = occupied
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), 4)
                cvzone.putTextRect(
                    img, "GATE",
                    (x, y - 8),
                    scale=1.2, thickness=2, offset=4,
                    colorR=(0, 140, 255),
                )
                continue

            # Exit gate slot: draw purple, skip from free/occ counts
            if idx == exit_gate_idx:
                exit_gate_occupied = occupied
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 4)
                cvzone.putTextRect(
                    img, "EXIT GATE",
                    (x, y - 8),
                    scale=1.2, thickness=2, offset=4,
                    colorR=(255, 0, 255),
                )
                continue

            if not occupied:
                color = (0, 255, 0)    # green — empty
                thickness = 5
                space_counter += 1
                empty_indices.append(idx)
            else:
                color = (0, 0, 255)    # red — occupied
                thickness = 2
                occupied_slots += 1

            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            cvzone.putTextRect(
                img,
                str(count),
                (x, y + h - 3),
                scale=1,
                thickness=2,
                offset=0,
                colorR=color,
            )

        # Display overall statistics (excluding gate slot)
        non_gate_total = len(self.posList)
        if 0 <= gate_idx < len(self.posList): non_gate_total -= 1
        if 0 <= exit_gate_idx < len(self.posList): non_gate_total -= 1
        
        cvzone.putTextRect(
            img,
            f"Free: {space_counter}/{non_gate_total}",
            (100, 50),
            scale=3,
            thickness=5,
            offset=20,
            colorR=(0, 200, 0),
        )

        return space_counter, occupied_slots, empty_indices, gate_occupied, exit_gate_occupied, thresh

    # ── Parking Reservation Database ─────────────────────────────────────────

    def _init_db(self) -> None:
        """Create the SQLite DB and reservations table if they don't exist."""
        try:
            os.makedirs(os.path.dirname(config.PARKING_DB_FILE), exist_ok=True)
            conn = sqlite3.connect(config.PARKING_DB_FILE)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    slot_index  INTEGER PRIMARY KEY,
                    reserved_at TEXT NOT NULL
                )
                """
            )
            conn.commit()
            conn.close()
            logger.info(f"Parking DB ready at {config.PARKING_DB_FILE}")
        except Exception as e:
            logger.error(f"Failed to initialise parking DB: {e}")

    def _clear_all_reservations(self) -> None:
        """Clear all active reservations unconditionally (e.g. at startup)."""
        try:
            conn = sqlite3.connect(config.PARKING_DB_FILE)
            conn.execute("DELETE FROM reservations")
            conn.commit()
            conn.close()
            logger.info("Cleared all old SQLite reservations from previous runs.")
        except Exception as e:
            logger.error(f"Failed to clear old reservations: {e}")

    def _get_reserved_slots(self) -> List[int]:
        """Return a list of slot indices currently reserved in the DB."""
        try:
            conn = sqlite3.connect(config.PARKING_DB_FILE)
            rows = conn.execute("SELECT slot_index FROM reservations").fetchall()
            conn.close()
            return [r[0] for r in rows]
        except Exception as e:
            logger.error(f"DB read error: {e}")
            return []

    def _reserve_slot(self, slot_idx: int) -> None:
        """Mark a slot as reserved (the car is being directed there)."""
        try:
            conn = sqlite3.connect(config.PARKING_DB_FILE)
            conn.execute(
                "INSERT OR REPLACE INTO reservations (slot_index, reserved_at) VALUES (?, ?)",
                (slot_idx, datetime.now().isoformat()),
            )
            conn.commit()
            conn.close()
            logger.info(f"Slot {slot_idx + 1} reserved in DB")
        except Exception as e:
            logger.error(f"DB reserve error: {e}")
        self._reserved_empty_since.pop(slot_idx, None)
        self._supabase_update_slot_status(slot_idx, "occupied")

    def _free_slot(self, slot_idx: int) -> None:
        """Remove a slot from the reservation table (car has left)."""
        try:
            conn = sqlite3.connect(config.PARKING_DB_FILE)
            conn.execute("DELETE FROM reservations WHERE slot_index = ?", (slot_idx,))
            conn.commit()
            conn.close()
            logger.info(f"Slot {slot_idx + 1} freed in DB")
        except Exception as e:
            logger.error(f"DB free error: {e}")
        self._slots_waiting_exit.add(slot_idx)
        self._supabase_update_slot_status(slot_idx, "available")

    # ── Supabase sync (optional) ──────────────────────────────────────────────

    def _supabase_iso_now(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def _supabase_table_url(self, table: str) -> str:
        return f"{config.SUPABASE_URL.rstrip('/')}/rest/v1/{table}"

    def _supabase_get_slot_row(self, slot_idx: int) -> Optional[Dict[str, Any]]:
        if not self._supabase_enabled:
            return None
        if not config.SUPABASE_LOT_ID:
            logger.warning("SUPABASE_LOT_ID not set; skipping Supabase slot lookup.")
            return None
        slot_number = slot_idx + 1
        try:
            resp = requests.get(
                self._supabase_table_url(config.SUPABASE_SLOTS_TABLE),
                headers=self._supabase_headers,
                params={
                    "select": "id,slot_number,status",
                    "lot_id": f"eq.{config.SUPABASE_LOT_ID}",
                    "slot_number": f"eq.{slot_number}",
                    "limit": 1,
                },
                timeout=5,
            )
            resp.raise_for_status()
            rows = resp.json()
            return rows[0] if rows else None
        except Exception as e:
            logger.error(f"Supabase slot fetch failed for slot {slot_number}: {e}")
            return None

    def _supabase_update_slot_status(self, slot_idx: int, status: str) -> None:
        if not self._supabase_enabled:
            return
        if not config.SUPABASE_LOT_ID:
            return
        slot_number = slot_idx + 1
        status_bool = status == "occupied"
        roi_payload = None
        map_payload = None
        if 0 <= slot_idx < len(self.posList):
            x, y, w, h = self.posList[slot_idx]
            roi_payload = {"x": x, "y": y, "w": w, "h": h}
            boundary_payload = None
            if self.lot_boundary:
                bx, by, bw, bh = self.lot_boundary
                boundary_payload = {"x": bx, "y": by, "w": bw, "h": bh}
            map_payload = {
                "center": {"x": x + (w // 2), "y": y + (h // 2)},
                "lot_boundary": boundary_payload,
            }
        payload = {
            "status": status_bool,
            "last_updated": self._supabase_iso_now(),
            "roi_coordinates": roi_payload,
            "map_coordinates": map_payload,
        }

        def _run():
            try:
                resp = requests.patch(
                    self._supabase_table_url(config.SUPABASE_SLOTS_TABLE),
                    headers=self._supabase_headers,
                    params={
                        "lot_id": f"eq.{config.SUPABASE_LOT_ID}",
                        "slot_number": f"eq.{slot_number}",
                    },
                    json=payload,
                    timeout=5,
                )
                resp.raise_for_status()
                # PATCH may legally return [] when no row matches filters.
                # In that case, insert a fallback row so status is never dropped.
                rows = []
                try:
                    rows = resp.json()
                except Exception:
                    rows = []
                if isinstance(rows, list) and len(rows) == 0:
                    insert_payload = {
                        "lot_id": config.SUPABASE_LOT_ID,
                        "slot_number": slot_number,
                        "status": status_bool,
                        "last_updated": self._supabase_iso_now(),
                        "roi_coordinates": roi_payload,
                        "map_coordinates": map_payload,
                    }
                    insert_resp = requests.post(
                        self._supabase_table_url(config.SUPABASE_SLOTS_TABLE),
                        headers=self._supabase_headers,
                        json=insert_payload,
                        timeout=5,
                    )
                    insert_resp.raise_for_status()
                    logger.info(
                        f"Supabase slot {slot_number} missing on update; inserted fallback row."
                    )
                logger.info(f"Supabase slot {slot_number} set to '{status}'.")
            except Exception as e:
                error_body = ""
                try:
                    error_body = f" response={resp.text}"
                except Exception:
                    pass
                logger.error(f"Supabase slot update failed for slot {slot_number}: {e}{error_body}")
                
        import threading
        threading.Thread(target=_run, daemon=True).start()

    def _supabase_sync_layout(self) -> None:
        """Sync local slot rectangles + lot boundary into Supabase parking_slots."""
        if not self._supabase_enabled:
            return
        if not config.SUPABASE_LOT_ID:
            return
        if not self.posList:
            return

        boundary_payload = None
        if self.lot_boundary:
            bx, by, bw, bh = self.lot_boundary
            boundary_payload = {"x": bx, "y": by, "w": bw, "h": bh}

        now_iso = self._supabase_iso_now()
        for idx, (x, y, w, h) in enumerate(self.posList):
            slot_number = idx + 1
            slot_center = {"x": x + (w // 2), "y": y + (h // 2)}
            patch_payload = {
                "roi_coordinates": {"x": x, "y": y, "w": w, "h": h},
                "map_coordinates": {"center": slot_center, "lot_boundary": boundary_payload},
                "last_updated": now_iso,
            }
            try:
                # Try update existing row first.
                patch_resp = requests.patch(
                    self._supabase_table_url(config.SUPABASE_SLOTS_TABLE),
                    headers=self._supabase_headers,
                    params={
                        "lot_id": f"eq.{config.SUPABASE_LOT_ID}",
                        "slot_number": f"eq.{slot_number}",
                    },
                    json=patch_payload,
                    timeout=5,
                )
                patch_resp.raise_for_status()

                # If no row matched, insert a new one.
                if patch_resp.text.strip() == "[]":
                    insert_payload = {
                        "lot_id": config.SUPABASE_LOT_ID,
                        "slot_number": slot_number,
                        "status": False,
                        **patch_payload,
                    }
                    post_resp = requests.post(
                        self._supabase_table_url(config.SUPABASE_SLOTS_TABLE),
                        headers=self._supabase_headers,
                        json=insert_payload,
                        timeout=5,
                    )
                    post_resp.raise_for_status()

            except Exception as e:
                error_body = ""
                try:
                    if "post_resp" in locals():
                        error_body = f" response={post_resp.text}"
                    else:
                        error_body = f" response={patch_resp.text}"
                except Exception:
                    pass
                logger.error(
                    f"Supabase layout sync failed for slot {slot_number}: {e}{error_body}"
                )

    def _check_pending_session(self) -> bool:
        """Poll the backend once to see if the frontend created a pending session.
        Returns True if a session with no slot_id exists (driver confirmed plate).
        Rate-limited to one poll per 1.5 seconds.
        """
        import time as _time
        now = _time.time()
        if now - self._last_pending_poll < 1.5:
            return self._guidance_enabled
        self._last_pending_poll = now
        try:
            resp = requests.get(
                f"{config.BACKEND_URL}/session/pending",
                timeout=2,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("has_pending"):
                    if not self._guidance_enabled:
                        logger.info("Frontend session detected — guidance path enabled.")
                    self._guidance_enabled = True
                else:
                    # Session was claimed or doesn't exist yet: keep current flag
                    pass
        except Exception:
            pass  # Backend unreachable — keep current state
        return self._guidance_enabled

    def _supabase_start_session(self, slot_idx: int) -> None:
        if not self._supabase_enabled:
            return
        if slot_idx in self._active_session_by_slot:
            return
        slot_row = self._supabase_get_slot_row(slot_idx)
        if not slot_row:
            return
        try:
            # Claim the session created by the frontend (driver typed their plate).
            # We do NOT create a new session here — the frontend already did that.
            payload = {
                "slot_id": str(slot_row["id"])
            }
            resp = requests.post(
                f"{config.BACKEND_URL}/session/claim",
                json=payload,
                timeout=5,
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Lưu lại Session ID do Backend trả về để dùng lúc xe ra
            self._active_session_by_slot[slot_idx] = data["session_id"]
            logger.info(f"Backend started session for slot {slot_idx + 1}. ID: {data['session_id']}")
        except Exception as e:
            logger.error(f"Backend auto-checkin failed for slot {slot_idx + 1}: {e}")

    def _supabase_complete_session(self, slot_idx: int) -> None:
        if not self._supabase_enabled:
            return
        # Pop it synchronously so it cannot be double-processed by main thread
        session_id = self._active_session_by_slot.pop(slot_idx, None)
        if not session_id:
            return
            
        def _run():
            try:
                # GỌI SANG BACKEND ĐỂ CHỐT ĐỖ XE / HOẶC HOÀN THÀNH
                payload = {
                    "session_id": session_id,
                    "selected_slot_id": None # Backend sẽ tự xử lý
                }
                resp = requests.post(
                    f"{config.BACKEND_URL}/finish",
                    json=payload,
                    timeout=5,
                )
                resp.raise_for_status()
                logger.info(f"Backend completed session for slot {slot_idx + 1}.")
            except Exception as e:
                logger.error(f"Backend complete session failed: {e}")
                
        import threading
        threading.Thread(target=_run, daemon=True).start()

    def _supabase_redirect_session(self, session_id: str, new_slot_idx: int) -> None:
        if not self._supabase_enabled or not session_id:
            return
        slot_row = self._supabase_get_slot_row(new_slot_idx)
        if not slot_row:
            return
            
        def _run():
            try:
                payload = {
                    "session_id": session_id,
                    "new_slot_id": str(slot_row["id"])
                }
                resp = requests.post(
                    f"{config.BACKEND_URL}/redirect",
                    json=payload,
                    timeout=5,
                )
                resp.raise_for_status()
                logger.info(f"Backend redirected session to slot {new_slot_idx + 1}")
            except Exception as e:
                logger.error(f"Backend redirect session failed: {e}")
                
        import threading
        threading.Thread(target=_run, daemon=True).start()

    # ── Path Guidance ─────────────────────────────────────────────────────────

    def find_closest_empty_slot(
        self, gate_idx: int, empty_indices: List[int]
    ) -> Optional[int]:
        """Return the index of the empty slot nearest to the gate.

        Slots already reserved in the DB or currently being tracked for an
        active recommendation are excluded so each car gets its own unique
        destination.  Returns None when there are no available slots.
        """
        # Exclude slots that are DB-reserved
        reserved = set(self._get_reserved_slots())

        available = [i for i in empty_indices if i not in reserved]
        if not available:
            return None

        gx, gy, gw, gh = self.posList[gate_idx]
        gate_cx, gate_cy = gx + gw // 2, gy + gh // 2
        return min(
            available,
            key=lambda i: math.dist(
                (self.posList[i][0] + self.posList[i][2] // 2,
                 self.posList[i][1] + self.posList[i][3] // 2),
                (gate_cx, gate_cy),
            ),
        )

    # ── A* obstacle-avoiding path ─────────────────────────────────────────────

    def _build_obstacle_grid(
        self, img_w: int, img_h: int, exclude_indices: List[int], cell: int
    ) -> np.ndarray:
        """Create a boolean grid where True = obstacle (another parking slot).

        Each cell covers ``cell`` pixels.  Slots whose index is in
        *exclude_indices* (i.e. the gate and target) are NOT marked as
        obstacles so the path can start/end inside them.
        """
        cols = math.ceil(img_w / cell)
        rows = math.ceil(img_h / cell)
        grid = np.zeros((rows, cols), dtype=bool)

        for idx, (sx, sy, sw, sh) in enumerate(self.posList):
            if idx in exclude_indices:
                continue
            # Mark all grid cells that overlap this slot as obstacles.
            # Shrink the obstacle area slightly (2px margin) so the path
            # can squeeze through narrow aisles between slots.
            margin = 2
            gx1 = max(0, (sx + margin) // cell)
            gy1 = max(0, (sy + margin) // cell)
            gx2 = min(cols - 1, (sx + sw - margin) // cell)
            gy2 = min(rows - 1, (sy + sh - margin) // cell)
            grid[gy1 : gy2 + 1, gx1 : gx2 + 1] = True

        # Clamp grid boundaries
        if self.lot_boundary:
            bx, by, bw, bh = self.lot_boundary
            for r in range(rows):
                for c in range(cols):
                    # Using the middle point of each cell for checking boundary
                    px_c = c * cell + cell // 2
                    py_r = r * cell + cell // 2
                    if not (bx <= px_c <= bx + bw and by <= py_r <= by + bh):
                        grid[r, c] = True

        return grid

    def _astar(
        self,
        grid: np.ndarray,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        turn_penalty: float = 2.5,
    ) -> List[Tuple[int, int]]:
        """A* on a 2-D boolean obstacle grid with a penalty for changing direction.

        Parameters
        ----------
        grid         : 2-D bool array, True = blocked
        start        : (col, row) in grid coordinates
        goal         : (col, row) in grid coordinates
        turn_penalty : added cost when path changes its directional heading

        Returns a list of (col, row) grid cells from start to goal
        (inclusive), or a direct [start, goal] fallback if no path found.
        """
        rows, cols = grid.shape
        sc, sr = start
        gc, gr = goal

        # Clamp to valid grid cells
        sc = max(0, min(sc, cols - 1))
        sr = max(0, min(sr, rows - 1))
        gc = max(0, min(gc, cols - 1))
        gr = max(0, min(gr, rows - 1))

        def h(c, r):
            return abs(c - gc) + abs(r - gr)

        open_heap: List = []
        # State: (f, g, c, r, prev_dc, prev_dr)
        heapq.heappush(open_heap, (h(sc, sr), 0, sc, sr, 0, 0))
        came_from: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {(sc, sr, 0, 0): None}
        g_score: Dict[Tuple[int, int, int, int], float] = {(sc, sr, 0, 0): 0.0}

        # 8-directional movement
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1),
        ]

        while open_heap:
            _, g, c, r, pdc, pdr = heapq.heappop(open_heap)

            if (c, r) == (gc, gr):
                # Reconstruct path
                path = []
                cur: Optional[Tuple[int, int, int, int]] = (gc, gr, pdc, pdr)
                while cur is not None:
                    path.append((cur[0], cur[1]))
                    cur = came_from[cur]
                path.reverse()
                return path

            for dc, dr in directions:
                nc, nr = c + dc, r + dr
                if not (0 <= nc < cols and 0 <= nr < rows):
                    continue
                if grid[nr, nc]:
                    continue  # blocked
                
                step_cost = math.sqrt(dc * dc + dr * dr)
                
                # Apply penalty if direction changes (and we were already moving)
                penalty = 0.0
                if (pdc, pdr) != (0, 0) and (pdc, pdr) != (dc, dr):
                    penalty = turn_penalty
                    
                ng = g + step_cost + penalty
                
                state = (nc, nr, dc, dr)
                if ng < g_score.get(state, float("inf")):
                    g_score[state] = ng
                    came_from[state] = (c, r, pdc, pdr)
                    heapq.heappush(open_heap, (ng + h(nc, nr), ng, nc, nr, dc, dr))

        # Fallback: straight line if A* cannot find a route
        return [start, goal]

    def _smooth_path(
        self, path: List[Tuple[int, int]], cell: int
    ) -> List[Tuple[int, int]]:
        """Convert grid-cell path to pixel coordinates and reduce intermediate
        waypoints using a simple line-of-sight shortcut pass."""
        # Convert to pixel centres
        px_path = [(c * cell + cell // 2, r * cell + cell // 2) for c, r in path]
        if len(px_path) <= 2:
            return px_path
        # Keep every Nth waypoint to smooth jagged grid steps
        step = max(1, len(px_path) // 20)
        smoothed = px_path[::step]
        if smoothed[-1] != px_path[-1]:
            smoothed.append(px_path[-1])
        return smoothed

    def draw_path_guidance(
        self, img: np.ndarray, gate_idx: int, closest_idx: Optional[int], is_full: bool = False, moving_pos: Optional[Tuple[int, int]] = None
    ) -> None:
        """Draw an obstacle-avoiding arrow path from the gate to the nearest free slot.

        Highlights the GATE slot with an orange border and the TARGET slot
        with a cyan border.  The path routes around other parking slots
        using A* on a coarse grid.

        If the lot is full, shows a 'PARKING FULL' banner instead.
        """
        h_img, w_img = img.shape[:2]
        gx, gy, gw, gh = self.posList[gate_idx]
        gate_cx, gate_cy = gx + gw // 2, gy + gh // 2

        start_px_x = moving_pos[0] if moving_pos is not None else gate_cx
        start_px_y = moving_pos[1] if moving_pos is not None else gate_cy

        # ── Always highlight the gate slot ────────────────────────────────────
        cv2.rectangle(img, (gx - 3, gy - 3), (gx + gw + 3, gy + gh + 3), (0, 140, 255), 4)
        
        # Track dot at actual tracking centroid
        if moving_pos is not None:
            cv2.circle(img, (start_px_x, start_px_y), 8, (0, 255, 0), -1)
            cv2.circle(img, (start_px_x, start_px_y), 14, (0, 150, 0), 2)
            
        cvzone.putTextRect(
            img, "GATE",
            (gx, gy - 8),
            scale=1.3, thickness=2, offset=5,
            colorR=(0, 120, 255),
        )

        if closest_idx is None:
            if is_full:
                # ── Parking full ──────────────────────────────────────────────────
                overlay = img.copy()
                bx1, by1 = w_img // 2 - 220, h_img // 2 - 50
                bx2, by2 = w_img // 2 + 220, h_img // 2 + 50
                cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 160), -1)
                cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
                cvzone.putTextRect(
                    img, "PARKING FULL",
                    (bx1 + 20, by2 - 14),
                    scale=2.5, thickness=3, offset=8,
                    colorR=(0, 0, 160),
                )
            return

        # ── Target slot ───────────────────────────────────────────────────────
        sx, sy, sw, sh = self.posList[closest_idx]
        slot_cx, slot_cy = sx + sw // 2, sy + sh // 2

        # Thick cyan border on the target slot
        cv2.rectangle(img, (sx - 4, sy - 4), (sx + sw + 4, sy + sh + 4), (0, 255, 255), 4)
        cvzone.putTextRect(
            img,
            f"SLOT {closest_idx + 1}",
            (sx, sy - 10),
            scale=1.5, thickness=2, offset=5,
            colorR=(0, 200, 200),
        )

        # ── Obstacle-avoiding path (A* on coarse grid) ────────────────────────
        # Cell size = roughly 1/4 of the average slot dimension
        avg_slot_dim = max(
            10,
            (sum(p[2] for p in self.posList) + sum(p[3] for p in self.posList))
            // (2 * len(self.posList)),
        )
        cell = max(6, avg_slot_dim // 4)

        # Gate + target are passable (not obstacles)
        passable = {gate_idx, closest_idx}
        grid = self._build_obstacle_grid(w_img, h_img, list(passable), cell)

        # Grid coordinates for dynamic start centre and slot centre
        start_cell = (start_px_x // cell, start_px_y // cell)
        s_cell = (slot_cx // cell, slot_cy // cell)

        path_cells = self._astar(grid, start_cell, s_cell)
        waypoints = self._smooth_path(path_cells, cell)
        
        # Anchor visual line exactly to the car coordinates and slot coordinates
        if waypoints:
            waypoints[0] = (start_px_x, start_px_y)
            waypoints[-1] = (slot_cx, slot_cy)

        # Draw dashed-line segments along the waypoints
        for i in range(len(waypoints) - 1):
            p0 = waypoints[i]
            p1 = waypoints[i + 1]
            seg_dx = p1[0] - p0[0]
            seg_dy = p1[1] - p0[1]
            seg_len = math.dist(p0, p1)
            n_seg = max(1, int(seg_len // 18))
            for seg in range(n_seg):
                t0 = seg / n_seg
                t1 = min(1.0, (seg + 0.55) / n_seg)
                lp0 = (int(p0[0] + seg_dx * t0), int(p0[1] + seg_dy * t0))
                lp1 = (int(p0[0] + seg_dx * t1), int(p0[1] + seg_dy * t1))
                cv2.line(img, lp0, lp1, (0, 255, 255), 3)

        # Arrowhead at destination
        if len(waypoints) >= 2:
            p_prev = waypoints[-2]
            p_last = waypoints[-1]
            cv2.arrowedLine(
                img,
                p_prev,
                (slot_cx, slot_cy),
                (0, 255, 255),
                3,
                tipLength=0.4,
            )

        # ── Bottom-left guidance banner ───────────────────────────────────────
        dist_px = math.dist((gate_cx, gate_cy), (slot_cx, slot_cy))
        dist_m = round(dist_px / config.PIXELS_PER_METER, 1) if config.PIXELS_PER_METER > 0 else None
        dist_str = f"  (~{dist_m} m)" if dist_m else ""
        cvzone.putTextRect(
            img,
            f"\u2794 Drive to Slot {closest_idx + 1}{dist_str}",
            (10, h_img - 20),
            scale=1.8, thickness=2, offset=8,
            colorR=(0, 140, 0),
        )

    # ─────────────────────────────────────────────────────────────────────────



    def generate_csv_report(
        self, total_slots: int, occupied_slots: int, available_slots: int
    ) -> None:
        """Generate CSV report with parking statistics"""
        try:
            data = {
                "Total Slots": [total_slots],
                "Occupied Slots": [occupied_slots],
                "Available Slots": [available_slots],
                "Timestamp": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            }
            df = pd.DataFrame(data)

            # Check if we should append or overwrite
            csv_path = os.path.join(config.DATA_DIR, config.CSV_FILE)
            if config.CSV_APPEND_MODE and os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False, index=False)
                logger.info(f"Appended data to {csv_path}")
            else:
                df.to_csv(csv_path, index=False)
                logger.info(f"Created new CSV report at {csv_path}")
        except Exception as e:
            logger.error(f"Error generating CSV report: {e}")

    def process_video(self) -> None:
        """Process video file for parking detection"""
        if not self.video_path:
            raise ValueError("Video path not provided")

        # Validate video file exists
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")

        logger.info(f"Processing video: {self.video_path}")
        cap = cv2.VideoCapture(self.video_path)

        if not cap.isOpened():
            raise IOError(f"Cannot open video file: {self.video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a window that can be resized
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, frame_width, frame_height)

        try:
            while True:
                if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                success, img = cap.read()
                if not success:
                    break

                # Currently video streaming with background subtraction is limited. 
                # Use none as background or set it up if needed.
                available_slots, occupied_slots, _empty, gate_occupied, exit_gate_occupied, thresh = self.check_parking_space_diff(img, None)

                # Generate CSV report every 30 frames
                if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % 30 == 0:
                    self.generate_csv_report(
                        total_slots=len(self.posList),
                        occupied_slots=occupied_slots,
                        available_slots=available_slots,
                    )

                cv2.imshow(config.WINDOW_NAME, img)
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Video processing completed")

    def process_camera(self, camera_url: Optional[str] = None) -> None:
        """Stream from a live camera URL and perform real-time parking detection.

        The method accepts any URL that OpenCV's VideoCapture understands:
          - RTSP  : ``rtsp://user:pass@<ip>:<port>/stream``
          - HTTP  : ``http://<ip>:<port>/video``
          - MJPEG : ``http://<ip>/mjpg/video.mjpg``

        It will automatically attempt to reconnect if the stream drops.
        Press **Q** to quit.
        """
        url = camera_url or self.camera_url or config.CAMERA_URL
        if not url:
            raise ValueError(
                "No camera URL provided. "
                "Supply it via --camera <url>, the camera_url constructor argument, "
                "or set CAMERA_URL in config.py."
            )

        logger.info(f"Connecting to camera stream: {url}")
        print(f"\n📷 Connecting to camera: {url}")
        print("Press 'Q' to quit\n")

        frame_fail_count = 0
        reconnect_delay = config.CAMERA_RECONNECT_DELAY
        frame_timeout = config.CAMERA_FRAME_TIMEOUT

        # Try to open the stream once before entering the display window so we
        # can provide a meaningful error message if the URL is wrong.
        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise IOError(
                f"Cannot connect to camera stream: {url}\n"
                "Check that the URL is reachable and the camera is online."
            )

        # ── Start MJPEG server ─────────────────────────────────────────
        # Serves annotated frames to the backend proxy so the browser sees
        # the fully-overlaid feed (slots, guiding path, green dot).
        _mjpeg = MJPEGServer(port=config.MJPEG_STREAM_PORT)
        _mjpeg.start()

        bg_gray = None
        if os.path.exists("camera_reference.jpg"):
            bg_img = cv2.imread("camera_reference.jpg")
            if bg_img is not None:
                bg_gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
                bg_gray = cv2.GaussianBlur(bg_gray, (5, 5), 0)
                logger.info("Loaded background reference for subtraction.")
            else:
                logger.warning("Could not read camera_reference.jpg")
        else:
            logger.warning("No camera_reference.jpg found for background subtraction!")

        # Grab an initial frame to size the window
        success, first_frame = cap.read()
        if success:
            frame_h, frame_w = first_frame.shape[:2]
        else:
            frame_w, frame_h = 1280, 720  # sensible fallback

        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, frame_w, frame_h)

        frame_count = 0

        # Warn and guide user when no layout has been defined yet
        if not self.posList:
            logger.warning(
                "No parking slots defined! "
                "Re-run with --setup to draw slots on a camera frame first."
            )
            print(
                "\n⚠️  WARNING: No parking slots are defined yet.\n"
                "   The camera feed will open but NO slots or detection will be shown.\n"
                "   ➡  Press Q, then re-run with the --setup flag to define your layout:\n"
                f"      python run.py --camera {url} --setup\n"
            )

        try:
            while True:
                success, img = cap.read()

                if not success:
                    frame_fail_count += 1
                    logger.warning(
                        f"Failed to read frame ({frame_fail_count}/{frame_timeout}). "
                        "Attempting reconnect..."
                    )

                    if frame_fail_count >= frame_timeout:
                        # Release and reconnect
                        cap.release()
                        logger.info(f"Reconnecting in {reconnect_delay}s...")
                        print(f"⚠️  Stream lost. Reconnecting in {reconnect_delay}s...")
                        time.sleep(reconnect_delay)

                        cap = cv2.VideoCapture(url)
                        if cap.isOpened():
                            logger.info("Reconnected to camera stream.")
                            print("✓ Reconnected to camera stream.")
                            frame_fail_count = 0
                        else:
                            logger.error("Reconnect failed. Retrying...")
                    continue  # skip the rest of the loop for this iteration

                # Successful read — reset failure counter
                frame_fail_count = 0
                frame_count += 1

                # Background Subtraction Detection
                available_slots, occupied_slots, empty_indices, gate_occupied, exit_gate_occupied, thresh = self.check_parking_space_diff(img, bg_gray)

                # ── Path guidance + reservation logic ─────────────────────────
                gate_idx = config.GATE_SLOT_INDEX
                exit_gate_idx = config.EXIT_GATE_SLOT_INDEX
                if self.posList and 0 <= gate_idx < len(self.posList):
                    empty_excl_gate = [i for i in empty_indices if i != gate_idx and i != exit_gate_idx]

                    # ── Release DB slots whose sensor now reads empty ──────────
                    for reserved_idx in self._get_reserved_slots():
                        if reserved_idx in empty_excl_gate:
                            # Slot became empty; require stable-empty duration to avoid
                            # flicker causing immediate false release.
                            now_ts = time.time()
                            if reserved_idx not in self._reserved_empty_since:
                                self._reserved_empty_since[reserved_idx] = now_ts
                            elif now_ts - self._reserved_empty_since[reserved_idx] >= 2.0:
                                self._free_slot(reserved_idx)
                                self._reserved_empty_since.pop(reserved_idx, None)
                                
                                # Start tracking exiting car
                                if 0 <= reserved_idx < len(self.posList):
                                    sx, sy, sw, sh = self.posList[reserved_idx]
                                    slot_center = (sx + sw // 2, sy + sh // 2)
                                    self._exiting_cars_tracking_pos[reserved_idx] = slot_center
                                    self._exiting_cars_loss_frames[reserved_idx] = 0
                                    self._exiting_cars_expected_area[reserved_idx] = None
                                
                                if self._recommended_slot == reserved_idx:
                                    self._recommended_slot = None
                                    self._recommended_occupied_start = None
                        else:
                            self._reserved_empty_since.pop(reserved_idx, None)

                    # ── Find valid moving objects via Contours ──────────────────
                    # Since YOLO won't reliably recognize a top-down toy car, we use
                    # background subtraction, but heavily filtered to ignore pencils/hands.
                    all_objects: List[Dict[str, Any]] = []

                    occupied_indices = [i for i in range(len(self.posList)) if i not in empty_indices]
                    reserved_slots = set(self._get_reserved_slots())
                    
                    if not hasattr(self, '_slot_occupied_since'):
                        self._slot_occupied_since = {}
                    now_ts = time.time()
                    for idx in occupied_indices:
                        if idx not in reserved_slots:
                            if idx not in self._slot_occupied_since:
                                self._slot_occupied_since[idx] = now_ts
                    for idx in list(self._slot_occupied_since.keys()):
                        if idx not in occupied_indices or idx in reserved_slots:
                            del self._slot_occupied_since[idx]
                            
                    blocked_rects = []
                    recommended = getattr(self, '_recommended_slot', None)
                    
                    min_px_x = min(p[0] for p in self.posList) - 40
                    max_px_x = max(p[0] + p[2] for p in self.posList) + 40
                    min_px_y = min(p[1] for p in self.posList) - 40
                    max_px_y = max(p[1] + p[3] for p in self.posList) + 40
                    
                    for idx, pos in enumerate(self.posList):
                        if idx != gate_idx and idx != recommended:
                            is_recently_occupied = (now_ts - self._slot_occupied_since.get(idx, 0) < 15.0)
                            if not is_recently_occupied and (idx in occupied_indices or idx in reserved_slots):
                                blocked_rects.append(pos)
                    
                    if thresh is not None:
                        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        avg_slot_area = (
                            sum(w * h for (_, _, w, h) in self.posList) / len(self.posList)
                            if self.posList else 2000
                        )
                        car_min = avg_slot_area * 0.25
                        car_max = avg_slot_area * 1.50

                        for c in contours:
                            area = cv2.contourArea(c)
                            if car_min < area < car_max:
                                # STRICT IGNORE FILTER for pencils and hands
                                rect = cv2.minAreaRect(c)
                                r_width, r_height = rect[1]
                                if r_width == 0 or r_height == 0:
                                    continue
                                    
                                aspect_ratio = max(r_width, r_height) / min(r_width, r_height)
                                
                                # 1. Pencils are very long and thin (high aspect ratio)
                                # 2. Cars are generally rectangular (aspect ratio ~ 1.2 to 2.5)
                                if aspect_ratio > 3.0:
                                    continue  # Ignore: Too long/thin (pencil or arm)
                                    
                                hull = cv2.convexHull(c)
                                hull_area = cv2.contourArea(hull)
                                if hull_area > 0:
                                    solidity = float(area) / hull_area
                                    # 3. Hands/fingers have irregular shapes with gaps (low solidity)
                                    # 4. Cars are solid blocks (high solidity)
                                    if solidity < 0.6:
                                        continue  # Ignore: Too irregular (hand)

                                M = cv2.moments(c)
                                if M["m00"] != 0:
                                    cx = int(M["m10"] / M["m00"])
                                    cy = int(M["m01"] / M["m00"])
                                    
                                    if not (min_px_x <= cx <= max_px_x and min_px_y <= cy <= max_px_y):
                                        continue  # Outside lot
                                        
                                    is_blocked = False
                                    for (brx, bry, brw, brh) in blocked_rects:
                                        # Expand blocking margin to swallow shadows and bleeding pixels
                                        bm = 30
                                        if brx - bm <= cx <= brx + brw + bm and bry - bm <= cy <= bry + brh + bm:
                                            is_blocked = True
                                            break
                                    
                                    if not is_blocked:
                                        all_objects.append({'cam': (cx, cy), 'area': area})

                    # ── Track Exiting Cars ────────────────────────────
                    for s_idx in list(self._exiting_cars_tracking_pos.keys()):
                        current_pos = self._exiting_cars_tracking_pos[s_idx]
                        expected_area = self._exiting_cars_expected_area.get(s_idx)
                        
                        dynamic_jump_px = 65
                        best_obj = None
                        best_score = float("inf")
                        for obj in all_objects:
                            d = math.dist(obj['cam'], current_pos)
                            if d > dynamic_jump_px:
                                continue
                            area_term = 0.0
                            if expected_area and expected_area > 1:
                                area_term = abs(obj['area'] - expected_area) / expected_area
                            score = d + (35.0 * area_term)
                            if score < best_score:
                                best_score = score
                                best_obj = obj
                                
                        if best_obj is not None:
                            best_cam = best_obj['cam']
                            SMOOTH_ALPHA = 0.45
                            self._exiting_cars_tracking_pos[s_idx] = (
                                int(current_pos[0] * (1 - SMOOTH_ALPHA) + best_cam[0] * SMOOTH_ALPHA),
                                int(current_pos[1] * (1 - SMOOTH_ALPHA) + best_cam[1] * SMOOTH_ALPHA),
                            )
                            if not expected_area:
                                self._exiting_cars_expected_area[s_idx] = best_obj['area']
                            else:
                                self._exiting_cars_expected_area[s_idx] = 0.85 * expected_area + 0.15 * best_obj['area']
                            self._exiting_cars_loss_frames[s_idx] = 0
                        else:
                            self._exiting_cars_loss_frames[s_idx] += 1
                            if self._exiting_cars_loss_frames[s_idx] > 30: # 1 second loss -> drop tracking
                                del self._exiting_cars_tracking_pos[s_idx]
                                del self._exiting_cars_loss_frames[s_idx]
                                if s_idx in self._exiting_cars_expected_area:
                                    del self._exiting_cars_expected_area[s_idx]

                    # ── Evaluate Guidance ─────────────────────────────────────────
                    is_full_lot = False

                    if all_objects:
                        if getattr(self, '_tracking_car_pos', None) is None:
                            # START tracking: MUST start at the contour physically nearest the gate,
                            # ignoring random large blobs in the background.
                            if gate_occupied and self._session_active_slot_idx is None:
                                # Only start tracking if guidance is enabled (driver confirmed plate).
                                if not self._guidance_enabled:
                                    self._check_pending_session()  # will flip flag if session exists
                                if not self._guidance_enabled:
                                    # Still waiting — draw green dot at gate but no path
                                    pass
                                else:
                                    gx, gy, gw, gh = self.posList[gate_idx]
                                    gate_center = (gx + gw // 2, gy + gh // 2)
                                    closest_to_gate = min(
                                        all_objects,
                                        key=lambda o: math.dist(o['cam'], gate_center)
                                    )
                                    # Expanded search radius: accept any object within
                                    # 60 % of the frame width so we always start tracking
                                    # even if the car isn't pixel-perfect at gate center.
                                    h_img, w_img = img.shape[:2]
                                    max_dist = w_img * 0.60
                                    if math.dist(closest_to_gate['cam'], gate_center) < max_dist:
                                        self._tracking_car_pos = closest_to_gate['cam']
                                        self._last_known_car_pos = closest_to_gate['cam']
                                        self._tracking_loss_frames = 0
                                        self._tracking_expected_area = closest_to_gate['area']
                            elif self._session_active_slot_idx is not None:
                                # RECOVER tracking mid-journey.
                                # Only use the target slot center as a fallback
                                # if we have absolutely no last-known position.
                                # Prefer last known car pos to avoid snapping to
                                # the wrong slot when the car is still near the gate.
                                if getattr(self, '_last_known_car_pos', None) is not None:
                                    recover_center = self._last_known_car_pos
                                    closest_to_target = min(
                                        all_objects,
                                        key=lambda o: math.dist(o['cam'], recover_center)
                                    )
                                    self._tracking_car_pos = closest_to_target['cam']
                                    self._last_known_car_pos = closest_to_target['cam']
                                    self._tracking_loss_frames = 0
                                    self._tracking_expected_area = closest_to_target['area']
                                # If no last_known position at all, leave tracking as None;
                                # the dot will only appear once tracking re-initialises.
                        else:
                            # CONTINUOUS tracking: Find contour closest to the car's last known position
                            # rather than just the largest one. This prevents "teleporting" to shadows.
                            current_pos = self._tracking_car_pos

                            dynamic_jump_px = 250 + min(150, int(self._tracking_last_step * 3.0))
                            expected_area = self._tracking_expected_area
                            best_obj = None
                            best_score = float("inf")
                            for obj in all_objects:
                                d = math.dist(obj['cam'], current_pos)
                                if d > dynamic_jump_px:
                                    continue

                                area_term = 0.0
                                if expected_area and expected_area > 1:
                                    area_term = abs(obj['area'] - expected_area) / expected_area

                                score = d + (35.0 * area_term)
                                if score < best_score:
                                    best_score = score
                                    best_obj = obj

                            if best_obj is not None:
                                best_cam = best_obj['cam']
                                # APPLY EMA SMOOTHING from user's algorithm
                                SMOOTH_ALPHA = 0.45
                                smoothed_pos = (
                                    int(current_pos[0] * (1 - SMOOTH_ALPHA) + best_cam[0] * SMOOTH_ALPHA),
                                    int(current_pos[1] * (1 - SMOOTH_ALPHA) + best_cam[1] * SMOOTH_ALPHA),
                                )
                                self._tracking_last_step = math.dist(current_pos, smoothed_pos)
                                self._tracking_car_pos = smoothed_pos
                                self._last_known_car_pos = smoothed_pos
                                if self._tracking_expected_area is None:
                                    self._tracking_expected_area = best_obj['area']
                                else:
                                    self._tracking_expected_area = (
                                        0.85 * self._tracking_expected_area + 0.15 * best_obj['area']
                                    )
                                self._tracking_loss_frames = 0
                            else:
                                self._tracking_loss_frames += 1
                                if self._tracking_loss_frames > 45: # Tolerant 1.5s visual dropout
                                    self._tracking_car_pos = None
                                    self._tracking_loss_frames = 0
                                    self._tracking_last_step = 0.0
                                    self._tracking_expected_area = None
                    else:
                        # No moving blobs this frame.  Hold last known position during grace period.
                        grace_limit = getattr(self, '_tracking_loss_frames', 0) + 1
                        self._tracking_loss_frames = grace_limit
                        # Keep guidance stable longer when a car is already being guided
                        # or when parking verification is in progress.
                        hold_limit = 25
                        if self._recommended_slot is not None:
                            hold_limit = 300
                        if self._recommended_occupied_start is not None:
                            hold_limit = 450
                        if grace_limit > hold_limit:
                            self._tracking_car_pos = None
                            self._tracking_loss_frames = 0
                            self._tracking_last_step = 0.0
                            self._tracking_expected_area = None

                    tracking_active = getattr(self, '_tracking_car_pos', None) is not None

                    if not tracking_active and self._session_active_slot_idx is None:
                        if getattr(self, '_parked_confirm_until', None) is None or time.time() > self._parked_confirm_until:
                            closest = None
                            self._recommended_slot = None
                            self._recommended_occupied_start = None
                            self._parked_confirm_until = None
                            self._tracking_expected_area = None
                    else:
                        # Force closest to active session if we have one and lost tracking, so path stays alive!
                        if not tracking_active and self._session_active_slot_idx is not None:
                            closest = self._session_active_slot_idx
                            self._recommended_slot = self._session_active_slot_idx

                        # Actively tracking a car (or recovering) — draw dot + live-recalculated path each frame
                        
                        # ── Confirm parking when the recommended slot turns occupied ─
                        if self._parked_confirm_until is not None:
                            # Showing confirmation banner
                            if time.time() < self._parked_confirm_until:
                                h_img, w_img = img.shape[:2]
                                confirmed_slot = self._recommended_slot
                                if confirmed_slot is not None:
                                    cvzone.putTextRect(
                                        img,
                                        f"✓ Parked in Slot {confirmed_slot + 1}!",
                                        (w_img // 2 - 220, h_img // 2),
                                        scale=2.5, thickness=3, offset=10,
                                        colorR=(0, 160, 0),
                                    )
                                closest = None
                            else:
                                # Confirmation banner expired
                                self._parked_confirm_until = None
                                self._recommended_slot = None
                                self._recommended_occupied_start = None
                                closest = self.find_closest_empty_slot(gate_idx, empty_excl_gate)
                                is_full_lot = (closest is None)
                                if closest is not None:
                                    self._recommended_slot = closest
                                    if gate_occupied and self._session_active_slot_idx is None:
                                        self._supabase_start_session(closest)
                                        self._session_active_slot_idx = closest

                        elif self._recommended_slot is not None and self._recommended_slot != gate_idx:
                            # Find which slot the tracked car is currently in (to allow parking in ANY slot)
                            car_in_any_slot_idx = None
                            now_ts = time.time()
                            if self._tracking_car_pos is not None:
                                tx, ty = self._tracking_car_pos
                                for idx_test, (sx, sy, sw, sh) in enumerate(self.posList):
                                    if idx_test == gate_idx or idx_test == exit_gate_idx:
                                        continue
                                    margin = -20  # Expand acceptable area
                                    if (sx + margin <= tx <= sx + sw - margin
                                        and sy + margin <= ty <= sy + sh - margin):
                                        car_in_any_slot_idx = idx_test
                                        break

                            # ── Occupancy-change fallback (works even when tracking is lost) ──
                            # Use raw background-subtraction to detect which slot the car
                            # parked in when the green dot tracking failed mid-journey.
                            if car_in_any_slot_idx is None and self._guidance_enabled:
                                newly_occupied = [
                                    idx for idx in occupied_indices
                                    if idx not in getattr(self, '_occupied_prev', set())
                                    and idx != gate_idx
                                    and idx != exit_gate_idx
                                ]
                                if newly_occupied:
                                    if getattr(self, '_last_known_car_pos', None) is not None:
                                        ref = self._last_known_car_pos
                                        car_in_any_slot_idx = min(
                                            newly_occupied,
                                            key=lambda i: math.dist(
                                                (self.posList[i][0] + self.posList[i][2] // 2,
                                                 self.posList[i][1] + self.posList[i][3] // 2),
                                                ref,
                                            )
                                        )
                                    elif len(newly_occupied) == 1:
                                        car_in_any_slot_idx = newly_occupied[0]
                                    # Re-anchor the dot to the detected slot center
                                    if car_in_any_slot_idx is not None:
                                        sx, sy, sw, sh = self.posList[car_in_any_slot_idx]
                                        self._tracking_car_pos = (sx + sw // 2, sy + sh // 2)
                                        self._last_known_car_pos = self._tracking_car_pos

                            if car_in_any_slot_idx is not None:
                                self._car_seen_in_recommended_until = now_ts + 2.0
                                self._recent_any_slot_idx = car_in_any_slot_idx
                                # ── Driver deviated: steer recommendation to actual slot ──
                                if car_in_any_slot_idx != self._recommended_slot:
                                    now_rd = time.time()
                                    if now_rd - self._last_redirect_time < self._redirect_cooldown:
                                        pass  # Still in cooldown, skip redirect to avoid flicker
                                    else:
                                        self._last_redirect_time = now_rd
                                        old = self._recommended_slot
                                        self._recommended_slot = car_in_any_slot_idx
                                        # Transfer any in-progress session to the new slot
                                        if self._session_active_slot_idx == old:
                                            sess_id = self._active_session_by_slot.pop(old, None)
                                            if sess_id:
                                                self._active_session_by_slot[car_in_any_slot_idx] = sess_id
                                                self._supabase_redirect_session(sess_id, car_in_any_slot_idx)
                                            self._session_active_slot_idx = car_in_any_slot_idx
                                        # Reset verification timer so it starts fresh for the new slot
                                        self._recommended_occupied_start = None
                                        self._recommended_occupied_loss_time = None
                                        logger.info(
                                            f"Driver deviated: redirected recommendation "
                                            f"from slot {old + 1} → slot {car_in_any_slot_idx + 1}"
                                        )

                            recent_in_slot = (
                                self._car_seen_in_recommended_until is not None
                                and now_ts <= self._car_seen_in_recommended_until
                            )
                            
                            # Decide which slot we are evaluating: always use current recommended
                            # (which has already been updated above if driver deviated)
                            recent_slot = getattr(self, '_recent_any_slot_idx', None)
                            if recent_in_slot and recent_slot is not None:
                                rec_idx = recent_slot
                            else:
                                rec_idx = self._recommended_slot

                            slot_is_occupied = rec_idx not in empty_excl_gate
                            car_in_eval_slot = (car_in_any_slot_idx == rec_idx)
                            
                            # Start/check confirmation only when the tracked car is inside the evaluated slot
                            # OR was seen there very recently, OR if it's our active session and it physically filled up
                            condition_met = slot_is_occupied and (
                                car_in_eval_slot 
                                or (recent_in_slot and recent_slot == rec_idx)
                                or (self._session_active_slot_idx == rec_idx)
                            )
                            
                            if condition_met:
                                getattr(self, '_recommended_occupied_loss_time', None) # Initialise securely
                                self._recommended_occupied_loss_time = None # Reset loss timer
                                
                                if self._recommended_occupied_start is None:
                                    self._recommended_occupied_start = time.time()
                                    self._verifying_slot = rec_idx

                                elapsed = time.time() - self._recommended_occupied_start
                                if elapsed >= 5.0:
                                    # 5 full seconds have passed! Car is truly parked.
                                    self._reserve_slot(rec_idx)
                                    # Show banner for 3 seconds
                                    self._parked_confirm_until = time.time() + 3.0
                                    self._db_saved_banner_until = time.time() + 3.0
                                    logger.info(
                                        f"Car parked in slot {rec_idx + 1}. "
                                        "Slot reserved via 5s wait; next car will be directed elsewhere."
                                    )
                                    
                                    # Session completion happens later when the car reaches the exit gate!
                                    # Do not clear the active session yet.
                                    
                                    closest = None
                                    self._tracking_car_pos = None  # Stop tracking
                                    self._last_known_car_pos = None
                                    self._tracking_last_step = 0.0
                                    self._tracking_expected_area = None
                                    if hasattr(self, '_slot_occupied_since'):
                                        self._slot_occupied_since.pop(rec_idx, None)
                                    
                                    # Transfer session ID if parked in a different slot than originally recommended
                                    if self._session_active_slot_idx is not None and self._session_active_slot_idx != rec_idx:
                                        sess_id = self._active_session_by_slot.pop(self._session_active_slot_idx, None)
                                        if sess_id:
                                            self._active_session_by_slot[rec_idx] = sess_id
                                            
                                    self._session_active_slot_idx = None  # CLEAR SESSION ALLOWING NEXT CAR!
                                    # Reset guidance gate so the NEXT driver must confirm plate first
                                    self._guidance_enabled = False
                                    self._last_pending_poll = 0.0
                                else:
                                    # Still waiting down the 5s timer
                                    closest = rec_idx

                                    h_img, w_img = img.shape[:2]
                                    countdown = max(0, 5 - int(elapsed))
                                    cvzone.putTextRect(
                                        img,
                                        f"Verifying space {rec_idx + 1}: {countdown}s...",
                                        (w_img // 2 - 220, h_img // 2 + 150),
                                        scale=1.8, thickness=2, offset=8,
                                        colorR=(0, 120, 200),
                                    )
                            else:
                                # Conditions not met. Tolerate up to 1.5 seconds of loss before resetting timer.
                                if self._recommended_occupied_start is not None:
                                    loss_time = getattr(self, '_recommended_occupied_loss_time', None)
                                    if loss_time is None:
                                        self._recommended_occupied_loss_time = time.time()
                                    elif time.time() - loss_time > 1.5:
                                        # Lost for too long, reset EVERYTHING
                                        self._recommended_occupied_start = None
                                        self._car_seen_in_recommended_until = None
                                        self._recommended_occupied_loss_time = None
                                    else:
                                        # Still tolerating loss, show countdown as paused
                                        closest = rec_idx
                                        elapsed = time.time() - self._recommended_occupied_start
                                        h_img, w_img = img.shape[:2]
                                        countdown = max(0, 5 - int(elapsed))
                                        cvzone.putTextRect(
                                            img,
                                            f"Verifying space {rec_idx + 1}: {countdown}s...",
                                            (w_img // 2 - 220, h_img // 2 + 150),
                                            scale=1.8, thickness=2, offset=8,
                                            colorR=(0, 120, 200),
                                        )
                                else:
                                    closest = rec_idx
                        else:
                            # Empty, or car just left, or no valid recommendation yet.
                            self._recommended_occupied_start = None
                            
                            # Sticky logic: pick the best available slot OR stick to current
                            if self._recommended_slot is not None and self._recommended_slot in empty_excl_gate and self._recommended_slot not in reserved_slots:
                                closest = self._recommended_slot
                            else:
                                closest = self.find_closest_empty_slot(gate_idx, empty_excl_gate)
                                is_full_lot = (closest is None)
                                if closest is not None:
                                    self._recommended_slot = closest
                                    if gate_occupied and self._session_active_slot_idx is None:
                                        self._supabase_start_session(closest)
                                        self._session_active_slot_idx = closest

                            # Car detected at gate with an active recommendation:
                            # create session once (start_time/is_completed=false).
                            if gate_occupied and self._session_active_slot_idx is None and self._recommended_slot is not None:
                                self._supabase_start_session(self._recommended_slot)
                                self._session_active_slot_idx = self._recommended_slot

                    # ── Draw guidance path (only when guidance is enabled) ────
                    # Green dot always drawn by draw_path_guidance via moving_pos.
                    # Guidance path and GATE highlight are suppressed until the
                    # driver confirms their plate.
                    if self._guidance_enabled:
                        self.draw_path_guidance(img, gate_idx, closest
                                                if getattr(self, '_parked_confirm_until', None) is None
                                                else None, is_full=is_full_lot,
                                                moving_pos=getattr(self, '_tracking_car_pos', None))
                    elif getattr(self, '_tracking_car_pos', None) is not None:
                        # Guidance not yet enabled but car is visible: draw green dot only
                        dot = self._tracking_car_pos
                        cv2.circle(img, dot, 8, (0, 255, 0), -1)
                        cv2.circle(img, dot, 14, (0, 150, 0), 2)
                        cvzone.putTextRect(img, "Waiting for confirmation...",
                                           (dot[0] - 80, dot[1] - 20),
                                           scale=1.0, thickness=1, offset=4,
                                           colorR=(30, 30, 30))
                    
                    # Draw red tracking dot for exiting cars
                    for s_idx, pos in self._exiting_cars_tracking_pos.items():
                        cv2.circle(img, pos, 8, (0, 0, 255), -1)  # Red inner circle
                        cv2.circle(img, pos, 14, (150, 0, 255), 2)  # Pink-ish outer ring
                        cvzone.putTextRect(
                            img,
                            f"EXITING {s_idx + 1}",
                            (pos[0] - 30, pos[1] - 20),
                            scale=1.2, thickness=2, offset=3,
                            colorR=(150, 0, 220),
                        )

                    # Always render the 10-second verification banner when timer is active,
                    # even if contour movement is temporarily missing.
                    if self._recommended_occupied_start is not None and getattr(self, '_verifying_slot', self._recommended_slot) is not None:
                        elapsed = time.time() - self._recommended_occupied_start
                        countdown = max(0, 5 - int(elapsed))
                        h_img, w_img = img.shape[:2]
                        v_slot = getattr(self, '_verifying_slot', self._recommended_slot)
                        cvzone.putTextRect(
                            img,
                            f"Verifying space {v_slot + 1}: {countdown}s...",
                            (w_img // 2 - 220, h_img // 2 + 150),
                            scale=1.8, thickness=2, offset=8,
                            colorR=(0, 120, 200),
                        )

                    # Complete active session when car reaches exit gate
                    if 0 <= exit_gate_idx < len(self.posList):
                        gx, gy, gw, gh = self.posList[exit_gate_idx]
                        margin = -10
                        
                        # Trigger completion if visual tracked exiting car enters exit gate
                        for s_idx in list(self._exiting_cars_tracking_pos.keys()):
                            tx, ty = self._exiting_cars_tracking_pos[s_idx]
                            if (gx + margin <= tx <= gx + gw - margin) and (gy + margin <= ty <= gy + gh - margin):
                                self._supabase_complete_session(s_idx)
                                if s_idx in self._slots_waiting_exit:
                                    self._slots_waiting_exit.remove(s_idx)
                                del self._exiting_cars_tracking_pos[s_idx]
                                del self._exiting_cars_loss_frames[s_idx]
                                if s_idx in self._exiting_cars_expected_area:
                                    del self._exiting_cars_expected_area[s_idx]
                                    f"CAR FINISHED {s_idx + 1}",

                    # Fallback trigger: if exit gate gets generally occupied, close the oldest waiting slot
                    if exit_gate_occupied and self._slots_waiting_exit:
                        s_idx = next(iter(self._slots_waiting_exit))
                        self._supabase_complete_session(s_idx)
                        self._slots_waiting_exit.remove(s_idx)
                        if s_idx in self._exiting_cars_tracking_pos:
                            del self._exiting_cars_tracking_pos[s_idx]
                            del self._exiting_cars_loss_frames[s_idx]
                        if s_idx in self._exiting_cars_expected_area:
                            del self._exiting_cars_expected_area[s_idx]
                # ─────────────────────────────────────────────────────────────

                # Add a small live-stream indicator in the top-left corner
                timestamp = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
                cvzone.putTextRect(
                    img,
                    f"LIVE  {timestamp}",
                    (10, 30),
                    scale=1.5,
                    thickness=2,
                    offset=5,
                    colorR=(0, 0, 200),
                )

                # Explicit DB write confirmation banner after successful reservation.
                if self._db_saved_banner_until is not None:
                    if time.time() < self._db_saved_banner_until and self._recommended_slot is not None:
                        h_img, w_img = img.shape[:2]
                        cvzone.putTextRect(
                            img,
                            f"DB Saved: Slot {self._recommended_slot + 1} reserved",
                            (w_img // 2 - 260, h_img // 2 + 210),
                            scale=1.4,
                            thickness=2,
                            offset=8,
                            colorR=(0, 100, 220),
                        )
                    else:
                        self._db_saved_banner_until = None

                # If no slots defined, overlay a clear guide message on the live feed
                if not self.posList:
                    h, w = img.shape[:2]
                    overlay = img.copy()
                    cv2.rectangle(overlay, (0, h // 2 - 60), (w, h // 2 + 60), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
                    cvzone.putTextRect(
                        img,
                        "No slots defined! Re-run with --setup",
                        (w // 2 - 320, h // 2 - 10),
                        scale=1.8,
                        thickness=2,
                        offset=8,
                        colorR=(0, 80, 200),
                    )
                    cvzone.putTextRect(
                        img,
                        f"python run.py --camera {url} --setup",
                        (w // 2 - 320, h // 2 + 40),
                        scale=1.2,
                        thickness=2,
                        offset=6,
                        colorR=(30, 30, 30),
                    )

                # Save CSV report every 30 frames — only when slots are configured
                if frame_count % 30 == 0 and self.posList:
                    self.generate_csv_report(
                        total_slots=len(self.posList),
                        occupied_slots=occupied_slots,
                        available_slots=available_slots,
                    )

                cv2.imshow(config.WINDOW_NAME, img)
                # Push the fully-annotated frame to the MJPEG server so the
                # browser frontend shows guidance overlays, not raw video.
                _mjpeg.push_frame(img)
                self._occupied_prev = set(occupied_indices)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            _mjpeg.stop()
            cv2.destroyAllWindows()
            logger.info("Camera stream processing completed")

    def setup_from_camera(self, camera_url: Optional[str] = None) -> None:
        """Capture a still frame from the live camera and open the slot-selection editor.

        This is the **first step** you must do before running camera detection:
        1. Connect to the camera stream
        2. Grab a clean reference frame
        3. Save it to ``camera_reference.jpg``
        4. Open the interactive slot-editor on that image
        5. Draw rectangles over each parking space and press **S** to save
        6. Exit — then run camera detection normally
        """
        url = camera_url or self.camera_url or config.CAMERA_URL
        if not url:
            raise ValueError("No camera URL provided for setup.")

        print(f"\n📷 Connecting to camera for setup: {url}")
        logger.info(f"Setup: connecting to {url}")

        cap = cv2.VideoCapture(url)
        if not cap.isOpened():
            raise IOError(
                f"Cannot connect to camera stream: {url}\n"
                "Check that the URL is reachable and the camera is online."
            )

        # Skip a handful of frames so the camera auto-exposure settles
        print("⏳ Waiting for camera to stabilise...")
        good_frame = None
        for _ in range(10):
            ok, frame = cap.read()
            if ok:
                good_frame = frame
        cap.release()

        if good_frame is None:
            raise IOError("Could not capture a frame from the camera during setup.")

        # Save the reference snapshot
        snapshot_path = "camera_reference.jpg"
        cv2.imwrite(snapshot_path, good_frame)
        logger.info(f"Saved camera reference frame to '{snapshot_path}'")
        print(f"✓ Saved reference frame → {snapshot_path}")

        print(
            "\n" + "=" * 62 + "\n"
            "  PARKING SLOT SETUP — draw rectangles over each space\n"
            "=" * 62 + "\n"
            "  Left-click + drag  → select an area of parking spaces\n"
            "  Right-click        → remove a parking space\n"
            "  S                  → SAVE layout and exit setup\n"
            "  Z                  → Undo last selection\n"
            "  R                  → Reset all selections\n"
            "  Q                  → Quit without saving\n"
            "=" * 62 + "\n"
        )

        # Reuse the existing image editor with the captured frame
        self.image_path = snapshot_path
        self.original_image = good_frame.copy()
        self.current_image = good_frame.copy()
        self.process_image()

        if self.posList:
            print(
                f"\n✅ Setup complete! {len(self.posList)} parking slot(s) saved.\n"
                "   Now run camera detection normally:\n"
                f"   python run.py --camera {url}\n"
            )
        else:
            print(
                "\n⚠️  No slots were saved. Re-run with --setup and press S to save."
            )

    def clear_all_markings(self) -> None:
        """Clear all markings and reset to original image"""
        if self.original_image is not None:
            self.current_image = self.original_image.copy()
        self.history.append(self.posList.copy())  # allow undo of a full reset
        self.posList = []
        self.is_reset = True
        self.save_parking_positions()
        logger.info("Cleared all markings")

    def undo_last_selection(self) -> None:
        """Remove the last selected grid completely"""
        if self.history:
            self.posList = self.history.pop()
            if self.original_image is not None:
                self.current_image = self.original_image.copy()
            self.save_parking_positions()
            logger.info("Undid last selection")
        else:
            logger.warning("No history to undo")

    def process_image(self) -> None:
        """Process image file for parking space selection and detection"""
        if not self.image_path:
            raise ValueError("Image path not provided")

        # Validate image file exists
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image file not found: {self.image_path}")

        logger.info(f"Processing image: {self.image_path}")

        # Read image with full resolution
        self.original_image = cv2.imread(self.image_path, cv2.IMREAD_UNCHANGED)
        if self.original_image is None:
            raise IOError(f"Could not load image from {self.image_path}")

        self.current_image = self.original_image.copy()

        # Get image dimensions
        height, width = self.original_image.shape[:2]

        # Calculate window size with borders
        border_size = config.BORDER_SIZE
        window_width = width + 2 * border_size
        window_height = height + 2 * border_size

        # Create a window that can be resized
        cv2.namedWindow(config.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(config.WINDOW_NAME, window_width, window_height)

        # Set mouse callback
        cv2.setMouseCallback(config.WINDOW_NAME, self.mouse_callback)

        try:
            while True:
                # Create a bordered image
                bordered_img = np.zeros((window_height, window_width, 3), dtype=np.uint8)
                bordered_img[
                    border_size : border_size + height, border_size : border_size + width
                ] = self.current_image.copy()

                # Only draw parking spaces if there are any and not in reset state
                if self.posList and not self.is_reset:
                    for idx, pos in enumerate(self.posList):
                        px, py, pw, ph = pos
                        # Adjust coordinates for border
                        draw_x = px + border_size
                        draw_y = py + border_size
                        cv2.rectangle(
                            bordered_img,
                            (draw_x, draw_y),
                            (draw_x + pw, draw_y + ph),
                            (255, 0, 255),
                            2,
                        )
                        # Label each slot with its index number
                        cv2.putText(
                            bordered_img,
                            str(idx + 1),
                            (draw_x + 4, draw_y + 16),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                # Draw lot boundary if it exists
                if self.lot_boundary:
                    bx, by, bw, bh = self.lot_boundary
                    draw_bx = bx + border_size
                    draw_by = by + border_size
                    cv2.rectangle(bordered_img, (draw_bx, draw_by), (draw_bx + bw, draw_by + bh), (255, 0, 0), 3)
                    cv2.putText(bordered_img, "LOT BOUNDARY", (draw_bx + 4, draw_by - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                # Draw the live selection rectangle while dragging
                if self.drawing and self.start_point and self.end_point:
                    x1, y1 = self.start_point
                    x2, y2 = self.end_point
                    # Adjust coordinates for border
                    x1 += border_size
                    y1 += border_size
                    x2 += border_size
                    y2 += border_size
                    
                    color = (255, 0, 0) if self.boundary_mode else (0, 255, 0)
                    cv2.rectangle(bordered_img, (x1, y1), (x2, y2), color, 2)
                    # Show live dimensions while drawing
                    dw = abs(x2 - x1)
                    dh = abs(y2 - y1)
                    cv2.putText(
                        bordered_img,
                        f"{dw}x{dh}",
                        (min(x1, x2) + 4, min(y1, y2) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        1,
                    )

                # Add essential information only if not in reset state
                if not self.is_reset:
                    total_spaces = len(self.posList)

                    # Only process image and show statistics if there are parking spaces (or boundary)
                    if total_spaces > 0 or self.lot_boundary:
                        if total_spaces > 0:
                            processed_img = self.process_frame(self.current_image)
                            available_slots, occupied_slots, _empty_img = self.check_parking_space(
                                processed_img, self.current_image
                            )
                            empty_spaces = available_slots

                            # Display statistics in the top-right corner
                            cvzone.putTextRect(
                                bordered_img,
                                f"Total Slots: {total_spaces}",
                                (window_width - 250, 30),
                                scale=2,
                                thickness=3,
                                offset=10,
                                colorR=(0, 200, 0),
                            )
                            cvzone.putTextRect(
                                bordered_img,
                                f"Empty Slots: {empty_spaces}",
                                (window_width - 250, 80),
                                scale=2,
                                thickness=3,
                                offset=10,
                                colorR=(0, 200, 0),
                            )

                # Show help text if enabled
                if self.show_help:
                    self.add_help_text(bordered_img)

                # Apply zoom and pan
                display_img = self.apply_zoom_and_pan(bordered_img)

                cv2.imshow(config.WINDOW_NAME, display_img)

                key = cv2.waitKey(1) & 0xFF
                # Accept both uppercase and lowercase for all shortcuts (macOS fix)
                if key in (ord("q"), ord("Q")):
                    break
                elif key in (ord("b"), ord("B")):
                    self.boundary_mode = not self.boundary_mode
                    state = "ENABLED" if self.boundary_mode else "DISABLED"
                    logger.info(f"Boundary mode {state}")
                    cvzone.putTextRect(
                        bordered_img,
                        f"Boundary Mode {state}",
                        (window_width // 2 - 150, window_height - 50),
                        scale=2, thickness=2, offset=10,
                        colorR=(255, 0, 0) if self.boundary_mode else (0, 0, 0),
                    )
                    cv2.imshow(config.WINDOW_NAME, bordered_img)
                    cv2.waitKey(500)
                elif key in (ord("r"), ord("R")):
                    self.clear_all_markings()
                elif key in (ord("z"), ord("Z")):
                    self.undo_last_selection()
                elif key in (ord("s"), ord("S")):
                    self.save_parking_positions()
                    self.is_reset = False
                    cvzone.putTextRect(
                        bordered_img,
                        "Layout Saved!",
                        (window_width // 2 - 100, window_height - 50),
                        scale=2,
                        thickness=2,
                        offset=10,
                        colorR=(0, 255, 0),
                    )
                    cv2.imshow(config.WINDOW_NAME, bordered_img)
                    cv2.waitKey(1000)
                elif key == ord("d"):
                    if len(self.posList) == 0:
                        logger.warning("Please select parking spaces first!")
                        print("Please select parking spaces first!")
                        continue

                    logger.info("Starting vehicle detection and report generation...")
                    print("\n🚗 Detecting vehicles...")

                    # Process the image and detect cars
                    processed_img = self.process_frame(self.current_image)
                    available_slots, occupied_slots, _empty_d = self.check_parking_space(
                        processed_img, self.current_image
                    )

                    print("🤖 Running ML-based vehicle detection...")
                    # Run ML-based car detection
                    results = self.car_detector.detect_cars(self.current_image)
                    detections, space_status = self.car_detector.process_detections(
                        results, self.posList, self.current_image
                    )

                    print("📊 Generating visual report...")
                    # Generate comprehensive report
                    report_path, text_report_path = self.car_detector.generate_report(
                        self.current_image, self.posList, detections, space_status
                    )

                    print("💾 Saving CSV data...")
                    logger.info(f"Report generated successfully!")
                    print(f"\n✓ Report generated successfully!")
                    print(f"  Visual report: {report_path}")
                    print(f"  Text report: {text_report_path}")

                    # Generate CSV report
                    self.generate_csv_report(
                        total_slots=len(self.posList),
                        occupied_slots=occupied_slots,
                        available_slots=available_slots,
                    )

                    # Show success message
                    cvzone.putTextRect(
                        bordered_img,
                        "Report Generated!",
                        (window_width // 2 - 150, window_height - 50),
                        scale=3,
                        thickness=3,
                        offset=10,
                        colorR=(0, 255, 0),
                    )
                    cv2.imshow(config.WINDOW_NAME, bordered_img)
                    cv2.waitKey(2000)
        finally:
            cv2.destroyAllWindows()
            logger.info("Image processing completed")


if __name__ == "__main__":
    # Example usage
    detector = EnhancedParkingDetector(video_path="carPark.mp4", image_path="carParkImg.jpg")

    # Process image for parking space selection
    detector.process_image()
