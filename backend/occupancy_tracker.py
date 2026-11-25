import time
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from backend.tables_manager import tables_manager, TableModel
from backend.database import record_seating, record_vacated, record_service_alert, acknowledge_service
from backend.telegram_alerts import telegram_alerts
from backend.firebase_sync import firebase_manager

def is_point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Ray-casting algorithm for 2D Point-in-Polygon test.
    Works for both normalized [0..1] and pixel coordinates.
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y

    return inside

class TableState:
    def __init__(self, table: TableModel, debounce_sec: float = 4.0):
        self.table = table
        self.debounce_sec = debounce_sec
        self.status = "FREE"  # FREE, OCCUPIED, NEEDS_SERVICE
        self.raw_headcount = 0
        self.confirmed_headcount = 0
        self.seated_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self.service_alert_fired: bool = False
        self.service_acknowledged: bool = False

        # Debounce buffer: list of (timestamp, raw_count)
        self.history: List[Tuple[float, int]] = []

    def update_raw(self, current_raw_count: int, now: float) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Feed new raw observation and update state machine with debouncing.
        Returns (event_name, event_data) if a state change or alert triggered.
        """
        self.raw_headcount = current_raw_count
        self.history.append((now, current_raw_count))

        # Keep only recent history within debounce window
        cutoff = now - self.debounce_sec
        self.history = [(t, c) for t, c in self.history if t >= cutoff]

        event_name = None
        event_data = None

        # Check debounce condition: has the observation been steady across the window?
        # A window is considered valid if it covers at least (debounce_sec * 0.75)
        window_duration = (self.history[-1][0] - self.history[0][0]) if self.history else 0
        is_window_full = window_duration >= (self.debounce_sec * 0.7)

        # Count positive frames vs zero frames
        positive_frames = sum(1 for _, c in self.history if c > 0)
        zero_frames = sum(1 for _, c in self.history if c == 0)
        total_frames = len(self.history)

        if total_frames > 0 and is_window_full:
            majority_positive = (positive_frames / total_frames) >= 0.75
            majority_zero = (zero_frames / total_frames) >= 0.75

            # Transition: FREE -> OCCUPIED
            if self.status == "FREE" and majority_positive:
                recent_counts = [c for _, c in self.history if c > 0]
                median_count = int(np.median(recent_counts)) if recent_counts else 1
                self.status = "OCCUPIED"
                self.confirmed_headcount = max(1, median_count)
                self.seated_time = now
                self.last_state_change = now
                self.service_alert_fired = False
                self.service_acknowledged = False

                # Record in DB
                record_seating(self.table.id, self.table.name, self.confirmed_headcount)
                event_name = "SEATED"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "headcount": self.confirmed_headcount
                }

            # Transition: OCCUPIED / NEEDS_SERVICE -> FREE
            elif self.status in ("OCCUPIED", "NEEDS_SERVICE") and majority_zero:
                duration_sec = int(now - (self.seated_time or now))
                self.status = "FREE"
                self.confirmed_headcount = 0
                self.seated_time = None
                self.last_state_change = now
                self.service_alert_fired = False
                self.service_acknowledged = False

                record_vacated(self.table.id)
                event_name = "VACATED"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "duration_sec": duration_sec
                }

        # Check Service Alert Threshold
        if self.status in ("OCCUPIED", "NEEDS_SERVICE") and self.seated_time:
            elapsed = now - self.seated_time
            if elapsed >= self.table.service_threshold_sec and not self.service_alert_fired and not self.service_acknowledged:
                self.status = "NEEDS_SERVICE"
                self.service_alert_fired = True
                record_service_alert(self.table.id)
                event_name = "SERVICE_ALERT"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "elapsed_sec": int(elapsed)
                }

                # Trigger real-time mobile Telegram push alert & Firebase sync
                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                elapsed_display = f"{elapsed_min}:{elapsed_sec:02d}"
                telegram_alerts.send_alert(self.table.name, elapsed_display, self.confirmed_headcount, table_id=self.table.id)
                firebase_manager.log_event({
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "event_type": "SERVICE_ALERT",
                    "elapsed_sec": int(elapsed),
                    "headcount": self.confirmed_headcount
                })

        # If occupied, update confirmed headcount if sustained higher count
        if self.status in ("OCCUPIED", "NEEDS_SERVICE") and current_raw_count > self.confirmed_headcount:
            self.confirmed_headcount = current_raw_count

        return event_name, event_data

    def get_elapsed_seconds(self, now: float) -> int:
        if self.seated_time and self.status in ("OCCUPIED", "NEEDS_SERVICE"):
            return int(now - self.seated_time)
        return 0

    def acknowledge_service(self):
        self.service_acknowledged = True
        if self.status == "NEEDS_SERVICE":
            self.status = "OCCUPIED"
        acknowledge_service(self.table.id)


class OccupancyTracker:
    def __init__(self, debounce_sec: float = 4.0):
        self.debounce_sec = debounce_sec
        self.states: Dict[str, TableState] = {}
        self._sync_tables()

    def set_debounce(self, debounce_sec: float):
        self.debounce_sec = debounce_sec
        for s in self.states.values():
            s.debounce_sec = debounce_sec

    def _sync_tables(self):
        all_tables = tables_manager.get_all()
        current_ids = set()
        for t in all_tables:
            current_ids.add(t.id)
            if t.id not in self.states:
                self.states[t.id] = TableState(t, self.debounce_sec)
            else:
                self.states[t.id].table = t

        # Remove deleted tables
        for tid in list(self.states.keys()):
            if tid not in current_ids:
                del self.states[tid]

    def update(
        self,
        person_boxes: List[Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
        active_channel: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        person_boxes: list of (x1, y1, x2, y2) in pixel coordinates.
        active_channel: optional channel number (1-8) to restrict bounding box tests.
        Returns:
          (table_statuses, emitted_events)
        """
        self._sync_tables()
        now = time.time()

        # Compute bottom-center anchor points in normalized coordinates [0.0 .. 1.0]
        person_anchors = []
        for (x1, y1, x2, y2) in person_boxes:
            # Bottom-center anchor
            bx = ((x1 + x2) / 2.0) / max(1.0, float(frame_width))
            by = float(y2) / max(1.0, float(frame_height))
            person_anchors.append((bx, by))

        # Check each table
        table_statuses = []
        emitted_events = []

        for table_id, state in self.states.items():
            poly_norm = state.table.polygon
            t_channel = getattr(state.table, 'camera_channel', 1)

            # If active_channel is specified and does not match, preserve previous observation
            if active_channel is not None and t_channel != active_channel:
                count = state.raw_headcount
            else:
                # Count people inside table polygon
                count = 0
                for anchor in person_anchors:
                    if is_point_in_polygon(anchor, poly_norm):
                        count += 1

            # Update table state
            evt_name, evt_data = state.update_raw(count, now)
            if evt_name and evt_data:
                emitted_events.append({"event": evt_name, "data": evt_data})

            elapsed = state.get_elapsed_seconds(now)
            elapsed_m = elapsed // 60
            elapsed_s = elapsed % 60
            elapsed_str = f"{elapsed_m:02d}:{elapsed_s:02d}"

            # Convert polygon to pixel coordinates for frontend rendering / canvas
            poly_px = [
                [int(pt[0] * frame_width), int(pt[1] * frame_height)]
                for pt in poly_norm
            ]

            table_statuses.append({
                "id": state.table.id,
                "name": state.table.name,
                "camera_channel": t_channel,
                "capacity": state.table.capacity,
                "service_threshold_sec": state.table.service_threshold_sec,
                "status": state.status,
                "raw_count": state.raw_headcount,
                "headcount": state.confirmed_headcount,
                "elapsed_seconds": elapsed,
                "elapsed_formatted": elapsed_str,
                "service_alert": state.service_alert_fired,
                "service_acknowledged": state.service_acknowledged,
                "polygon_norm": poly_norm,
                "polygon_px": poly_px
            })

        return table_statuses, emitted_events

    def acknowledge_service(self, table_id: str) -> bool:
        if table_id in self.states:
            self.states[table_id].acknowledge_service()
            return True
        return False
