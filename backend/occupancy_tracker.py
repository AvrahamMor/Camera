import time
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


def expand_polygon(polygon: List[List[float]], scale: float = 1.40) -> List[List[float]]:
    """Expand normalized table polygon outward around its centroid for waiter-approach detection."""
    if len(polygon) < 3:
        return polygon
    cx = sum(pt[0] for pt in polygon) / len(polygon)
    cy = sum(pt[1] for pt in polygon) / len(polygon)
    expanded = []
    for px, py in polygon:
        nx = max(0.0, min(1.0, cx + (px - cx) * scale))
        ny = max(0.0, min(1.0, cy + (py - cy) * scale))
        expanded.append([nx, ny])
    return expanded


class TableState:
    def __init__(self, table: TableModel, debounce_sec: float = 4.0, menu_alert_sec: int = 120):
        self.table = table
        self.debounce_sec = debounce_sec
        self.menu_alert_sec = menu_alert_sec

        self.status = "FREE"  # FREE, OCCUPIED, NEEDS_SERVICE
        self.raw_headcount = 0
        self.raw_ring_count = 0
        self.confirmed_headcount = 0
        self.seated_time: Optional[float] = None
        self.last_state_change: float = time.time()
        self.last_waiter_visit_time: Optional[float] = None
        self.last_seated_seen_time: Optional[float] = None

        self.menu_served: bool = False
        self.no_menu_alert_fired: bool = False
        self.service_alert_fired: bool = False
        self.service_acknowledged: bool = False

        # Waiter approach timers (filters quick walk-bys / aisle glances)
        self.waiter_ring_start: Optional[float] = None
        self.hand_reach_start: Optional[float] = None
        self.inner_spike_start: Optional[float] = None

        # Debounce buffer: list of (timestamp, raw_count)
        self.history: List[Tuple[float, int]] = []

    def update_raw(
        self,
        current_raw_count: int,
        now: float,
        ring_count: int = 0,
        hold_count: Optional[int] = None,
        hand_reach_count: int = 0,
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        Feed new raw observation and update state machine with asymmetric hysteresis:
        - Seating (FREE -> OCCUPIED) uses strict high-confidence detections + debounce_sec (~4s).
        - Keeping/Vacating uses sticky hold_count (lower conf >= 0.20) + >=20s continuous empty table.
        """
        effective_hold = current_raw_count if hold_count is None else max(current_raw_count, hold_count)
        self.raw_headcount = effective_hold if self.status in ("OCCUPIED", "NEEDS_SERVICE") else current_raw_count
        self.raw_ring_count = ring_count

        self.history.append((now, current_raw_count))
        cutoff = now - self.debounce_sec
        self.history = [(t, c) for t, c in self.history if t >= cutoff]

        event_name = None
        event_data = None

        window_duration = (self.history[-1][0] - self.history[0][0]) if self.history else 0
        is_window_full = window_duration >= (self.debounce_sec * 0.9)

        positive_frames = sum(1 for _, c in self.history if c > 0)
        total_frames = len(self.history)

        if total_frames >= 3 and is_window_full:
            majority_positive = (positive_frames / total_frames) >= 0.80 and positive_frames >= 3

            # Transition: FREE -> OCCUPIED (strict high-confidence trigger)
            if self.status == "FREE" and majority_positive:
                recent_counts = [c for _, c in self.history if c > 0]
                median_count = int(np.median(recent_counts)) if recent_counts else 1
                self.status = "OCCUPIED"
                self.confirmed_headcount = max(1, median_count)
                self.seated_time = now
                self.last_state_change = now
                self.last_seated_seen_time = now
                self.last_waiter_visit_time = None
                self.menu_served = False
                self.no_menu_alert_fired = False
                self.service_alert_fired = False
                self.service_acknowledged = False
                self.waiter_ring_start = None
                self.hand_reach_start = None
                self.inner_spike_start = None

                record_seating(self.table.id, self.table.name, self.confirmed_headcount)
                event_name = "SEATED"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "headcount": self.confirmed_headcount
                }

        # Sticky occupied hold + 20-second asymmetric vacate timer
        if self.status in ("OCCUPIED", "NEEDS_SERVICE"):
            if effective_hold > 0:
                self.last_seated_seen_time = now
            else:
                vacate_sec = max(20.0, self.debounce_sec * 4.0)
                if self.last_seated_seen_time is not None and (now - self.last_seated_seen_time) >= vacate_sec:
                    duration_sec = int(now - (self.seated_time or now))
                    self.status = "FREE"
                    self.confirmed_headcount = 0
                    self.seated_time = None
                    self.last_state_change = now
                    self.last_seated_seen_time = None
                    self.last_waiter_visit_time = None
                    self.menu_served = False
                    self.no_menu_alert_fired = False
                    self.service_alert_fired = False
                    self.service_acknowledged = False
                    self.waiter_ring_start = None
                    self.inner_spike_start = None

                    record_vacated(self.table.id)
                    event_name = "VACATED"
                    event_data = {
                        "table_id": self.table.id,
                        "table_name": self.table.name,
                        "duration_sec": duration_sec
                    }

        # --- Waiter / Menu Approach Detection while occupied ---
        if self.status in ("OCCUPIED", "NEEDS_SERVICE"):
            required_dwell = 8.0 if self.status == "NEEDS_SERVICE" else 6.0
            required_reach = 2.0 if self.status == "NEEDS_SERVICE" else 1.5
            served_now = False

            # Signal A: Waiter at table edge reaches wrist/hand onto table surface (placing menus)
            if hand_reach_count >= 1:
                if self.hand_reach_start is None:
                    self.hand_reach_start = now
                elif (now - self.hand_reach_start) >= required_reach:
                    served_now = True
            else:
                self.hand_reach_start = None

            # Signal B: Waiter stands stationary right beside table edge (tight 1.22x ring) for >= 6.0s
            if ring_count >= 1:
                if self.waiter_ring_start is None:
                    self.waiter_ring_start = now
                elif (now - self.waiter_ring_start) >= required_dwell:
                    served_now = True
            else:
                self.waiter_ring_start = None

            if served_now:
                self.last_waiter_visit_time = now
                if not self.menu_served:
                    self.menu_served = True
                    if self.status == "NEEDS_SERVICE":
                        self.status = "OCCUPIED"
                        self.service_acknowledged = True
                    event_name = event_name or "MENU_SERVED"
                    event_data = event_data or {
                        "table_id": self.table.id,
                        "table_name": self.table.name,
                        "headcount": self.confirmed_headcount,
                    }

            # Signal C: Extra person stepped inside polygon for 5s..20s then left
            if current_raw_count > self.confirmed_headcount:
                if self.inner_spike_start is None:
                    self.inner_spike_start = now
                elif (now - self.inner_spike_start) >= 20.0:
                    # Extra guest sat down permanently -> update headcount
                    self.confirmed_headcount = current_raw_count
                    self.inner_spike_start = None
            else:
                if self.inner_spike_start is not None:
                    spike_dur = now - self.inner_spike_start
                    if 5.0 <= spike_dur < 20.0:
                        self.last_waiter_visit_time = now
                        if not self.menu_served:
                            self.menu_served = True
                            if self.status == "NEEDS_SERVICE":
                                self.status = "OCCUPIED"
                                self.service_acknowledged = True
                            event_name = event_name or "MENU_SERVED"
                            event_data = event_data or {
                                "table_id": self.table.id,
                                "table_name": self.table.name,
                                "headcount": self.confirmed_headcount,
                            }
                    self.inner_spike_start = None

        # --- Check 2-Minute No-Menu Alert (Unattended Seating) ---
        if (
            self.status in ("OCCUPIED", "NEEDS_SERVICE")
            and self.seated_time
            and not self.menu_served
            and not self.no_menu_alert_fired
            and not self.service_acknowledged
        ):
            elapsed = now - self.seated_time
            if elapsed >= self.menu_alert_sec:
                self.status = "NEEDS_SERVICE"
                self.no_menu_alert_fired = True
                self.service_alert_fired = True
                record_service_alert(self.table.id)

                elapsed_min = int(elapsed // 60)
                elapsed_sec = int(elapsed % 60)
                elapsed_display = f"{elapsed_min}:{elapsed_sec:02d}"

                telegram_alerts.send_no_menu_alert(
                    self.table.name, elapsed_display, self.confirmed_headcount, table_id=self.table.id
                )
                firebase_manager.log_event({
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "event_type": "NO_MENU_ALERT",
                    "elapsed_sec": int(elapsed),
                    "headcount": self.confirmed_headcount
                })
                event_name = "SERVICE_ALERT"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "elapsed_sec": int(elapsed),
                    "reason": "no_menu",
                }

        # --- Ongoing General Service Alert (counts from last waiter visit or seating) ---
        elif self.status in ("OCCUPIED", "NEEDS_SERVICE") and self.seated_time:
            ref_time = self.last_waiter_visit_time or self.seated_time
            elapsed_since_visit = now - ref_time
            if (
                elapsed_since_visit >= self.table.service_threshold_sec
                and not self.service_alert_fired
                and not self.service_acknowledged
            ):
                self.status = "NEEDS_SERVICE"
                self.service_alert_fired = True
                record_service_alert(self.table.id)
                event_name = "SERVICE_ALERT"
                event_data = {
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "elapsed_sec": int(elapsed_since_visit)
                }

                elapsed_min = int(elapsed_since_visit // 60)
                elapsed_sec = int(elapsed_since_visit % 60)
                elapsed_display = f"{elapsed_min}:{elapsed_sec:02d}"
                telegram_alerts.send_alert(
                    self.table.name, elapsed_display, self.confirmed_headcount, table_id=self.table.id
                )
                firebase_manager.log_event({
                    "table_id": self.table.id,
                    "table_name": self.table.name,
                    "event_type": "SERVICE_ALERT",
                    "elapsed_sec": int(elapsed_since_visit),
                    "headcount": self.confirmed_headcount
                })

        return event_name, event_data

    def get_elapsed_seconds(self, now: float) -> int:
        if self.seated_time and self.status in ("OCCUPIED", "NEEDS_SERVICE"):
            return int(now - self.seated_time)
        return 0

    def acknowledge_service(self):
        self.service_acknowledged = True
        self.menu_served = True
        self.last_waiter_visit_time = time.time()
        if self.status == "NEEDS_SERVICE":
            self.status = "OCCUPIED"
        acknowledge_service(self.table.id)


class OccupancyTracker:
    def __init__(self, debounce_sec: float = 4.0, menu_alert_sec: int = 120):
        self.debounce_sec = debounce_sec
        self.menu_alert_sec = menu_alert_sec
        self.states: Dict[str, TableState] = {}
        self._sync_tables()

    def set_debounce(self, debounce_sec: float):
        self.debounce_sec = debounce_sec
        for s in self.states.values():
            s.debounce_sec = debounce_sec

    def set_menu_alert_seconds(self, menu_alert_sec: int):
        self.menu_alert_sec = menu_alert_sec
        for s in self.states.values():
            s.menu_alert_sec = menu_alert_sec

    def _sync_tables(self):
        all_tables = tables_manager.get_all()
        current_ids = set()
        for t in all_tables:
            current_ids.add(t.id)
            if t.id not in self.states:
                self.states[t.id] = TableState(t, self.debounce_sec, self.menu_alert_sec)
            else:
                self.states[t.id].table = t

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
        """
        self._sync_tables()
        now = time.time()

        person_anchors = []
        for (x1, y1, x2, y2) in person_boxes:
            bx = ((x1 + x2) / 2.0) / max(1.0, float(frame_width))
            # 88% down the box keeps anchor on the seated diner's lower body without overshooting into forward tables
            by = (y1 + (y2 - y1) * 0.88) / max(1.0, float(frame_height))
            person_anchors.append((bx, by))

        table_statuses = []
        emitted_events = []

        for table_id, state in self.states.items():
            poly_norm = state.table.polygon
            t_channel = getattr(state.table, 'camera_channel', 1)

            if active_channel is not None and t_channel != active_channel:
                count = state.raw_headcount
                ring_count = state.raw_ring_count
            else:
                halo_poly = expand_polygon(poly_norm, scale=1.40)
                count = 0
                ring_count = 0
                for anchor in person_anchors:
                    if is_point_in_polygon(anchor, poly_norm):
                        count += 1
                    elif is_point_in_polygon(anchor, halo_poly):
                        ring_count += 1

            evt_name, evt_data = state.update_raw(count, now, ring_count=ring_count)
            if evt_name and evt_data:
                emitted_events.append({"event": evt_name, "data": evt_data})

            elapsed = state.get_elapsed_seconds(now)
            elapsed_m = elapsed // 60
            elapsed_s = elapsed % 60
            elapsed_str = f"{elapsed_m:02d}:{elapsed_s:02d}"

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
                "menu_alert_seconds": state.menu_alert_sec,
                "status": state.status,
                "menu_served": state.menu_served,
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

    def update_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        active_channel: Optional[int] = None,
        strict_conf: float = 0.38,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Update state machine from pinpoint pose/box detections with confidence hysteresis.
        - strict detections (conf >= strict_conf) start sessions & detect waiters.
        - hold detections (conf >= 0.20) keep already-occupied tables locked even when leaning/shifting.
        """
        self._sync_tables()
        now = time.time()

        strict_dets = []
        hold_anchors = []
        fw = max(1.0, float(frame_width))
        fh = max(1.0, float(frame_height))

        for det in detections:
            ax, ay = det["anchor"]
            norm_pt = (ax / fw, ay / fh)
            hold_anchors.append(norm_pt)
            if det.get("conf", 0.0) >= strict_conf:
                kpts = det.get("keypoints") or {}
                wrists_norm = [
                    (kpts[wk][0] / fw, kpts[wk][1] / fh)
                    for wk in ("l_wrist", "r_wrist")
                    if wk in kpts
                ]
                strict_dets.append((norm_pt, wrists_norm))

        table_statuses = []
        emitted_events = []

        for table_id, state in self.states.items():
            poly_norm = state.table.polygon
            t_channel = getattr(state.table, 'camera_channel', 1)

            if active_channel is not None and t_channel != active_channel:
                strict_count = state.raw_headcount
                hold_count = state.raw_headcount
                ring_count = state.raw_ring_count
                hand_reach_count = 0
            else:
                # Tight 1.22x edge zone avoids clearing when waiters walk past in the main aisle
                halo_poly = expand_polygon(poly_norm, scale=1.22)
                hold_poly = expand_polygon(poly_norm, scale=1.10)
                strict_count = 0
                ring_count = 0
                hand_reach_count = 0
                for anchor, wrists_norm in strict_dets:
                    if is_point_in_polygon(anchor, poly_norm):
                        strict_count += 1
                    elif is_point_in_polygon(anchor, halo_poly):
                        ring_count += 1
                        if any(is_point_in_polygon(w_pt, poly_norm) for w_pt in wrists_norm):
                            hand_reach_count += 1

                hold_count = 0
                for anchor in hold_anchors:
                    if is_point_in_polygon(anchor, hold_poly):
                        hold_count += 1

            evt_name, evt_data = state.update_raw(
                strict_count,
                now,
                ring_count=ring_count,
                hold_count=hold_count,
                hand_reach_count=hand_reach_count,
            )
            if evt_name and evt_data:
                emitted_events.append({"event": evt_name, "data": evt_data})

            elapsed = state.get_elapsed_seconds(now)
            elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
            poly_px = [[int(pt[0] * frame_width), int(pt[1] * frame_height)] for pt in poly_norm]

            table_statuses.append({
                "id": state.table.id,
                "name": state.table.name,
                "camera_channel": t_channel,
                "capacity": state.table.capacity,
                "service_threshold_sec": state.table.service_threshold_sec,
                "menu_alert_seconds": state.menu_alert_sec,
                "status": state.status,
                "menu_served": state.menu_served,
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
