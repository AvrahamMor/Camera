import json
import threading
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from backend.tables_manager import tables_manager, TableModel
from backend.database import record_seating, record_vacated, record_service_alert, acknowledge_service
from backend.telegram_alerts import telegram_alerts
from backend.firebase_sync import firebase_manager
from backend.tabletop_detector import TabletopDetector, tabletop_detector

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_PATH = BASE_DIR / "config" / "live_state_cache.json"


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
        self.ever_occupied: bool = False

        # Waiter approach & tabletop item timers (filters quick walk-bys / aisle glances)
        self.waiter_ring_start: Optional[float] = None
        self.hand_reach_start: Optional[float] = None
        self.inner_spike_start: Optional[float] = None
        self.tabletop_items_start: Optional[float] = None

        # Debounce buffer: list of (timestamp, raw_count)
        self.history: List[Tuple[float, int]] = []

    def update_raw(
        self,
        current_raw_count: int,
        now: float,
        ring_count: int = 0,
        hold_count: Optional[int] = None,
        hand_reach_count: int = 0,
        tabletop_items: bool = False,
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
                self.ever_occupied = True
                self.waiter_ring_start = None
                self.hand_reach_start = None
                self.inner_spike_start = None
                self.tabletop_items_start = None

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
                    self.hand_reach_start = None
                    self.inner_spike_start = None
                    self.tabletop_items_start = None

                    record_vacated(self.table.id)
                    event_name = "VACATED"
                    event_data = {
                        "table_id": self.table.id,
                        "table_name": self.table.name,
                        "duration_sec": duration_sec
                    }

        # --- Waiter / Menu Approach Detection while occupied ---
        if self.status in ("OCCUPIED", "NEEDS_SERVICE"):
            required_dwell = 4.0
            required_reach = 1.5
            served_now = False

            # Signal A: Waiter beside table reaches wrist/hand onto table surface
            if hand_reach_count >= 1:
                if self.hand_reach_start is None:
                    self.hand_reach_start = now
                elif (now - self.hand_reach_start) >= required_reach:
                    served_now = True
            else:
                self.hand_reach_start = None

            # Signal B: Waiter stands beside table (1.45x edge ring) for >= 4.0s
            if ring_count >= 1:
                if self.waiter_ring_start is None:
                    self.waiter_ring_start = now
                elif (now - self.waiter_ring_start) >= required_dwell:
                    served_now = True
            else:
                self.waiter_ring_start = None

            # Signal D: Persistent plates, food, drinks, or open menus detected on unoccluded tabletop center
            if tabletop_items and not self.menu_served:
                if self.tabletop_items_start is None:
                    self.tabletop_items_start = now
                elif (now - self.tabletop_items_start) >= 8.0:
                    served_now = True
            else:
                self.tabletop_items_start = None

            if served_now:
                evt_name, evt_data = self._mark_waiter_arrived(now)
                event_name = event_name or evt_name
                event_data = event_data or evt_data

            # Signal C: Extra person stepped inside polygon for 4s..20s then left
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
                    if 4.0 <= spike_dur < 20.0:
                        evt_name, evt_data = self._mark_waiter_arrived(now)
                        event_name = event_name or evt_name
                        event_data = event_data or evt_data
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
                    self.table.name,
                    elapsed_display,
                    self.confirmed_headcount,
                    table_id=self.table.id,
                    channel=getattr(self.table, "camera_channel", 1),
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
                    self.table.name,
                    elapsed_display,
                    self.confirmed_headcount,
                    table_id=self.table.id,
                    channel=getattr(self.table, "camera_channel", 1),
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
        self.service_alert_fired = False
        self.last_waiter_visit_time = time.time()
        if self.status == "NEEDS_SERVICE":
            self.status = "OCCUPIED"
        acknowledge_service(self.table.id)

    def _mark_waiter_arrived(self, now: float) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Record a confirmed waiter arrival and emit follow-up updates if an alert was active."""
        self.last_waiter_visit_time = now
        was_alerting = self.status == "NEEDS_SERVICE" or self.service_alert_fired
        first_menu = not self.menu_served
        if not first_menu and not was_alerting:
            return None, None

        self.menu_served = True
        if was_alerting:
            self.status = "OCCUPIED"
            self.service_alert_fired = False
            self.service_acknowledged = True
            acknowledge_service(self.table.id)

            elapsed = (now - self.seated_time) if self.seated_time else 0
            elapsed_display = f"{int(elapsed // 60)}:{int(elapsed % 60):02d}"
            telegram_alerts.send_waiter_arrived_update(
                self.table.name,
                elapsed_display,
                table_id=self.table.id,
            )
            firebase_manager.log_event({
                "table_id": self.table.id,
                "table_name": self.table.name,
                "event_type": "WAITER_ARRIVED",
                "elapsed_sec": int(elapsed),
                "headcount": self.confirmed_headcount,
            })

        return "MENU_SERVED", {
            "table_id": self.table.id,
            "table_name": self.table.name,
            "headcount": self.confirmed_headcount,
            "after_alert": was_alerting,
        }


class OccupancyTracker:
    def __init__(
        self,
        debounce_sec: float = 4.0,
        menu_alert_sec: int = 120,
        cache_path: Optional[Path] = DEFAULT_CACHE_PATH,
        detector: Optional[TabletopDetector] = None,
    ):
        self.debounce_sec = debounce_sec
        self.menu_alert_sec = menu_alert_sec
        self.cache_path = Path(cache_path) if cache_path else None
        self.tabletop_detector = detector if detector is not None else tabletop_detector
        self._lock = threading.RLock()
        self.states: Dict[str, TableState] = {}
        self._last_cache_save = 0.0
        self._sync_tables()
        self._load_live_cache()

    def _load_live_cache(self) -> None:
        if not self.cache_path or not self.cache_path.exists():
            return
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            now = time.time()
            saved_at = float(data.get("saved_at", 0.0))
            if not (0 <= (now - saved_at) <= 600.0):
                return
            restored = 0
            for tid, info in (data.get("tables") or {}).items():
                st = self.states.get(tid)
                if not st or info.get("status") not in ("OCCUPIED", "NEEDS_SERVICE"):
                    continue
                st.status = info["status"]
                st.confirmed_headcount = max(1, int(info.get("confirmed_headcount", 1)))
                st.seated_time = info.get("seated_time") or saved_at
                st.last_state_change = info.get("last_state_change") or st.seated_time
                st.last_seated_seen_time = now
                st.last_waiter_visit_time = info.get("last_waiter_visit_time")
                st.menu_served = bool(info.get("menu_served", False))
                st.no_menu_alert_fired = bool(info.get("no_menu_alert_fired", False))
                st.service_alert_fired = bool(info.get("service_alert_fired", False))
                st.service_acknowledged = bool(info.get("service_acknowledged", False))
                st.ever_occupied = True
                restored += 1
            if restored:
                print(f"[OccupancyTracker] Restored {restored} active table(s) from cache.")
        except Exception as e:
            print(f"[OccupancyTracker] Cache restore skipped: {e}")

    def _save_live_cache(self, now: Optional[float] = None, force: bool = False) -> None:
        if not self.cache_path:
            return
        cur = now if now is not None else time.time()
        if not force and (cur - self._last_cache_save) < 10.0:
            return
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            active = {}
            for tid, st in self.states.items():
                if st.status in ("OCCUPIED", "NEEDS_SERVICE"):
                    active[tid] = {
                        "status": st.status,
                        "confirmed_headcount": st.confirmed_headcount,
                        "seated_time": st.seated_time,
                        "last_state_change": st.last_state_change,
                        "last_waiter_visit_time": st.last_waiter_visit_time,
                        "menu_served": st.menu_served,
                        "no_menu_alert_fired": st.no_menu_alert_fired,
                        "service_alert_fired": st.service_alert_fired,
                        "service_acknowledged": st.service_acknowledged,
                    }
            payload = {"saved_at": cur, "tables": active}
            tmp_path = self.cache_path.with_name(self.cache_path.name + ".tmp")
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            tmp_path.replace(self.cache_path)
            self._last_cache_save = cur
        except Exception as e:
            print(f"[OccupancyTracker] Cache save error: {e}")

    def set_debounce(self, debounce_sec: float):
        with self._lock:
            self.debounce_sec = debounce_sec
            for s in self.states.values():
                s.debounce_sec = debounce_sec

    def set_menu_alert_seconds(self, menu_alert_sec: int):
        with self._lock:
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

    def _serialize_state(self, state: TableState, now: float, frame_width: int = 1280, frame_height: int = 720) -> Dict[str, Any]:
        poly_norm = state.table.polygon
        t_channel = getattr(state.table, 'camera_channel', 1)
        elapsed = state.get_elapsed_seconds(now)
        elapsed_str = f"{elapsed // 60:02d}:{elapsed % 60:02d}"
        poly_px = [[int(pt[0] * frame_width), int(pt[1] * frame_height)] for pt in poly_norm]
        return {
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
            "polygon_px": poly_px,
        }

    def get_all_statuses(self, frame_width: int = 1280, frame_height: int = 720) -> List[Dict[str, Any]]:
        with self._lock:
            self._sync_tables()
            now = time.time()
            return [self._serialize_state(state, now, frame_width, frame_height) for state in self.states.values()]

    def update(
        self,
        person_boxes: List[Tuple[float, float, float, float]],
        frame_width: int,
        frame_height: int,
        active_channel: Optional[int] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Legacy box wrapper calling update_detections."""
        detections = []
        for (x1, y1, x2, y2) in person_boxes:
            detections.append({
                "box": (x1, y1, x2, y2),
                "conf": 0.9,
                "anchor": ((x1 + x2) / 2.0, y1 + (y2 - y1) * 0.82),
                "keypoints": {},
            })
        return self.update_detections(detections, frame_width, frame_height, active_channel=active_channel, strict_conf=0.38)

    def update_detections(
        self,
        detections: List[Dict[str, Any]],
        frame_width: int,
        frame_height: int,
        active_channel: Optional[int] = None,
        strict_conf: float = 0.38,
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Update state machine from pinpoint pose/box detections with confidence hysteresis.
        Only mutates tables matching active_channel (so multi-camera round-robin never double-ticks other channels).
        """
        with self._lock:
            self._sync_tables()
            now = time.time()

            person_boxes = [det["box"] for det in detections if "box" in det]
            tabletop_flags = (
                self.tabletop_detector.evaluate_channel(frame, active_channel, self.states, person_boxes, now)
                if (frame is not None and self.tabletop_detector is not None)
                else {}
            )

            strict_anchors = []
            waiter_dets = []
            hold_anchors = []
            waiter_conf = max(0.28, strict_conf - 0.08)
            fw = max(1.0, float(frame_width))
            fh = max(1.0, float(frame_height))

            for det in detections:
                ax, ay = det["anchor"]
                norm_pt = (ax / fw, ay / fh)
                hold_anchors.append(norm_pt)
                conf = det.get("conf", 0.0)
                if conf >= strict_conf:
                    strict_anchors.append(norm_pt)
                if conf >= waiter_conf:
                    kpts = det.get("keypoints") or {}
                    ring_pts = [norm_pt] + [
                        (kpts[sk][0] / fw, kpts[sk][1] / fh)
                        for sk in ("l_sh", "r_sh")
                        if sk in kpts
                    ]
                    wrists_norm = [
                        (kpts[wk][0] / fw, kpts[wk][1] / fh)
                        for wk in ("l_wrist", "r_wrist")
                        if wk in kpts
                    ]
                    waiter_dets.append((norm_pt, ring_pts, wrists_norm))

            table_statuses = []
            emitted_events = []

            for table_id, state in self.states.items():
                poly_norm = state.table.polygon
                t_channel = getattr(state.table, 'camera_channel', 1)

                if active_channel is None or t_channel == active_channel:
                    # 1.45x halo catches waiters standing beside the table in the aisle (including backs turned)
                    halo_poly = expand_polygon(poly_norm, scale=1.45)
                    hold_poly = expand_polygon(poly_norm, scale=1.10)
                    other_polys = [
                        st.table.polygon
                        for tid, st in self.states.items()
                        if tid != table_id
                        and getattr(st.table, 'camera_channel', 1) == t_channel
                        and len(st.table.polygon) >= 3
                    ]

                    strict_count = sum(1 for pt in strict_anchors if is_point_in_polygon(pt, poly_norm))
                    ring_count = 0
                    hand_reach_count = 0
                    for anchor, ring_pts, wrists_norm in waiter_dets:
                        if is_point_in_polygon(anchor, poly_norm):
                            continue
                        if any(is_point_in_polygon(anchor, op) for op in other_polys):
                            continue
                        if any(is_point_in_polygon(pt, halo_poly) for pt in ring_pts):
                            ring_count += 1
                            if any(is_point_in_polygon(w_pt, poly_norm) for w_pt in wrists_norm):
                                hand_reach_count += 1

                    hold_count = sum(1 for anchor in hold_anchors if is_point_in_polygon(anchor, hold_poly))

                    evt_name, evt_data = state.update_raw(
                        strict_count,
                        now,
                        ring_count=ring_count,
                        hold_count=hold_count,
                        hand_reach_count=hand_reach_count,
                        tabletop_items=tabletop_flags.get(table_id, False),
                    )
                    if evt_name and evt_data:
                        emitted_events.append({"event": evt_name, "data": evt_data})

                table_statuses.append(self._serialize_state(state, now, frame_width, frame_height))

            self._save_live_cache(now=now, force=bool(emitted_events))
            return table_statuses, emitted_events

    def acknowledge_service(self, table_id: str) -> bool:
        with self._lock:
            if table_id in self.states:
                self.states[table_id].acknowledge_service()
                self._save_live_cache(force=True)
                return True
            return False
