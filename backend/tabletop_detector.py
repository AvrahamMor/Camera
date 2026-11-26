import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE_PATH = BASE_DIR / "config" / "table_baselines.npz"

PATCH_SIZE = 48
CORE_SCALE = 0.58


def _polygon_signature(polygon: List[List[float]]) -> str:
    """Return a short hash identifying a table polygon's calibrated coordinates."""
    rounded = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in polygon]
    raw = json.dumps(rounded, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def _shrink_polygon(polygon: List[List[float]], scale: float = CORE_SCALE) -> List[List[float]]:
    """Shrink normalized polygon inward around centroid to isolate the central tabletop."""
    if len(polygon) < 3:
        return polygon
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    return [
        [max(0.0, min(1.0, cx + (p[0] - cx) * scale)), max(0.0, min(1.0, cy + (p[1] - cy) * scale))]
        for p in polygon
    ]


class TabletopDetector:
    """
    Learns per-table empty tabletop baselines while tables are FREE and detects
    when food, plates, drinks, or open menus occupy the unoccluded tabletop center.
    """

    def __init__(self, baseline_path: Optional[Path] = None):
        self.baseline_path = Path(baseline_path) if baseline_path else DEFAULT_BASELINE_PATH
        self._lock = threading.Lock()
        self._baselines: Dict[str, Tuple[str, np.ndarray]] = {}
        self._dirty = False
        self._last_save_time = 0.0
        self._load_baselines()

    def _load_baselines(self) -> None:
        if not self.baseline_path.exists():
            return
        try:
            with np.load(str(self.baseline_path), allow_pickle=False) as data:
                keys = list(data.files)
                for k in keys:
                    if not k.startswith("patch__"):
                        continue
                    tid = k[len("patch__"):]
                    sig_key = f"sig__{tid}"
                    patch = data[k]
                    sig = str(data[sig_key].item()) if sig_key in data else ""
                    if patch.shape == (PATCH_SIZE, PATCH_SIZE):
                        self._baselines[tid] = (sig, patch.astype(np.uint8))
        except Exception as e:
            print(f"[TabletopDetector] Failed loading baselines: {e}")

    def _save_baselines(self, force: bool = False) -> None:
        now = time.time()
        if not self._dirty:
            return
        if not force and (now - self._last_save_time) < 20.0:
            return
        try:
            self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
            payload: Dict[str, Any] = {}
            for tid, (sig, patch) in self._baselines.items():
                payload[f"patch__{tid}"] = patch
                payload[f"sig__{tid}"] = np.array(sig)
            tmp_path = self.baseline_path.with_name(self.baseline_path.name + ".tmp.npz")
            with open(tmp_path, "wb") as f:
                np.savez_compressed(f, **payload)
            tmp_path.replace(self.baseline_path)
            self._dirty = False
            self._last_save_time = now
        except Exception as e:
            print(f"[TabletopDetector] Failed saving baselines: {e}")

    def _extract_core_patch(
        self,
        frame_gray: np.ndarray,
        poly_norm: List[List[float]],
        person_boxes: List[Tuple[float, float, float, float]],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract canonical 48x48 tabletop center patch and boolean unoccluded mask.
        Person bounding boxes are masked out so leaning guests do not register as dishes.
        """
        if len(poly_norm) < 3 or frame_gray is None:
            return None, None

        fh, fw = frame_gray.shape[:2]
        core_norm = _shrink_polygon(poly_norm, CORE_SCALE)
        pts_px = np.array([[p[0] * fw, p[1] * fh] for p in core_norm], dtype=np.float32)

        x_min = max(0, int(np.floor(pts_px[:, 0].min())) - 2)
        y_min = max(0, int(np.floor(pts_px[:, 1].min())) - 2)
        x_max = min(fw, int(np.ceil(pts_px[:, 0].max())) + 2)
        y_max = min(fh, int(np.ceil(pts_px[:, 1].max())) + 2)
        if (x_max - x_min) < 12 or (y_max - y_min) < 12:
            return None, None

        roi_gray = cv2.GaussianBlur(frame_gray[y_min:y_max, x_min:x_max], (3, 3), 0)
        rh, rw = roi_gray.shape[:2]
        roi_vis = np.full((rh, rw), 255, dtype=np.uint8)
        for bx1, by1, bx2, by2 in person_boxes:
            ix1 = max(0, int(bx1) - x_min)
            iy1 = max(0, int(by1) - y_min)
            ix2 = min(rw, int(bx2) - x_min)
            iy2 = min(rh, int(by2) - y_min)
            if ix2 > ix1 and iy2 > iy1:
                roi_vis[iy1:iy2, ix1:ix2] = 0

        local_pts = pts_px - np.array([[x_min, y_min]], dtype=np.float32)
        if len(local_pts) == 4:
            dst_pts = np.array(
                [[0, 0], [PATCH_SIZE - 1, 0], [PATCH_SIZE - 1, PATCH_SIZE - 1], [0, PATCH_SIZE - 1]],
                dtype=np.float32,
            )
            mat = cv2.getPerspectiveTransform(local_pts, dst_pts)
            patch_gray = cv2.warpPerspective(roi_gray, mat, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_LINEAR)
            patch_mask = cv2.warpPerspective(roi_vis, mat, (PATCH_SIZE, PATCH_SIZE), flags=cv2.INTER_NEAREST) > 128
            return patch_gray, patch_mask

        poly_mask = np.zeros((rh, rw), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [local_pts.astype(np.int32)], 255)
        combined_vis = cv2.bitwise_and(roi_vis, poly_mask)
        patch_gray = cv2.resize(roi_gray, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_LINEAR)
        patch_mask = cv2.resize(combined_vis, (PATCH_SIZE, PATCH_SIZE), interpolation=cv2.INTER_NEAREST) > 128
        return patch_gray, patch_mask

    def evaluate_channel(
        self,
        frame: Optional[np.ndarray],
        active_channel: Optional[int],
        states: Dict[str, Any],
        person_boxes: List[Tuple[float, float, float, float]],
        now: float,
    ) -> Dict[str, bool]:
        """
        Update clean baselines for FREE tables and return {table_id: has_food_or_menu}
        for unserved OCCUPIED / NEEDS_SERVICE tables on active_channel.
        """
        if frame is None or frame.size == 0:
            return {}

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        results: Dict[str, bool] = {}

        with self._lock:
            for tid, state in states.items():
                t_ch = getattr(state.table, "camera_channel", 1)
                if active_channel is not None and t_ch != active_channel:
                    continue

                poly_norm = state.table.polygon
                sig = _polygon_signature(poly_norm)

                existing = self._baselines.get(tid)
                if existing and existing[0] and existing[0] != sig:
                    del self._baselines[tid]
                    existing = None
                    self._dirty = True

                # Capture or gradually track baseline ONLY when table and its halo are cleanly unoccupied
                min_free_sec = 45.0 if getattr(state, "ever_occupied", False) else 8.0
                if (
                    state.status == "FREE"
                    and state.raw_headcount == 0
                    and state.raw_ring_count == 0
                    and (now - state.last_state_change) >= min_free_sec
                ):
                    patch, vis_mask = self._extract_core_patch(frame_gray, poly_norm, person_boxes)
                    if patch is not None and vis_mask is not None and float(np.mean(vis_mask)) >= 0.85:
                        if existing is None:
                            self._baselines[tid] = (sig, patch.copy())
                            self._dirty = True
                        else:
                            base = existing[1].astype(np.float32)
                            cur = patch.astype(np.float32)
                            shift = float(np.median(cur[vis_mask]) - np.median(base[vis_mask]))
                            diff_frac = float(np.mean(np.abs(cur[vis_mask] - (base[vis_mask] + shift)) > 26.0))
                            if diff_frac < 0.18:
                                updated = np.clip(0.90 * base + 0.10 * cur, 0, 255).astype(np.uint8)
                                self._baselines[tid] = (sig, updated)
                                self._dirty = True
                    continue

                if state.status not in ("OCCUPIED", "NEEDS_SERVICE") or state.menu_served:
                    continue
                if existing is None:
                    continue

                patch, vis_mask = self._extract_core_patch(frame_gray, poly_norm, person_boxes)
                if patch is None or vis_mask is None:
                    continue
                # Need at least 25% of inner tabletop visible outside diner silhouettes
                if int(np.sum(vis_mask)) < int(0.25 * PATCH_SIZE * PATCH_SIZE):
                    continue

                base_vals = existing[1].astype(np.float32)[vis_mask]
                cur_vals = patch.astype(np.float32)[vis_mask]
                shift = float(np.median(cur_vals) - np.median(base_vals))
                diff_frac = float(np.mean(np.abs(cur_vals - (base_vals + shift)) > 26.0))
                results[tid] = diff_frac >= 0.22

            if self._dirty:
                self._save_baselines()

        return results


tabletop_detector = TabletopDetector()
