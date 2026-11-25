import json
import re
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent.parent
TABLES_FILE = BASE_DIR / "config" / "tables_config.json"

class TableModel(BaseModel):
    id: str
    name: str
    camera_channel: int = Field(default=1, ge=1, le=8)
    capacity: int = Field(default=4, ge=1)
    service_threshold_sec: int = Field(default=180, ge=10)
    polygon: List[List[float]]  # List of [x, y] in normalized [0.0 .. 1.0] coordinates
    old_id: Optional[str] = None  # If renaming from an existing table ID

    @field_validator("id")
    @classmethod
    def validate_table_id(cls, v: str) -> str:
        clean = (v or "").strip()
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", clean):
            raise ValueError("Table ID must contain only English letters, numbers, '_' or '-' (max 64 chars)")
        return clean

class TablesManager:
    def __init__(self, config_path: Path = TABLES_FILE):
        self.config_path = config_path
        self._lock = threading.RLock()
        self.tables: Dict[str, TableModel] = {}
        self.load()

    def load(self) -> Dict[str, TableModel]:
        with self._lock:
            if not self.config_path.exists():
                self._create_default_config()
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tables = {item["id"]: TableModel(**item) for item in data}
            except Exception as e:
                print(f"[TablesManager] Failed to load {self.config_path}: {e}")
                self._create_default_config()
            return self.tables

    def save(self) -> None:
        with self._lock:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            data = [t.model_dump(exclude={'old_id'}) for t in self.tables.values()]
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def get_all(self) -> List[TableModel]:
        with self._lock:
            return list(self.tables.values())

    def get(self, table_id: str) -> Optional[TableModel]:
        with self._lock:
            return self.tables.get(table_id)

    def add_or_update(self, table: TableModel) -> TableModel:
        # Validate polygon bounds
        validated_poly = []
        for pt in table.polygon:
            x = max(0.0, min(1.0, float(pt[0])))
            y = max(0.0, min(1.0, float(pt[1])))
            validated_poly.append([round(x, 4), round(y, 4)])
        table.polygon = validated_poly

        with self._lock:
            # If renaming from an existing table ID, remove the previous ID entry
            if table.old_id and table.old_id != table.id and table.old_id in self.tables:
                del self.tables[table.old_id]
            self.tables[table.id] = table
        self.save()
        return table

    def delete(self, table_id: str) -> bool:
        deleted = False
        with self._lock:
            if table_id in self.tables:
                del self.tables[table_id]
                deleted = True
        if deleted:
            self.save()
            return True
        return False

    def reset_defaults(self) -> List[TableModel]:
        self._create_default_config()
        return self.get_all()

    def _create_default_config(self):
        defaults = [
            {
                "id": "table_01",
                "name": "Table 01 (Booth Left)",
                "capacity": 4,
                "service_threshold_sec": 180,
                "polygon": [[0.08, 0.45], [0.32, 0.45], [0.32, 0.85], [0.08, 0.85]]
            },
            {
                "id": "table_02",
                "name": "Table 02 (Center Round)",
                "capacity": 2,
                "service_threshold_sec": 180,
                "polygon": [[0.40, 0.35], [0.60, 0.35], [0.60, 0.68], [0.40, 0.68]]
            },
            {
                "id": "table_03",
                "name": "Table 03 (Booth Right)",
                "capacity": 4,
                "service_threshold_sec": 180,
                "polygon": [[0.68, 0.45], [0.92, 0.45], [0.92, 0.85], [0.68, 0.85]]
            },
            {
                "id": "table_04",
                "name": "Table 04 (Window High-Top)",
                "capacity": 2,
                "service_threshold_sec": 120,
                "polygon": [[0.15, 0.15], [0.35, 0.15], [0.35, 0.38], [0.15, 0.38]]
            },
            {
                "id": "table_05",
                "name": "Table 05 (VIP Lounge)",
                "capacity": 6,
                "service_threshold_sec": 240,
                "polygon": [[0.65, 0.15], [0.90, 0.15], [0.90, 0.38], [0.65, 0.38]]
            }
        ]
        self.tables = {item["id"]: TableModel(**item) for item in defaults}
        self.save()

# Global singleton
tables_manager = TablesManager()
