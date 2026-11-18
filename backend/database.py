import os
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, func
)
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "restaurant_analytics.db"

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)
SessionLocal = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()

class OccupancySession(Base):
    __tablename__ = "occupancy_sessions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    table_id = Column(String(50), index=True, nullable=False)
    table_name = Column(String(100), nullable=False)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    end_time = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, default=0)
    max_headcount = Column(Integer, default=1)
    service_alert_triggered = Column(Boolean, default=False)
    service_acknowledged = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True, index=True)

class TableEvent(Base):
    __tablename__ = "table_events"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    table_id = Column(String(50), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(50), nullable=False) # SEATED, VACATED, SERVICE_ALERT, SERVICE_ACK
    headcount = Column(Integer, default=0)
    details = Column(String(255), default="")

def init_db():
    Base.metadata.create_all(bind=engine)
    seed_mock_history_if_empty()

def record_seating(table_id: str, table_name: str, headcount: int = 1) -> int:
    session = SessionLocal()
    try:
        # Close any lingering active sessions for this table
        lingering = session.query(OccupancySession).filter(
            OccupancySession.table_id == table_id,
            OccupancySession.is_active == True
        ).all()
        now = datetime.utcnow()
        for s in lingering:
            s.is_active = False
            s.end_time = now
            s.duration_seconds = max(1, int((now - s.start_time).total_seconds()))

        # Create new session
        new_session = OccupancySession(
            table_id=table_id,
            table_name=table_name,
            start_time=now,
            max_headcount=max(1, headcount),
            is_active=True
        )
        session.add(new_session)

        # Log event
        event = TableEvent(
            table_id=table_id,
            timestamp=now,
            event_type="SEATED",
            headcount=headcount,
            details=f"התיישבו {headcount} סועדים ב-{table_name}"
        )
        session.add(event)
        session.commit()
        session.refresh(new_session)
        return new_session.id
    except Exception as e:
        session.rollback()
        print(f"[DB Error] record_seating: {e}")
        return 0
    finally:
        session.close()

def record_vacated(table_id: str) -> Optional[int]:
    session = SessionLocal()
    try:
        active = session.query(OccupancySession).filter(
            OccupancySession.table_id == table_id,
            OccupancySession.is_active == True
        ).first()

        now = datetime.utcnow()
        duration = 0
        table_name = active.table_name if active else table_id
        if active:
            active.is_active = False
            active.end_time = now
            duration = max(1, int((now - active.start_time).total_seconds()))
            active.duration_seconds = duration

        dur_min = duration // 60
        dur_sec = duration % 60
        dur_text = f"{dur_min} דק' ו-{dur_sec} שנ'" if dur_min > 0 else f"{dur_sec} שנ'"

        event = TableEvent(
            table_id=table_id,
            timestamp=now,
            event_type="VACATED",
            headcount=0,
            details=f"שולחן {table_name} פונה לאחר {dur_text}"
        )
        session.add(event)
        session.commit()
        return duration
    except Exception as e:
        session.rollback()
        print(f"[DB Error] record_vacated: {e}")
        return None
    finally:
        session.close()

def record_service_alert(table_id: str) -> None:
    session = SessionLocal()
    try:
        active = session.query(OccupancySession).filter(
            OccupancySession.table_id == table_id,
            OccupancySession.is_active == True
        ).first()
        if active and not active.service_alert_triggered:
            active.service_alert_triggered = True

        table_name = active.table_name if active else table_id
        event = TableEvent(
            table_id=table_id,
            timestamp=datetime.utcnow(),
            event_type="SERVICE_ALERT",
            headcount=active.max_headcount if active else 1,
            details=f"התראת שירות: {table_name} ממתין למלצר ללא מענה"
        )
        session.add(event)
        session.commit()
    except Exception as e:
        session.rollback()
        print(f"[DB Error] record_service_alert: {e}")
    finally:
        session.close()

def acknowledge_service(table_id: str) -> bool:
    session = SessionLocal()
    try:
        active = session.query(OccupancySession).filter(
            OccupancySession.table_id == table_id,
            OccupancySession.is_active == True
        ).first()
        if active:
            active.service_acknowledged = True

        table_name = active.table_name if active else table_id
        event = TableEvent(
            table_id=table_id,
            timestamp=datetime.utcnow(),
            event_type="SERVICE_ACK",
            details=f"אושר שירות עבור {table_name}"
        )
        session.add(event)
        session.commit()
        return True
    except Exception as e:
        session.rollback()
        print(f"[DB Error] acknowledge_service: {e}")
        return False
    finally:
        session.close()

def get_kpis() -> Dict[str, Any]:
    session = SessionLocal()
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Completed sessions today
        completed_today = session.query(OccupancySession).filter(
            OccupancySession.start_time >= today_start,
            OccupancySession.is_active == False
        ).all()

        total_sessions = len(completed_today)
        total_diners = sum(s.max_headcount for s in completed_today)
        avg_turnover_sec = (
            sum(s.duration_seconds for s in completed_today) / total_sessions
            if total_sessions > 0 else 0
        )

        # Service alerts today
        service_alerts_count = session.query(func.count(TableEvent.id)).filter(
            TableEvent.timestamp >= today_start,
            TableEvent.event_type == "SERVICE_ALERT"
        ).scalar() or 0

        return {
            "completed_sessions_today": total_sessions,
            "total_diners_today": total_diners,
            "avg_turnover_minutes": round(avg_turnover_sec / 60.0, 1),
            "service_alerts_today": service_alerts_count
        }
    except Exception as e:
        print(f"[DB Error] get_kpis: {e}")
        return {
            "completed_sessions_today": 0,
            "total_diners_today": 0,
            "avg_turnover_minutes": 0,
            "service_alerts_today": 0
        }
    finally:
        session.close()

def get_recent_events(limit: int = 40) -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        events = session.query(TableEvent).order_by(TableEvent.timestamp.desc()).limit(limit).all()
        return [
            {
                "id": e.id,
                "table_id": e.table_id,
                "timestamp": e.timestamp.strftime("%H:%M:%S"),
                "date": e.timestamp.strftime("%Y-%m-%d"),
                "event_type": e.event_type,
                "headcount": e.headcount,
                "details": e.details
            }
            for e in events
        ]
    except Exception as e:
        print(f"[DB Error] get_recent_events: {e}")
        return []
    finally:
        session.close()

def get_hourly_analytics() -> Dict[str, Any]:
    session = SessionLocal()
    try:
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        sessions = session.query(OccupancySession).filter(
            OccupancySession.start_time >= today_start
        ).all()

        hourly_counts = [0] * 24
        hourly_diners = [0] * 24

        for s in sessions:
            h = s.start_time.hour
            if 0 <= h < 24:
                hourly_counts[h] += 1
                hourly_diners[h] += s.max_headcount

        hours_labels = [f"{h:02d}:00" for h in range(24)]
        return {
            "labels": hours_labels,
            "turnovers": hourly_counts,
            "diners": hourly_diners
        }
    except Exception as e:
        print(f"[DB Error] get_hourly_analytics: {e}")
        return {"labels": [], "turnovers": [], "diners": []}
    finally:
        session.close()

def get_table_turnover_analytics() -> List[Dict[str, Any]]:
    session = SessionLocal()
    try:
        # Group by table_id
        results = session.query(
            OccupancySession.table_id,
            OccupancySession.table_name,
            func.count(OccupancySession.id).label("turnover_count"),
            func.avg(OccupancySession.duration_seconds).label("avg_duration"),
            func.sum(OccupancySession.max_headcount).label("total_guests")
        ).group_by(OccupancySession.table_id, OccupancySession.table_name).all()

        data = []
        for r in results:
            avg_m = round((r.avg_duration or 0) / 60.0, 1)
            data.append({
                "table_id": r.table_id,
                "table_name": r.table_name,
                "turnovers": r.turnover_count,
                "avg_turnover_minutes": avg_m,
                "total_guests": r.total_guests or 0
            })
        return data
    except Exception as e:
        print(f"[DB Error] get_table_turnover_analytics: {e}")
        return []
    finally:
        session.close()

def seed_mock_history_if_empty():
    """Seed baseline realistic session records so the dashboard has rich historical analytics from minute 1."""
    session = SessionLocal()
    try:
        count = session.query(func.count(OccupancySession.id)).scalar()
        if count > 0:
            return

        now = datetime.utcnow()
        mock_tables = [
            ("table_01", "שולחן 1 (תא שמאלי)", 4),
            ("table_02", "שולחן 2 (עגול מרכזי)", 2),
            ("table_03", "שולחן 3 (תא ימני)", 4),
            ("table_04", "שולחן 4 (בר חלון)", 2),
            ("table_05", "שולחן 5 (לאונג' VIP)", 6)
        ]

        import random
        # Create 18 historical sessions across the current day
        for i in range(18):
            t_id, t_name, cap = random.choice(mock_tables)
            hours_ago = random.uniform(1.0, 10.0)
            duration_minutes = random.randint(22, 58)
            start = now - timedelta(hours=hours_ago)
            end = start + timedelta(minutes=duration_minutes)
            guests = random.randint(1, cap)
            alert = random.random() < 0.25

            sess = OccupancySession(
                table_id=t_id,
                table_name=t_name,
                start_time=start,
                end_time=end,
                duration_seconds=duration_minutes * 60,
                max_headcount=guests,
                service_alert_triggered=alert,
                service_acknowledged=True,
                is_active=False
            )
            session.add(sess)

            # Add corresponding events
            session.add(TableEvent(
                table_id=t_id,
                timestamp=start,
                event_type="SEATED",
                headcount=guests,
                details=f"התיישבו {guests} סועדים ב-{t_name}"
            ))
            if alert:
                session.add(TableEvent(
                    table_id=t_id,
                    timestamp=start + timedelta(minutes=15),
                    event_type="SERVICE_ALERT",
                    headcount=guests,
                    details=f"התראת שירות: {t_name} ממתין למלצר"
                ))
            session.add(TableEvent(
                table_id=t_id,
                timestamp=end,
                event_type="VACATED",
                headcount=0,
                details=f"שולחן {t_name} פונה לאחר {duration_minutes} דק'"
            ))

        session.commit()
        print("[Database] Seeded initial sample analytics history.")
    except Exception as e:
        session.rollback()
        print(f"[DB Seed Error]: {e}")
    finally:
        session.close()
