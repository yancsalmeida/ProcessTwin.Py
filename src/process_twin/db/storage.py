import aiosqlite
from typing import List, Dict, Any, Optional
from process_twin.core.models import PlantTelemetry, AlarmEvent


class TelemetryDatabase:
    """Async SQLite storage for process telemetry time-series and alarm logs."""

    def __init__(self, db_path: str = "processtwin.db") -> None:
        self.db_path = db_path

    async def init_db(self) -> None:
        """Initializes relational schema with indexes for fast time-series retrieval."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS telemetry_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    tank_level_m REAL NOT NULL,
                    tank_volume_m3 REAL NOT NULL,
                    feed_flow_m3_h REAL NOT NULL,
                    discharge_flow_m3_h REAL NOT NULL,
                    reactor_temp_c REAL NOT NULL,
                    reactor_pressure_bar REAL NOT NULL,
                    cooling_valve_pct REAL NOT NULL,
                    feed_valve_open INTEGER NOT NULL,
                    interlock_tripped INTEGER NOT NULL,
                    status TEXT NOT NULL
                );
            """)

            await db.execute("""
                CREATE TABLE IF NOT EXISTS alarms_log (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    message TEXT NOT NULL,
                    acknowledged INTEGER NOT NULL
                );
            """)

            await db.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry_history(timestamp);")
            await db.commit()

    async def insert_telemetry(self, frame: PlantTelemetry) -> None:
        """Records a single telemetry snapshot into the time-series table."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO telemetry_history (
                    timestamp, tank_level_m, tank_volume_m3, feed_flow_m3_h,
                    discharge_flow_m3_h, reactor_temp_c, reactor_pressure_bar,
                    cooling_valve_pct, feed_valve_open, interlock_tripped, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (
                frame.timestamp.isoformat(),
                frame.tank_level_m,
                frame.tank_volume_m3,
                frame.feed_flow_m3_h,
                frame.discharge_flow_m3_h,
                frame.reactor_temp_c,
                frame.reactor_pressure_bar,
                frame.cooling_valve_pct,
                1 if frame.feed_valve_open else 0,
                1 if frame.interlock_tripped else 0,
                frame.status.value,
            ))
            await db.commit()

    async def insert_alarm(self, alarm: AlarmEvent) -> None:
        """Logs an annunciated alarm event."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO alarms_log (
                    id, timestamp, severity, tag, message, acknowledged
                ) VALUES (?, ?, ?, ?, ?, ?);
            """, (
                alarm.id,
                alarm.timestamp.isoformat(),
                alarm.severity.value,
                alarm.tag,
                alarm.message,
                1 if alarm.acknowledged else 0,
            ))
            await db.commit()

    async def get_recent_telemetry(self, limit: int = 60) -> List[Dict[str, Any]]:
        """Retrieves the latest N telemetry frames ordered chronologically."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM telemetry_history
                ORDER BY id DESC
                LIMIT ?;
            """, (limit,))
            rows = await cursor.fetchall()
            # Reverse so it's ascending chronological order
            return [dict(row) for row in reversed(rows)]

    async def get_recent_alarms(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieves recent alarm logs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT * FROM alarms_log
                ORDER BY timestamp DESC
                LIMIT ?;
            """, (limit,))
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
