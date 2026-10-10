"""SQLite helpers for the BreathCam database (schema in sql/schema.sql)."""

import sqlite3
from pathlib import Path

SCHEMA = Path(__file__).resolve().parent.parent / "sql" / "schema.sql"


def connect(path):
    """Open the database at `path` (":memory:" works) with foreign keys enforced.

    Inputs:
        path: database file path.
    Returns:
        sqlite3.Connection.
    """
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init(conn):
    """Create all tables from sql/schema.sql. Fails if they already exist.

    Inputs:
        conn: sqlite3.Connection.
    Returns:
        None.
    """
    conn.executescript(SCHEMA.read_text())


def add_subject(conn, subject_id, consent_date, notes=None):
    """Insert a participant (pseudonymous ID only, never a name).

    Inputs:
        conn: sqlite3.Connection.
        subject_id: e.g. "S01".
        consent_date: ISO date string.
        notes: optional free text.
    Returns:
        None.
    """
    conn.execute("INSERT INTO subjects VALUES (?, ?, ?)", (subject_id, consent_date, notes))


def add_session(conn, subject_id, video_path, **fields):
    """Insert one recording.

    Inputs:
        conn: sqlite3.Connection.
        subject_id: an existing subject.
        video_path: path of the clip.
        **fields: any of paced_bpm, lighting, clothing, distance_m,
            camera_mount, sensor_path, notes.
    Returns:
        int, the new session_id.
    """
    cols = ["subject_id", "video_path"] + list(fields)
    cur = conn.execute(
        f"INSERT INTO sessions ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [subject_id, str(video_path), *fields.values()],
    )
    return cur.lastrowid


def add_reference(conn, session_id, t_start, t_end, ref_bpm, ref_source):
    """Insert a reference rate valid for [t_start, t_end) seconds of a session.

    Inputs:
        conn: sqlite3.Connection.
        session_id: an existing session.
        t_start, t_end: seconds into the clip.
        ref_bpm: reference breaths/min.
        ref_source: "metronome" or "accelerometer".
    Returns:
        None.
    """
    conn.execute("INSERT INTO reference_readings VALUES (?, ?, ?, ?, ?)",
                 (session_id, t_start, t_end, ref_bpm, ref_source))


def add_predictions(conn, session_id, method, window_starts, est_bpm, quality):
    """Insert one prediction per window for one method.

    Inputs:
        conn: sqlite3.Connection.
        session_id: an existing session.
        method: "welch", "peaks", "framediff", ...
        window_starts: window start times (s).
        est_bpm: estimated rates; NaN or None is stored as NULL (rejected).
        quality: quality scores, same length.
    Returns:
        None.
    """
    rows = [(session_id, float(t), method, None if e is None or e != e else float(e), float(q))
            for t, e, q in zip(window_starts, est_bpm, quality)]
    conn.executemany("INSERT INTO predictions VALUES (?, ?, ?, ?, ?)", rows)
