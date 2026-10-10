-- BreathCam database schema (SQLite)

CREATE TABLE subjects (
  subject_id   TEXT PRIMARY KEY,          -- pseudonymous: S01, S02, ...
  consent_date TEXT NOT NULL,
  notes        TEXT
);

CREATE TABLE sessions (
  session_id   INTEGER PRIMARY KEY,
  subject_id   TEXT NOT NULL REFERENCES subjects(subject_id),
  paced_bpm    REAL,                      -- metronome rate; NULL if unpaced
  lighting     TEXT,                      -- bright | dim | backlit
  clothing     TEXT,                      -- fitted | loose | hoodie
  distance_m   REAL,
  camera_mount TEXT,                      -- tripod | handheld
  video_path   TEXT NOT NULL,
  sensor_path  TEXT,                      -- accelerometer CSV, if any
  notes        TEXT                       -- e.g. "breath-hold at ~17-32 s"
);

CREATE TABLE reference_readings (
  session_id INTEGER NOT NULL REFERENCES sessions(session_id),
  t_start    REAL NOT NULL,
  t_end      REAL NOT NULL,
  ref_bpm    REAL NOT NULL,
  ref_source TEXT NOT NULL                -- metronome (target, not truth) | accelerometer
);

CREATE TABLE predictions (
  session_id    INTEGER NOT NULL REFERENCES sessions(session_id),
  window_start  REAL NOT NULL,
  method        TEXT NOT NULL,            -- welch | peaks | framediff | cnn1d | r3d18
  est_bpm       REAL,                     -- NULL = rejected by the quality gate
  quality_score REAL
);
