-- Evaluation queries. Run with:
--   sqlite3 -header -column data/breathcam.db < sql/eval_queries.sql
-- A prediction is matched to the reference reading that covers its window start.
-- ref_source is always shown so metronome targets and accelerometer references
-- are never mixed in one number.

-- 1. MAE (breaths/min) by lighting, method and reference source
SELECT COALESCE(s.lighting, 'unknown') AS lighting, p.method, r.ref_source,
       COUNT(*) AS n_windows,
       ROUND(AVG(ABS(p.est_bpm - r.ref_bpm)), 2) AS mae
FROM predictions p
JOIN sessions s ON s.session_id = p.session_id
JOIN reference_readings r
  ON r.session_id = p.session_id
 AND p.window_start >= r.t_start AND p.window_start < r.t_end
WHERE p.est_bpm IS NOT NULL
GROUP BY lighting, p.method, r.ref_source;

-- 2. Share of windows within +/-2 bpm of the reference, by method and reference source
SELECT p.method, r.ref_source,
       COUNT(*) AS n_windows,
       ROUND(100.0 * AVG(CASE WHEN ABS(p.est_bpm - r.ref_bpm) <= 2 THEN 1 ELSE 0 END), 1) AS pct_within_2
FROM predictions p
JOIN reference_readings r
  ON r.session_id = p.session_id
 AND p.window_start >= r.t_start AND p.window_start < r.t_end
WHERE p.est_bpm IS NOT NULL
GROUP BY p.method, r.ref_source;

-- 3. Windows rejected by the quality gate, by condition (welch only, the gate is shared)
SELECT COALESCE(s.lighting, 'unknown') AS lighting,
       COALESCE(s.camera_mount, 'unknown') AS camera_mount,
       COUNT(*) AS windows,
       SUM(CASE WHEN p.est_bpm IS NULL THEN 1 ELSE 0 END) AS rejected
FROM predictions p
JOIN sessions s ON s.session_id = p.session_id
WHERE p.method = 'welch'
GROUP BY lighting, camera_mount;

-- 4. Per clip: windows, rejected, and MAE where a reference exists
SELECT s.session_id, s.video_path, p.method,
       COUNT(*) AS windows,
       SUM(CASE WHEN p.est_bpm IS NULL THEN 1 ELSE 0 END) AS rejected,
       ROUND(AVG(ABS(p.est_bpm - r.ref_bpm)), 2) AS mae_vs_ref
FROM predictions p
JOIN sessions s ON s.session_id = p.session_id
LEFT JOIN reference_readings r
  ON r.session_id = p.session_id
 AND p.window_start >= r.t_start AND p.window_start < r.t_end
GROUP BY s.session_id, p.method
ORDER BY s.session_id, p.method;
