# data/

Everything in this folder except this file is gitignored.

Planned contents (created in Phases 1 and 3):

- `bidmc/`: BIDMC PPG and Respiration Dataset from PhysioNet (Phase 1)
- `raw/`: recorded clips and accelerometer CSVs, named by pseudonymous ID (`S01_...`), never by participant name
- `breathcam.db`: SQLite database (see `sql/schema.sql`)
- `consent_log.*` and `subject_key.*`: signed consents and the ID-to-name mapping. Keep these off the repo and off cloud notebooks.
