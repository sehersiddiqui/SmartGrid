import sqlite3, json
from pathlib import Path
from datetime import datetime, timezone

def log_experiment(metrics, db_path='artifacts/experiments.sqlite'):
    path=Path(db_path); path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(path) as con:
        con.execute('CREATE TABLE IF NOT EXISTS experiments (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, metrics_json TEXT NOT NULL)')
        con.execute('INSERT INTO experiments(created_at,metrics_json) VALUES (?,?)',(datetime.now(timezone.utc).isoformat(),json.dumps(metrics,default=float)))
        con.commit()

def read_experiments(db_path='artifacts/experiments.sqlite'):
    path=Path(db_path)
    if not path.exists(): return []
    with sqlite3.connect(path) as con: return con.execute('SELECT id,created_at,metrics_json FROM experiments ORDER BY id DESC').fetchall()
