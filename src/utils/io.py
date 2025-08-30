import json, yaml, pathlib, pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
RAW.mkdir(parents=True, exist_ok=True)
PROC.mkdir(parents=True, exist_ok=True)

def save_json(data, fname):
    f = RAW / fname
    f.write_text(json.dumps(data, indent=2))

def save_csv(df: pd.DataFrame, fname: str):
    df.to_csv(RAW / fname, index=False)

def load_yaml(path="config/queries.yaml"):
    with open(ROOT / path) as fh:
        return yaml.safe_load(fh)
