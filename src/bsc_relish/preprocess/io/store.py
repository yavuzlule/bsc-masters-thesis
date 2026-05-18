from pathlib import Path
from datetime import datetime
import json
import pandas as pd


class DatasetStore:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)

    def save(self, df: pd.DataFrame, name: str, metadata: dict = None):

        run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = self.base_dir / name / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = out_dir / "dataset.parquet"
        meta_path = out_dir / "metadata.json"

        df.to_parquet(parquet_path, index=False)

        metadata = metadata or {}
        metadata.update({
            "n_rows": len(df),
            "n_cols": len(df.columns),
            "columns": list(df.columns),
        })

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return str(parquet_path)