from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

REQUIRED_COLUMNS = ("filename", "start", "stop", "label")


@dataclass
class AnnotationSet:
    df: pd.DataFrame
    source_path: Path

    @property
    def labels(self) -> list[str]:
        return sorted(self.df["label"].unique().tolist())

    def for_file(self, filename: str) -> "AnnotationSet":
        subset = self.df[self.df["filename"] == filename].reset_index(drop=True)
        return AnnotationSet(df=subset, source_path=self.source_path)

    def __len__(self) -> int:
        return len(self.df)


def _parse_time_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    parsed = pd.to_timedelta(series, errors="coerce")
    if parsed.isna().any():
        bad = series[parsed.isna()].tolist()
        raise ValueError(
            f"Failed to parse time values: {bad!r}. "
            "Expected format H:MM:SS.ffffff or a number of seconds."
        )
    return parsed.dt.total_seconds()


def load_annotations(tsv_path: str | Path) -> AnnotationSet:
    tsv_path = Path(tsv_path)
    df = pd.read_csv(tsv_path, sep="\t")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Failed to find required columns in {tsv_path}: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    df["start_s"] = _parse_time_column(df["start"])
    df["stop_s"] = _parse_time_column(df["stop"])

    if (df["stop_s"] < df["start_s"]).any():
        bad_rows = df[df["stop_s"] < df["start_s"]]
        raise ValueError(
            "Found annotations where 'stop' is before 'start':\n"
            f'{bad_rows[["filename", "start", "stop", "label"]]}'
        )

    df["duration"] = df["stop_s"] - df["start_s"]
    df = df.sort_values("start_s").reset_index(drop=True)

    return AnnotationSet(df=df, source_path=tsv_path)