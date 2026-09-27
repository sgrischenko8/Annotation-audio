from __future__ import annotations
import pandas as pd
from .annotations import AnnotationSet


def compute_stats(annotations: AnnotationSet) -> pd.DataFrame:
    df = annotations.df
    if df.empty:
        return pd.DataFrame(
            columns=["label", "count", "total_duration_s", "mean_duration_s",
                     "min_duration_s", "max_duration_s", "share_of_labeled_time"]
        )

    grouped = df.groupby("label")["duration"].agg(
        count="count",
        total_duration_s="sum",
        mean_duration_s="mean",
        min_duration_s="min",
        max_duration_s="max",
    ).reset_index().rename(columns={"label": "label"})

    total_time = grouped["total_duration_s"].sum()
    grouped["share_of_labeled_time"] = (
        grouped["total_duration_s"] / total_time if total_time > 0 else 0.0
    )

    grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    for col in ("total_duration_s", "mean_duration_s", "min_duration_s",
                "max_duration_s", "share_of_labeled_time"):
        grouped[col] = grouped[col].round(3)

    return grouped


def format_stats_report(annotations: AnnotationSet, stats_df: pd.DataFrame) -> str:
    df = annotations.df
    lines = []
    lines.append("=== Statistics ===")
    lines.append(f"Annotation file: {annotations.source_path}")
    if not df.empty:
        lines.append(f"Audio files: {', '.join(sorted(df['filename'].unique()))}")
        lines.append(f"Total number of events: {len(df)}")
        lines.append(f"Number of classes: {df['label'].nunique()}")
        lines.append(
            f"Time range of annotations: {df['start_s'].min():.3f}s – {df['stop_s'].max():.3f}s"
        )
        lines.append(f"Total duration of all events: {df['duration'].sum():.3f}s")
    lines.append("")
    lines.append("Statistics by class:")
    lines.append(stats_df.to_string(index=False))
    return "\n".join(lines)