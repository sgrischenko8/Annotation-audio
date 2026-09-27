from __future__ import annotations
from pathlib import Path
import matplotlib

matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from .annotations import AnnotationSet


_PALETTE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


def _label_colors(labels: list[str]) -> dict[str, str]:
    return {label: _PALETTE[i % len(_PALETTE)] for i, label in enumerate(sorted(labels))}


def plot_spectrogram_with_events(
    S_db: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    annotations: AnnotationSet,
    title: str = "Spectrogram with annotations",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 6),
    freq_scale: str = "linear",
) -> plt.Figure:
    df = annotations.df
    labels = sorted(df["label"].unique().tolist()) if not df.empty else []
    colors = _label_colors(labels)

    fig, ax = plt.subplots(figsize=figsize)

    extent = [times[0] if len(times) else 0, times[-1] if len(times) else 1,
              freqs[0], freqs[-1]]
    img = ax.imshow(
        S_db, origin="lower", aspect="auto", extent=extent,
        cmap="magma", interpolation="nearest",
    )
    ax.set_yscale(freq_scale if freq_scale in ("linear", "log") else "linear")
    if freq_scale == "log":
        ax.set_ylim(bottom=max(freqs[freqs > 0].min() if (freqs > 0).any() else 1, 1))

    fig.colorbar(img, ax=ax, format="%+2.0f dB", pad=0.01)

    freq_span = freqs[-1] - freqs[0]
    band_height = 0.12 * freq_span
    band_bottom = freqs[-1] - band_height

    for _, row in df.iterrows():
        rect = mpatches.Rectangle(
            (row["start_s"], band_bottom),
            row["duration"],
            band_height,
            facecolor=colors[row["label"]],
            edgecolor="white",
            linewidth=0.4,
            alpha=0.85,
        )
        ax.add_patch(rect)

    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency, Hz")
    ax.set_title(title)

    if labels:
        handles = [mpatches.Patch(color=colors[l], label=l) for l in labels]
        ax.legend(handles=handles, loc="upper right", framealpha=0.85, title="Event Classes")

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig


def plot_event_timeline(
    annotations: AnnotationSet,
    title: str = "Event activity over time",
    output_path: str | Path | None = None,
    figsize: tuple[float, float] = (14, 4),
) -> plt.Figure:
    df = annotations.df
    labels = sorted(df["label"].unique().tolist()) if not df.empty else []
    colors = _label_colors(labels)
    label_to_row = {label: i for i, label in enumerate(labels)}

    fig, ax = plt.subplots(figsize=figsize)

    for _, row in df.iterrows():
        y = label_to_row[row["label"]]
        ax.barh(
            y, width=row["duration"], left=row["start_s"], height=0.6,
            color=colors[row["label"]], edgecolor="black", linewidth=0.3,
        )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Event Class")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_ylim(-0.5, max(len(labels) - 0.5, 0.5))

    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=150)

    return fig