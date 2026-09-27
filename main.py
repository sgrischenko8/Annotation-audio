#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from src.annotations import load_annotations
from src.audio import compute_spectrogram, load_audio
from src.stats import compute_stats, format_stats_report
from src.visualize import plot_event_timeline, plot_spectrogram_with_events


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Visualization of an audio spectrogram with overlaid event annotations and calculation of class-based statistics."
    )
    p.add_argument("--audio", required=True, type=Path, help="Path to the audio file")
    p.add_argument("--tsv", required=True, type=Path, help="Path to the TSV annotation file")
    p.add_argument("--output-dir", type=Path, default=Path("output"),
                    help="Directory for saving plots and reports (default: 'output')")
    p.add_argument("--spectrogram-type", choices=["mel", "stft"], default="mel",
                    help="Spectrogram type: mel (default) or standard STFT")
    p.add_argument("--sr", type=int, default=None,
                    help="Target sampling rate (default: original)")
    p.add_argument("--n-fft", type=int, default=2048, help="FFT window size")
    p.add_argument("--hop-length", type=int, default=512, help="Window step (hop length)")
    p.add_argument("--no-show", action="store_true",
                    help="Do not display plots interactively (only save files)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading annotations: {args.tsv}")
    all_annotations = load_annotations(args.tsv)
    annotations = all_annotations.for_file(args.audio.name)
    if len(annotations) == 0:
        available = sorted(all_annotations.df["filename"].unique())
        raise SystemExit(
            f"No annotations found for '{args.audio.name}' in the TSV file. "
            f"Available filenames in TSV: {available}"
        )
    print(f"      Found {len(annotations)} events, classes: {annotations.labels}")

    print(f"[2/4] Loading audio: {args.audio}")
    clip = load_audio(args.audio, sr=args.sr)
    print(f"      Duration: {clip.duration:.2f}s, sr={clip.sr} Hz")

    print(f"[3/4] Computing spectrogram ({args.spectrogram_type})...")
    S_db, times, freqs = compute_spectrogram(
        clip, n_fft=args.n_fft, hop_length=args.hop_length, kind=args.spectrogram_type,
    )

    print("[4/4] Building plots and statistics...")
    stem = args.audio.stem
    spec_path = args.output_dir / f"{stem}_spectrogram.png"
    timeline_path = args.output_dir / f"{stem}_timeline.png"
    stats_path = args.output_dir / f"{stem}_stats.txt"
    stats_csv_path = args.output_dir / f"{stem}_stats.csv"

    fig1 = plot_spectrogram_with_events(
        S_db, times, freqs, annotations,
        title=f"Spectrogram with annotations: {args.audio.name}",
        output_path=spec_path,
    )
    fig2 = plot_event_timeline(
        annotations,
        title=f"Event activity: {args.audio.name}",
        output_path=timeline_path,
    )

    stats_df = compute_stats(annotations)
    report = format_stats_report(annotations, stats_df)
    stats_path.write_text(report, encoding="utf-8")
    stats_df.to_csv(stats_csv_path, index=False)

    print()
    print(report)
    print()
    print(f"Saved:\n  {spec_path}\n  {timeline_path}\n  {stats_path}\n  {stats_csv_path}")

    if not args.no_show:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == "__main__":
    main()