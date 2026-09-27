import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.annotations import load_annotations
from src.stats import compute_stats

FIXTURE_TSV = Path(__file__).parent / "fixture.tsv"


@pytest.fixture(scope="module", autouse=True)
def make_fixture():
    FIXTURE_TSV.write_text(
        "id\tfilename\tstart\tstop\tlabel\tsource\tscore\n"
        "1\ta.mp3\t0:00:00.000000\t0:00:01.000000\tdog\tmanual\t1.0\n"
        "2\ta.mp3\t0:00:02.000000\t0:00:03.500000\tOwl\tmanual\t1.0\n"
        "3\ta.mp3\t0:00:05.000000\t0:00:06.000000\tdog\tmanual\t1.0\n"
        "4\tb.mp3\t0:00:00.000000\t0:00:02.000000\tcar\tmanual\t1.0\n"
    )
    yield
    FIXTURE_TSV.unlink(missing_ok=True)


def test_load_annotations_parses_time_correctly():
    ann = load_annotations(FIXTURE_TSV)
    assert len(ann) == 4
    # Annotations are sorted globally by start_s, so we filter by file
    # to uniquely fetch the row for Owl (2.0–3.5s).
    sub = ann.for_file("a.mp3")
    row = sub.df.iloc[1]
    assert row["label"] == "Owl"
    assert row["start_s"] == pytest.approx(2.0)
    assert row["stop_s"] == pytest.approx(3.5)
    assert row["duration"] == pytest.approx(1.5)


def test_for_file_filters_correctly():
    ann = load_annotations(FIXTURE_TSV)
    sub = ann.for_file("a.mp3")
    assert len(sub) == 3
    assert set(sub.df["filename"]) == {"a.mp3"}


def test_missing_required_column_raises(tmp_path):
    bad_tsv = tmp_path / "bad.tsv"
    bad_tsv.write_text("filename\tstart\tstop\na.mp3\t0:00:00\t0:00:01\n")
    with pytest.raises(ValueError, match="required columns"):
        load_annotations(bad_tsv)


def test_stop_before_start_raises(tmp_path):
    bad_tsv = tmp_path / "bad_order.tsv"
    bad_tsv.write_text(
        "filename\tstart\tstop\tlabel\n"
        "a.mp3\t0:00:05.000000\t0:00:01.000000\tdog\n"
    )
    with pytest.raises(ValueError, match="before"):
        load_annotations(bad_tsv)


def test_compute_stats_counts_and_durations():
    ann = load_annotations(FIXTURE_TSV).for_file("a.mp3")
    stats = compute_stats(ann)
    dog_row = stats[stats["label"] == "dog"].iloc[0]
    owl_row = stats[stats["label"] == "Owl"].iloc[0]

    assert dog_row["count"] == 2
    assert dog_row["total_duration_s"] == pytest.approx(2.0)
    assert owl_row["count"] == 1
    assert owl_row["total_duration_s"] == pytest.approx(1.5)
    assert stats["share_of_labeled_time"].sum() == pytest.approx(1.0)