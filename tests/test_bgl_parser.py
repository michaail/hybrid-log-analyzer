"""Tests for src/parser/bgl_parser.py

The tests deliberately avoid running Drain3 end-to-end (that requires a
full log file and ~minutes of wall time). Instead they exercise the
format-specific logic added by :class:`BGLParser` — header-token
stripping and the :meth:`_extract_row` column mapping — on hand-crafted
lines covering the real LogHub edge cases.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

pytest.importorskip("drain3", reason="drain3 not installed in this environment")

from src.parser import BGLParser, DrainParser  # noqa: E402

CONFIG = str(Path(__file__).resolve().parents[1] / "configs" / "drain_bgl.ini")


# ---------- fixtures --------------------------------------------------

NORMAL_LINE = (
    "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.363779 "
    "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected"
)

ANOMALY_LINE = (
    "KERNDTLB 1117955293 2005.06.05 R20-M0-N2-C:J10-U11 2005-06-05-00.08.13.410695 "
    "R20-M0-N2-C:J10-U11 RAS KERNEL FATAL data TLB error interrupt"
)

NULL_NODE_LINE = (
    "- 1123110937 2005.08.03 NULL 2005-08-03-16.15.37.221795 NULL "
    "RAS MMCS ERROR idoproxydb hit ASSERT condition"
)

IO_NODE_LINE = (
    "APPSEV 1126539187 2005.09.12 R51-M1-N0-I:J18-U11 2005-09-12-08.33.07.281577 "
    "R51-M1-N0-I:J18-U11 RAS APP FATAL ciod: Error reading message prefix"
)


@pytest.fixture()
def parser() -> BGLParser:
    return BGLParser(config_path=CONFIG)


# ---------- header-stripping -----------------------------------------

def test_header_tokens_constant_overridden() -> None:
    """BGLParser must strip 6 tokens; base DrainParser strips 3."""
    assert BGLParser._HEADER_TOKENS == 6
    assert DrainParser._HEADER_TOKENS == 3


def test_preprocess_line_strips_six_tokens() -> None:
    """After stripping 6 tokens, Drain should see `<component> <subcomponent> <level> <message>`."""
    stripped = BGLParser._preprocess_line(NORMAL_LINE, strip_tokens=6)
    assert stripped.startswith("RAS KERNEL INFO ")
    assert "1117838570" not in stripped
    assert "R02-M1-N0-C:J12-U11" not in stripped


def test_preprocess_line_short_line_returns_original() -> None:
    """Lines with fewer tokens than requested fall through unchanged."""
    short = "KERNDTLB 1117955293 2005.06.05"
    assert BGLParser._preprocess_line(short, strip_tokens=6) == short


# ---------- _extract_row schema --------------------------------------

EXPECTED_COLUMNS = {
    "label", "is_anomaly", "unix_ts", "date", "node_id", "timestamp",
    "component", "subcomponent", "level", "raw", "cluster_id",
    "template", "parameters",
}


def test_extract_row_has_expected_columns(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, cluster_id=1, template_str="RAS KERNEL INFO <*>", params=[])
    assert set(row.keys()) == EXPECTED_COLUMNS


def test_extract_row_normal_flagged_not_anomalous(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["label"] == "-"
    assert row["is_anomaly"] is False


def test_extract_row_kerndtlb_flagged_anomalous(parser: BGLParser) -> None:
    row = parser._extract_row(ANOMALY_LINE, 2, "t", [])
    assert row["label"] == "KERNDTLB"
    assert row["is_anomaly"] is True


def test_extract_row_parses_unix_ts(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["unix_ts"] == 1117838570


def test_extract_row_parses_datetime(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["timestamp"] == datetime(2005, 6, 3, 15, 42, 50, 363779)


def test_extract_row_node_and_component(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["node_id"] == "R02-M1-N0-C:J12-U11"
    assert row["component"] == "RAS"
    assert row["subcomponent"] == "KERNEL"
    assert row["level"] == "INFO"


def test_extract_row_keeps_null_node_literal(parser: BGLParser) -> None:
    """NULL is a legitimate BGL node sentinel (MMCS logs)."""
    row = parser._extract_row(NULL_NODE_LINE, 1, "t", [])
    assert row["node_id"] == "NULL"
    assert row["subcomponent"] == "MMCS"


def test_extract_row_io_node_suffix(parser: BGLParser) -> None:
    """I/O nodes use the ``-I:J..-U..`` suffix instead of ``-C:J..-U..``."""
    row = parser._extract_row(IO_NODE_LINE, 1, "t", [])
    assert row["node_id"] == "R51-M1-N0-I:J18-U11"
    assert row["component"] == "RAS"
    assert row["subcomponent"] == "APP"
    assert row["level"] == "FATAL"


def test_extract_row_preserves_raw(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["raw"] == NORMAL_LINE


def test_extract_row_cluster_id_and_params_passthrough(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, cluster_id=42, template_str="tpl", params=["a", "b"])
    assert row["cluster_id"] == 42
    assert row["template"] == "tpl"
    assert row["parameters"] == ["a", "b"]


def test_extract_row_empty_params_becomes_empty_list(parser: BGLParser) -> None:
    row = parser._extract_row(NORMAL_LINE, 1, "t", [])
    assert row["parameters"] == []


def test_extract_row_malformed_ts_returns_none(parser: BGLParser) -> None:
    """Corrupt datetime token should not raise — downstream code must treat None."""
    bad = NORMAL_LINE.replace("2005-06-03-15.42.50.363779", "not-a-date")
    row = parser._extract_row(bad, 1, "t", [])
    assert row["timestamp"] is None


def test_extract_row_malformed_unix_ts_returns_none(parser: BGLParser) -> None:
    bad = NORMAL_LINE.replace("1117838570", "notaninteger")
    row = parser._extract_row(bad, 1, "t", [])
    assert row["unix_ts"] is None


def test_extract_row_truncated_line_pads_safely(parser: BGLParser) -> None:
    """Short lines should yield empty strings for missing tokens, not crash."""
    truncated = "- 1117838570 2005.06.03"
    row = parser._extract_row(truncated, 1, "t", [])
    # Required keys still present; missing fields are empty strings or None
    assert set(row.keys()) == EXPECTED_COLUMNS
    assert row["label"] == "-"
    assert row["component"] == ""


# ---------- end-to-end smoke test ------------------------------------

def test_fit_and_annotate_roundtrip(tmp_path: Path, parser: BGLParser) -> None:
    """Small 4-line file: 2 normals + 2 anomalies → assert template count & columns."""
    log = tmp_path / "mini_bgl.log"
    log.write_text("\n".join([NORMAL_LINE, ANOMALY_LINE, NULL_NODE_LINE, IO_NODE_LINE]) + "\n")

    parser.fit_file(str(log))
    df = parser.annotate_file(str(log))

    # Smoke checks: row count preserved, anomaly count correct, schema intact.
    assert len(df) == 4
    assert df["is_anomaly"].sum() == 2                # KERNDTLB + APPSEV
    assert set(df.columns) == EXPECTED_COLUMNS
    # Every line got a cluster assignment.
    assert df["cluster_id"].notna().all()
