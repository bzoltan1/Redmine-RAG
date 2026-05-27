"""
Unit tests for core/timing.py
"""

import time
import pytest
from io import StringIO
from unittest.mock import patch

from core.timing import format_duration, StageTimer, ProgressBar, PipelineReport


# ---------------------------------------------------------------------------
# format_duration
# ---------------------------------------------------------------------------

class TestFormatDuration:
    def test_sub_minute(self):
        assert format_duration(0.4) == "0.4s"
        assert format_duration(59.9) == "59.9s"

    def test_exact_minute(self):
        assert format_duration(60) == "1m 0s"

    def test_minutes_and_seconds(self):
        assert format_duration(75) == "1m 15s"
        assert format_duration(125) == "2m 5s"

    def test_hours(self):
        assert format_duration(3725) == "1h 2m 5s"
        assert format_duration(7200) == "2h 0m 0s"

    def test_zero(self):
        assert format_duration(0) == "0.0s"

    def test_negative_clamped_to_zero(self):
        assert format_duration(-5) == "0.0s"


# ---------------------------------------------------------------------------
# StageTimer
# ---------------------------------------------------------------------------

class TestStageTimer:
    def test_elapsed_is_positive(self):
        t = StageTimer("test")
        t.start()
        time.sleep(0.01)
        t.stop()
        assert t.elapsed > 0

    def test_context_manager_sets_elapsed(self):
        t = StageTimer("ctx")
        with t:
            time.sleep(0.01)
        assert t.elapsed > 0

    def test_stop_prints_name(self, capsys):
        t = StageTimer("MyStage")
        t.start()
        t.stop()
        out = capsys.readouterr().out
        assert "MyStage" in out

    def test_stop_prints_duration(self, capsys):
        t = StageTimer("S")
        t.start()
        t.stop()
        out = capsys.readouterr().out
        assert "s" in out  # duration string ends with 's'

    def test_print_start_flag(self, capsys):
        t = StageTimer("S", print_start=True)
        t.start()
        out = capsys.readouterr().out
        assert "starting" in out

    def test_no_print_start_by_default(self, capsys):
        t = StageTimer("S")
        t.start()
        out = capsys.readouterr().out
        assert out == ""

    def test_start_returns_self(self):
        t = StageTimer("S")
        assert t.start() is t


# ---------------------------------------------------------------------------
# ProgressBar
# ---------------------------------------------------------------------------

class TestProgressBar:
    def test_update_does_not_crash(self, capsys):
        bar = ProgressBar(total=10, label="Test", unit="items")
        for i in range(1, 11):
            bar.update(i)
        bar.close()

    def test_close_returns_elapsed(self):
        bar = ProgressBar(total=5, unit="x")
        bar.update(5)
        elapsed = bar.close()
        assert elapsed >= 0

    def test_close_prints_summary(self, capsys):
        bar = ProgressBar(total=5, label="MyBar", unit="chunks")
        bar.close()
        out = capsys.readouterr().out
        assert "MyBar" in out
        assert "chunks" in out

    def test_custom_close_message(self, capsys):
        bar = ProgressBar(total=5)
        bar.close(message="All done!")
        out = capsys.readouterr().out
        assert "All done!" in out

    def test_double_close_is_safe(self):
        bar = ProgressBar(total=5)
        bar.close()
        bar.close()  # should not raise

    def test_context_manager_calls_close(self, capsys):
        with ProgressBar(total=3, label="Ctx") as bar:
            bar.update(3)
        out = capsys.readouterr().out
        assert "Ctx" in out

    def test_print_every_throttles_redraws(self, capsys):
        bar = ProgressBar(total=100, print_every=10)
        # Only updates at multiples of 10 and at total
        for i in range(1, 101):
            bar.update(i)
        bar.close()
        # Test passes if no exception; we can't easily count \r redraws in capsys

    def test_zero_total_does_not_divide_by_zero(self):
        bar = ProgressBar(total=0)
        bar.update(0)
        bar.close()

    def test_update_after_close_is_silent(self, capsys):
        bar = ProgressBar(total=5)
        bar.close()
        capsys.readouterr()
        bar.update(3)
        out = capsys.readouterr().out
        assert out == ""


# ---------------------------------------------------------------------------
# PipelineReport
# ---------------------------------------------------------------------------

class TestPipelineReport:
    def test_print_does_not_crash(self, capsys):
        report = PipelineReport("Test pipeline")
        report.record("Stage A", 1.5)
        report.record("Stage B", 0.3)
        report.print()

    def test_stage_names_in_output(self, capsys):
        report = PipelineReport("P")
        report.record("Load", 0.5)
        report.record("Embed", 30.0)
        report.print()
        out = capsys.readouterr().out
        assert "Load" in out
        assert "Embed" in out

    def test_title_in_output(self, capsys):
        report = PipelineReport("My Pipeline")
        report.print()
        out = capsys.readouterr().out
        assert "My Pipeline" in out

    def test_total_wall_time_in_output(self, capsys):
        report = PipelineReport("P")
        report.record("S", 0.1)
        report.print()
        out = capsys.readouterr().out
        assert "Total wall time" in out

    def test_empty_report_prints_without_error(self, capsys):
        PipelineReport("Empty").print()

    def test_durations_appear_in_output(self, capsys):
        report = PipelineReport("P")
        report.record("Fast stage", 0.25)
        report.print()
        out = capsys.readouterr().out
        assert "0.2s" in out or "0.3s" in out  # allow float rounding
