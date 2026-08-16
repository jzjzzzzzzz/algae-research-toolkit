import csv
from pathlib import Path

import pytest

from algae_research.experiments.sound import (
    SoundExperimentConfig,
    SoundExperimentLogEntry,
    append_log,
    build_frequency_url,
    run_sound_experiment,
)


class FakeButton:
    def __init__(self):
        self.text = "Play"
        self.clicks = 0

    def get_attribute(self, _name):
        return self.text

    def click(self):
        self.clicks += 1
        self.text = "Stop" if self.text == "Play" else "Play"


class FakeDriver:
    def __init__(self):
        self.button = FakeButton()
        self.urls = []
        self.closed = False

    def get(self, url):
        self.urls.append(url)
        self.button.text = "Play"

    def find_element(self, _method, _selector):
        return self.button

    def find_elements(self, _method, _selector):
        return [self.button]

    def quit(self):
        self.closed = True


class CleanupFailDriver(FakeDriver):
    def quit(self):
        self.closed = True
        raise RuntimeError("cleanup failed")


def test_frequency_url_uses_hash_route():
    assert build_frequency_url(200) == "https://www.szynalski.com/tone-generator/#200"


def test_fake_sound_run_logs_each_frequency(tmp_path: Path):
    driver = FakeDriver()
    log = tmp_path / "experiment.csv"
    config = SoundExperimentConfig(
        frequencies_hz=(200, 10_000),
        duration_seconds=0,
        break_seconds=0,
        log_path=log,
    )
    entries = run_sound_experiment(
        config,
        driver_factory=lambda _headless: driver,
        sleep=lambda _seconds: None,
    )
    assert [entry.status for entry in entries] == ["success", "success"]
    assert driver.button.clicks == 4
    assert driver.closed is True
    with log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frequency_hz"]) for row in rows] == [200, 10_000]


@pytest.mark.parametrize("frequencies", [(), (0,), (-1,), (200.5,), ("200",), (True,)])
def test_configuration_rejects_invalid_frequencies(frequencies):
    with pytest.raises(ValueError, match="positive integer|positive integers"):
        SoundExperimentConfig(frequencies_hz=frequencies)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration_seconds", -1),
        ("duration_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("duration_seconds", True),
        ("break_seconds", -1),
        ("break_seconds", float("nan")),
        ("break_seconds", float("inf")),
        ("break_seconds", True),
    ],
)
def test_configuration_rejects_invalid_durations(field, value):
    with pytest.raises(ValueError, match=field):
        SoundExperimentConfig(**{field: value})


def test_zero_duration_and_break_remain_supported():
    config = SoundExperimentConfig(duration_seconds=0, break_seconds=0)
    assert config.duration_seconds == config.break_seconds == 0


def _log_entry(frequency_hz: int = 200) -> SoundExperimentLogEntry:
    return SoundExperimentLogEntry(
        date="2026-08-16",
        frequency_hz=frequency_hz,
        duration_seconds=10,
        start_time="2026-08-16T10:00:00+08:00",
        end_time="2026-08-16T10:00:10+08:00",
        system="TestOS",
        browser="Test Browser",
        website=f"https://example.test/#{frequency_hz}",
        status="success",
    )


def test_append_log_accepts_its_existing_schema(tmp_path: Path):
    log = tmp_path / "experiment.csv"
    append_log(_log_entry(200), log)
    append_log(_log_entry(10_000), log)

    with log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["frequency_hz"]) for row in rows] == [200, 10_000]


def test_append_log_rejects_incompatible_header_without_modifying_file(tmp_path: Path):
    log = tmp_path / "experiment.csv"
    original = "wrong,header\n1,2\n"
    log.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible header"):
        append_log(_log_entry(), log)

    assert log.read_text(encoding="utf-8") == original


def test_cleanup_failure_warns_without_erasing_completed_entries(tmp_path: Path):
    driver = CleanupFailDriver()
    log = tmp_path / "experiment.csv"
    config = SoundExperimentConfig(
        frequencies_hz=(200,),
        duration_seconds=0,
        break_seconds=0,
        log_path=log,
    )

    with pytest.warns(RuntimeWarning, match="cleanup failed"):
        entries = run_sound_experiment(
            config,
            driver_factory=lambda _headless: driver,
            sleep=lambda _seconds: None,
        )

    assert driver.closed is True
    assert [entry.status for entry in entries] == ["success"]
    with log.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["success"]
