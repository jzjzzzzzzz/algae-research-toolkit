import csv
from pathlib import Path

from algae_research.experiments.sound import (
    SoundExperimentConfig,
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


def test_configuration_rejects_invalid_frequency():
    try:
        SoundExperimentConfig(frequencies_hz=(0,))
    except ValueError as exc:
        assert "positive frequency" in str(exc)
    else:
        raise AssertionError("Expected invalid frequency to fail.")
