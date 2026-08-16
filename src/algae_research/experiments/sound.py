"""Browser-assisted sound-exposure sessions derived from base-algae."""

from __future__ import annotations

import csv
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from numbers import Integral, Real
from pathlib import Path
from typing import Any

DEFAULT_TONE_URL = "https://www.szynalski.com/tone-generator/"
LOG_FIELDS = (
    "date",
    "frequency_hz",
    "duration_seconds",
    "start_time",
    "end_time",
    "system",
    "browser",
    "website",
    "status",
)


@dataclass(frozen=True, slots=True)
class SoundExperimentConfig:
    frequencies_hz: tuple[int, ...] = (200, 10_000, 18_000)
    duration_seconds: float = 10
    break_seconds: float = 10
    log_path: Path = Path("outputs/sound_experiment_log.csv")
    base_url: str = DEFAULT_TONE_URL
    set_windows_volume: bool = False
    headless: bool = False

    def __post_init__(self) -> None:
        if not self.frequencies_hz:
            raise ValueError("At least one positive integer frequency is required.")
        for frequency in self.frequencies_hz:
            if isinstance(frequency, bool) or not isinstance(frequency, Integral) or frequency <= 0:
                raise ValueError("Frequencies must be positive integers.")
        for name, value in (
            ("duration_seconds", self.duration_seconds),
            ("break_seconds", self.break_seconds),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite, non-negative real number.")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValueError("base_url must be an HTTP(S) URL.")


@dataclass(frozen=True, slots=True)
class SoundExperimentLogEntry:
    date: str
    frequency_hz: int
    duration_seconds: float
    start_time: str
    end_time: str
    system: str
    browser: str
    website: str
    status: str


def build_frequency_url(frequency_hz: int, base_url: str = DEFAULT_TONE_URL) -> str:
    if frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive.")
    return f"{base_url.rstrip('/')}/#{frequency_hz}"


def append_log(entry: SoundExperimentLogEntry, destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    if not write_header:
        with path.open(newline="", encoding="utf-8") as handle:
            existing_header = next(csv.reader(handle), None)
        if tuple(existing_header or ()) != LOG_FIELDS:
            raise ValueError(
                f"Existing sound experiment log has an incompatible header: {path}"
            )
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(entry))
    return path


def _default_driver_factory(headless: bool):
    try:
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
    except ImportError as exc:
        raise RuntimeError(
            "Selenium is required for sound experiments. Install with "
            "pip install 'algae-research-toolkit[experiment]'."
        ) from exc
    options = Options()
    options.add_argument("--autoplay-policy=no-user-gesture-required")
    options.add_argument("--disable-background-timer-throttling")
    options.add_argument("--disable-backgrounding-occluded-windows")
    if headless:
        options.add_argument("--headless=new")
    return webdriver.Edge(options=options)


def _button_text(button: Any) -> str:
    return str(button.text or button.get_attribute("value") or "").strip().lower()


def _find_play_button(driver: Any):
    try:
        return driver.find_element("id", "play-button")
    except Exception:  # noqa: BLE001 - Selenium driver exceptions vary by version.
        candidates = driver.find_elements("css selector", "button, input[type='button'], a")
        for candidate in candidates:
            if any(word in _button_text(candidate) for word in ("play", "stop")):
                return candidate
    raise RuntimeError("The tone generator's Play/Stop control was not found.")


def _set_windows_audio_to_maximum() -> None:
    if platform.system() != "Windows":
        return
    try:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume
    except ImportError as exc:
        raise RuntimeError("Windows volume control requires pycaw and comtypes.") from exc

    speakers = AudioUtilities.GetSpeakers()
    try:
        volume = speakers.EndpointVolume
    except AttributeError:
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(interface, POINTER(IAudioEndpointVolume))
    volume.SetMute(0, None)
    volume.SetMasterVolumeLevelScalar(1.0, None)
    for session in AudioUtilities.GetAllSessions():
        try:
            if session.Process and session.Process.name().lower() == "msedge.exe":
                edge_volume = session._ctl.QueryInterface(ISimpleAudioVolume)
                edge_volume.SetMute(0, None)
                edge_volume.SetMasterVolume(1.0, None)
        except Exception:  # noqa: BLE001, S112 - skip unrelated Windows audio sessions.
            continue


def run_sound_experiment(
    config: SoundExperimentConfig,
    *,
    driver_factory: Callable[[bool], Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> list[SoundExperimentLogEntry]:
    """Run configured tone sessions and persist one auditable row per frequency."""

    factory = driver_factory or _default_driver_factory
    if config.set_windows_volume:
        _set_windows_audio_to_maximum()
    driver = factory(config.headless)
    entries: list[SoundExperimentLogEntry] = []
    try:
        for index, frequency in enumerate(config.frequencies_hz):
            started = datetime.now().astimezone()
            status = "success"
            url = build_frequency_url(frequency, config.base_url)
            try:
                driver.get(url)
                if config.set_windows_volume:
                    _set_windows_audio_to_maximum()
                button = _find_play_button(driver)
                if "play" in _button_text(button) or "stop" not in _button_text(button):
                    button.click()
                sleep(config.duration_seconds)
                button = _find_play_button(driver)
                if "stop" in _button_text(button) or "play" not in _button_text(button):
                    button.click()
            except Exception as exc:  # noqa: BLE001 - failures are captured in the audit log.
                status = f"failed: {exc}"
            ended = datetime.now().astimezone()
            entry = SoundExperimentLogEntry(
                date=started.date().isoformat(),
                frequency_hz=frequency,
                duration_seconds=config.duration_seconds,
                start_time=started.isoformat(timespec="seconds"),
                end_time=ended.isoformat(timespec="seconds"),
                system=platform.system(),
                browser="Microsoft Edge",
                website=url,
                status=status,
            )
            append_log(entry, config.log_path)
            entries.append(entry)
            if index < len(config.frequencies_hz) - 1:
                sleep(config.break_seconds)
    finally:
        driver.quit()
    return entries
