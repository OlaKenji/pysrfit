"""Editable run-correction state shared by the GUI and data processing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .io.asymmetry import detector_group
from .io.base import HistogramRun
from .models import RunConfig


@dataclass(frozen=True)
class HistogramTarget:
    """One raw detector histogram selectable in the correction editor."""

    role: str
    position: int
    detector_index: int
    detector_number: int
    label: str


class RunCorrectionState:
    """Hold visual t0/data/background selections before applying them to a RUN."""

    MARKERS = {
        "t": ("t0", None),
        "d": ("data", 0),
        "D": ("data", 1),
        "b": ("background", 0),
        "B": ("background", 1),
    }

    def __init__(self, data: HistogramRun, run: RunConfig):
        if run.fittype not in (0, 2):
            raise ValueError("Visual corrections currently support fittype 0 and 2")
        self.data = data
        self.fittype = run.fittype
        self._has_fixed_background = bool(run.tokens("backgr.fix"))
        self._indices = {"forward": detector_group(data, run.tokens("forward"))}
        if run.fittype == 2:
            self._indices["backward"] = detector_group(data, run.tokens("backward"))
            if len(self._indices["forward"]) != len(self._indices["backward"]):
                raise ValueError("Asymmetry detector groups must have the same size")

        self.targets = self._make_targets()
        self.t0 = self._initial_t0(run)
        self.ranges = {
            "data": self._initial_ranges(run, "data"),
            "background": self._initial_ranges(run, "background"),
        }

    @property
    def roles(self) -> tuple[str, ...]:
        return ("forward", "backward") if self.fittype == 2 else ("forward",)

    def _make_targets(self) -> list[HistogramTarget]:
        targets = []
        for position in range(len(self._indices["forward"])):
            for role in self.roles:
                index = self._indices[role][position]
                detector = self.data.detectors[index]
                targets.append(HistogramTarget(
                    role=role,
                    position=position,
                    detector_index=index,
                    detector_number=detector.number,
                    label=f"{role.title()} {position + 1}: {detector.number} — {detector.name}",
                ))
        return targets

    def _initial_t0(self, run: RunConfig) -> dict[tuple[str, int], int | None]:
        tokens = run.tokens("t0")
        expected = len(self._indices["forward"]) * len(self.roles)
        configured = [float(value) for value in tokens] if len(tokens) == expected else []
        result = {}
        for position in range(len(self._indices["forward"])):
            for role_index, role in enumerate(self.roles):
                if configured:
                    value = configured[position * len(self.roles) + role_index]
                else:
                    detector = self.data.detectors[self._indices[role][position]]
                    value = detector.t0_bin
                result[(role, position)] = None if value is None else int(round(value))
        return result

    def _initial_ranges(self, run: RunConfig, name: str) -> dict[str, list[int | None]]:
        tokens = run.tokens(name)
        expected = 2 * len(self.roles)
        if len(tokens) >= expected:
            values = [int(float(value)) for value in tokens[:expected]]
            return {
                role: values[2 * role_index:2 * role_index + 2]
                for role_index, role in enumerate(self.roles)
            }
        result = {}
        for role in self.roles:
            detector = self.data.detectors[self._indices[role][0]]
            if name == "data" and None not in (detector.first_good_bin, detector.last_good_bin):
                result[role] = [int(detector.first_good_bin), int(detector.last_good_bin)]
            else:
                result[role] = [None, None]
        return result

    def counts(self, target_index: int) -> np.ndarray:
        return self.data.counts[self.targets[target_index].detector_index]

    def marker_values(self, target_index: int) -> dict[str, int | None | list[int | None]]:
        target = self.targets[target_index]
        return {
            "t0": self.t0[(target.role, target.position)],
            "data": list(self.ranges["data"][target.role]),
            "background": list(self.ranges["background"][target.role]),
        }

    def set_marker(self, target_index: int, key: str, channel: int) -> None:
        if key not in self.MARKERS:
            raise ValueError(f"Unknown correction key {key!r}")
        channel = int(channel)
        if not 0 <= channel < self.data.counts.shape[1]:
            raise ValueError(f"Channel must be between 0 and {self.data.counts.shape[1] - 1}")
        target = self.targets[target_index]
        name, endpoint = self.MARKERS[key]
        if name == "t0":
            self.t0[(target.role, target.position)] = channel
        else:
            self.ranges[name][target.role][endpoint] = channel

    @staticmethod
    def _validate_pair(name: str, role: str, pair: list[int | None]) -> None:
        if pair == [None, None]:
            return
        if None in pair:
            raise ValueError(f"Set both {name} limits for {role}")
        if pair[0] > pair[1]:
            raise ValueError(f"{name.title()} start exceeds end for {role}")

    def validate(self) -> None:
        if any(value is None for value in self.t0.values()):
            raise ValueError("Set t0 for every selected detector")
        for name in ("data", "background"):
            for role in self.roles:
                self._validate_pair(name, role, self.ranges[name][role])
            present = [self.ranges[name][role] != [None, None] for role in self.roles]
            if any(present) and not all(present):
                raise ValueError(f"Set {name} limits for both forward and backward")
        if (self.fittype == 2 and not self._has_fixed_background
                and all(self.ranges["background"][role] == [None, None] for role in self.roles)):
            raise ValueError("Set background limits for both forward and backward")

    def apply_to(self, run: RunConfig) -> None:
        self.validate()
        t0_values = []
        for position in range(len(self._indices["forward"])):
            for role in self.roles:
                t0_values.append(str(self.t0[(role, position)]))
        run.options["t0"] = t0_values
        for name in ("data", "background"):
            values = [
                value
                for role in self.roles
                for value in self.ranges[name][role]
            ]
            if all(value is None for value in values):
                run.options.pop(name, None)
            else:
                run.options[name] = [str(value) for value in values]
