"""Minimum-image neighborhoods for protonation Monte Carlo."""

from __future__ import annotations

import argparse

import numpy as np

from Montecarlo_2 import Neighborhood


PBC_MODES = ("auto", "trajectory", "box", "bounds", "none")


def add_pbc_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared periodic-boundary options to an argument parser."""
    parser.add_argument(
        "--pbc-mode",
        choices=PBC_MODES,
        default="auto",
        help=(
            "Box source: auto, trajectory, box, bounds, or none "
            "(default: auto)"
        ),
    )
    parser.add_argument(
        "--box-lengths",
        type=float,
        nargs=3,
        metavar=("LX", "LY", "LZ"),
        help="Side lengths of an orthorhombic box in nm",
    )
    parser.add_argument(
        "--bounds-padding",
        type=float,
        default=0.1,
        help="Padding in nm added to each side in bounds mode (default: 0.1)",
    )


def _valid_box(box_vectors: np.ndarray | None) -> bool:
    if box_vectors is None:
        return False
    box = np.asarray(box_vectors, dtype=float)
    return (
        box.shape == (3, 3)
        and np.all(np.isfinite(box))
        and abs(float(np.linalg.det(box))) > 1.0e-12
    )


def resolve_box_vectors(
    mode: str,
    trajectory_box: np.ndarray | None,
    xyz: np.ndarray,
    box_lengths: list[float] | tuple[float, float, float] | None,
    bounds_padding: float = 0.1,
) -> tuple[np.ndarray | None, str]:
    """Resolve a frame's box and return its source."""
    if mode not in PBC_MODES:
        raise ValueError(f"Unknown PBC mode: {mode}")
    if not np.isfinite(bounds_padding) or bounds_padding < 0:
        raise ValueError("--bounds-padding must be a non-negative number")

    trajectory_is_valid = _valid_box(trajectory_box)
    supplied_box = None
    if box_lengths is not None:
        lengths = np.asarray(box_lengths, dtype=float)
        if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
            raise ValueError("--box-lengths requires three positive lengths in nm")
        supplied_box = np.diag(lengths)

    if mode == "none":
        return None, "none"
    if mode == "trajectory":
        if not trajectory_is_valid:
            raise ValueError("The DCD does not contain a valid periodic box")
        return np.asarray(trajectory_box, dtype=float), "trajectory"
    if mode == "box":
        if supplied_box is None:
            raise ValueError("--pbc-mode box requires --box-lengths LX LY LZ")
        return supplied_box, "box"

    if mode == "auto":
        if trajectory_is_valid:
            return np.asarray(trajectory_box, dtype=float), "trajectory"
        if supplied_box is not None:
            return supplied_box, "box"

    # In bounds mode, or as auto's final fallback, interpret the frame's atomic
    # extent as an orthorhombic box.
    coords = np.asarray(xyz, dtype=float)
    lengths = np.ptp(coords, axis=0) + 2.0 * bounds_padding
    if lengths.shape != (3,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("Could not estimate a valid box from the coordinates")
    return np.diag(lengths), "bounds"


class PeriodicNeighborhood(Neighborhood):
    """Implement ``Neighborhood`` using the DCD cell."""

    def __init__(self, protein, cutoff=6.0):
        super().__init__(protein, cutoff=cutoff)
        self.box_vectors = None

    def set_box(self, box_vectors: np.ndarray | None) -> None:
        """Set the current frame's cell vectors, expressed in nm."""
        if not _valid_box(box_vectors):
            self.box_vectors = None
        else:
            self.box_vectors = np.asarray(box_vectors, dtype=float)

    def _prepare_frame(self) -> None:
        if self._geometry_version == self.protein._geometry_version:
            return

        coords = self.protein.representative_coords
        center_indices = [
            self.protein._representative_resid_to_index[resid]
            for resid in self.protein._charge_resids
            if resid in self.protein._representative_resid_to_index
        ]

        if center_indices:
            displacement = (
                coords[np.asarray(center_indices), np.newaxis, :]
                - coords[np.newaxis, :, :]
            )
            if self.box_vectors is not None:
                fractional = displacement @ np.linalg.inv(self.box_vectors)
                fractional -= np.rint(fractional)
                displacement = fractional @ self.box_vectors
            distances = np.linalg.norm(displacement, axis=2)
            self._distance_rows = dict(zip(center_indices, distances))
        else:
            self._distance_rows = {}

        self._masks.clear()
        self._geometry_version = self.protein._geometry_version
