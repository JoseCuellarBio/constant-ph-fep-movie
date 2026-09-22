#!/usr/bin/env python3
"""Stage two: calculate energies from saved Monte Carlo states."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import mdtraj as md
import numpy as np

from periodic_neighborhood import add_pbc_arguments, resolve_box_vectors

CHARGE_COLUMN = re.compile(r"^([A-Za-z]+)(-?\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--dcd", required=True)
    parser.add_argument("--charges-csv", required=True)
    parser.add_argument("--attempts-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk", type=int, default=100)
    add_pbc_arguments(parser)
    return parser.parse_args()


def representative_atom_indices(topology: md.Topology, columns: list[str]) -> np.ndarray:
    residues = {residue.resSeq: residue for residue in topology.residues}
    if len(residues) != topology.n_residues:
        raise ValueError("The topology contains duplicate residue numbers")
    indices = []
    for column in columns:
        match = CHARGE_COLUMN.fullmatch(column)
        if match is None:
            raise ValueError(f"Invalid charge column: {column}")
        name, number = match.groups()
        residue = residues.get(int(number))
        if residue is None or residue.name.upper() != name.upper():
            raise ValueError(f"Residue {column} is missing or differs in the topology")
        atoms = {atom.name: atom.index for atom in residue.atoms}
        index = next((atoms[a] for a in ("CB", "CA", "O") if a in atoms), None)
        if index is None:
            raise ValueError(f"Residue {column} has no CB, CA, or O atom")
        indices.append(index)
    return np.asarray(indices, dtype=int)


def interaction_matrix(coords: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    displacement = np.asarray(coords, dtype=float)[None, :, :] - np.asarray(coords, dtype=float)[:, None, :]
    if box is not None and np.all(np.isfinite(box)):
        fractional = displacement @ np.linalg.inv(box)
        fractional -= np.rint(fractional)
        displacement = fractional @ box
    distances = np.linalg.norm(displacement, axis=2)
    matrix = np.zeros(distances.shape, dtype=float)
    valid = (distances > 0.0) & (distances < 3.0)
    matrix[valid] = 0.5 * np.exp(-distances[valid]) / distances[valid]
    return matrix


def read_final_states(path: Path) -> tuple[list[str], list[np.ndarray]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "frame" not in reader.fieldnames:
            raise ValueError(f"Invalid final-state CSV: {path}")
        columns = [c for c in reader.fieldnames if CHARGE_COLUMN.fullmatch(c)]
        if not columns:
            raise ValueError(f"No charge columns found in {path}")
        states = []
        for expected, row in enumerate(reader):
            if int(row["frame"]) != expected:
                raise ValueError(f"Missing or out-of-order frames in {path}")
            states.append(np.asarray([row[c] for c in columns], dtype=float))
    return columns, states


def read_attempts(path: Path) -> dict[int, list[dict[str, str]]]:
    required = {"frame", "attempt", "residue", "old_charge", "accepted", "resulting_charge"}
    grouped: dict[int, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Invalid attempts CSV: {path}")
        for row in reader:
            grouped.setdefault(int(row["frame"]), []).append(row)
    for frame, rows in grouped.items():
        if [int(r["attempt"]) for r in rows] != list(range(1, len(rows) + 1)):
            raise ValueError(f"Frame {frame}: missing or out-of-order attempts")
    return grouped


def main() -> None:
    args = parse_args()
    if args.chunk < 1:
        raise ValueError("--chunk must be positive")
    for filename in (args.pdb, args.dcd, args.charges_csv, args.attempts_csv):
        if not Path(filename).is_file():
            raise FileNotFoundError(filename)
    columns, final_states = read_final_states(Path(args.charges_csv))
    grouped = read_attempts(Path(args.attempts_csv))
    if len(final_states) != len(grouped):
        raise ValueError(f"Frame counts differ: final={len(final_states)}, attempts={len(grouped)}")

    topology = md.load_topology(args.pdb)
    atom_indices = representative_atom_indices(topology, columns)
    charge_index = {column: index for index, column in enumerate(columns)}
    previous_final = None
    frame_number = 0
    total_attempts = 0
    with Path(args.output).open("w", newline="") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([
            "frame", "debye_huckel_final_kcal_mol",
            "debye_huckel_visited_mean_kcal_mol", "visited_states",
        ])
        for chunk in md.iterload(args.dcd, top=args.pdb, chunk=args.chunk):
            for local_frame, xyz in enumerate(chunk.xyz):
                if frame_number >= len(final_states) or frame_number not in grouped:
                    raise ValueError(f"Frame {frame_number}: missing Monte Carlo data")
                final_state = final_states[frame_number]
                attempts = grouped[frame_number]
                if not attempts:
                    raise ValueError(f"Frame {frame_number}: contains no attempts")
                if previous_final is None:
                    state = final_state.copy()
                    for attempt in reversed(attempts):
                        index = charge_index[attempt["residue"]]
                        if not np.isclose(state[index], float(attempt["resulting_charge"])):
                            raise ValueError("Could not reconstruct the initial state of frame 0")
                        state[index] = float(attempt["old_charge"])
                else:
                    state = previous_final.copy()

                trajectory_box = (
                    chunk.unitcell_vectors[local_frame]
                    if chunk.unitcell_vectors is not None
                    else None
                )
                box, _ = resolve_box_vectors(
                    args.pbc_mode, trajectory_box, xyz, args.box_lengths,
                    args.bounds_padding,
                )
                matrix = interaction_matrix(xyz[atom_indices], box)
                energy = float(0.5 * state @ matrix @ state)
                visited_sum = 0.0
                for attempt in attempts:
                    index = charge_index[attempt["residue"]]
                    old = float(attempt["old_charge"])
                    new = float(attempt["resulting_charge"])
                    if not np.isclose(state[index], old):
                        raise ValueError(f"Frame {frame_number}, attempt {attempt['attempt']}: inconsistent state")
                    if not int(attempt["accepted"]) and not np.isclose(new, old):
                        raise ValueError("A rejected attempt changes the charge")
                    delta = new - old
                    if delta:
                        energy += delta * float(matrix[index] @ state)
                        state[index] = new
                    visited_sum += energy
                if not np.allclose(state, final_state):
                    raise ValueError(f"Frame {frame_number}: final state does not match")
                final_energy = float(0.5 * final_state @ matrix @ final_state)
                writer.writerow([
                    frame_number, f"{final_energy:.8f}",
                    f"{visited_sum / len(attempts):.8f}", len(attempts),
                ])
                previous_final = final_state
                total_attempts += len(attempts)
                frame_number += 1
    if frame_number != len(final_states):
        raise ValueError(f"The DCD has {frame_number} frames and the CSV has {len(final_states)}")
    print(f"Done: {frame_number} frames and {total_attempts} visited states; output: {args.output}")


if __name__ == "__main__":
    main()
