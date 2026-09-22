#!/usr/bin/env python3
"""Stage one: perform Monte Carlo trials without calculating energies."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import mdtraj as md
import numpy as np

from Montecarlo_2 import Protein, calculate_new_charge
from periodic_neighborhood import (
    PeriodicNeighborhood,
    add_pbc_arguments,
    resolve_box_vectors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", required=True)
    parser.add_argument("--dcd", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attempts-output", required=True)
    parser.add_argument("--ph", type=float, default=7.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=100)
    parser.add_argument("--attempts-per-frame", type=int, default=10)
    parser.add_argument(
        "--polar", action=argparse.BooleanOptionalAction, default=True,
        help="Enable or disable the polar penalty (default: enabled)",
    )
    parser.add_argument(
        "--electrostatic", action=argparse.BooleanOptionalAction, default=True,
        help="Enable or disable the electrostatic penalty (default: enabled)",
    )
    add_pbc_arguments(parser)
    return parser.parse_args()


def topology_rows(topology: md.Topology, xyz: np.ndarray) -> list[dict]:
    return [
        {
            "residue_number": atom.residue.resSeq,
            "residue_name": atom.residue.name,
            "atom": atom.name,
            "position_xyz": position,
        }
        for atom, position in zip(topology.atoms, xyz)
    ]


def validate_topology(topology: md.Topology) -> None:
    residue_numbers = [residue.resSeq for residue in topology.residues]
    if len(residue_numbers) != len(set(residue_numbers)):
        raise ValueError("The topology must have unique residue numbers.")


def main() -> None:
    args = parse_args()
    if args.chunk < 1 or args.attempts_per_frame < 1:
        raise ValueError("--chunk and --attempts-per-frame must be positive")
    for path in (Path(args.pdb), Path(args.dcd)):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    topology = md.load_topology(args.pdb)
    validate_topology(topology)
    first = md.load_frame(args.dcd, 0, top=args.pdb)
    rows = topology_rows(topology, first.xyz[0])
    protein = Protein(
        rows, pH=args.ph, use_polar=args.polar,
        use_electrostatic=args.electrostatic,
    )
    protein.neighborhood = PeriodicNeighborhood(protein)
    ionizable = [
        (resid, info["resname"])
        for resid, info in protein.data.items()
        if info["type"] in ("A", "B")
    ]
    if not ionizable:
        raise ValueError("The topology does not contain ionizable residues.")

    charge_state = protein.list_charged_residues.copy()
    columns = [f"{resname}{resid}" for resid, resname in ionizable]
    frame_number = 0
    accepted_count = 0
    pbc_sources = set()
    with Path(args.output).open("w", newline="", buffering=1024 * 1024) as output, Path(
        args.attempts_output
    ).open("w", newline="", buffering=1024 * 1024) as attempts_output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["frame", "attempted_residue", "accepted", *columns])
        attempts_writer = csv.writer(attempts_output, lineterminator="\n")
        attempts_writer.writerow(
            ["frame", "attempt", "residue", "old_charge", "proposed_charge", "accepted", "resulting_charge"]
        )

        for chunk in md.iterload(args.dcd, top=args.pdb, chunk=args.chunk):
            for local_frame, xyz in enumerate(chunk.xyz):
                for row, position in zip(rows, xyz):
                    row["position_xyz"] = position
                protein.update_from_rows(rows)
                trajectory_box = (
                    chunk.unitcell_vectors[local_frame]
                    if chunk.unitcell_vectors is not None
                    else None
                )
                box_vectors, pbc_source = resolve_box_vectors(
                    args.pbc_mode, trajectory_box, xyz, args.box_lengths,
                    args.bounds_padding,
                )
                pbc_sources.add(pbc_source)
                protein.neighborhood.set_box(box_vectors)
                last_accepted = False
                attempted_resid = None
                for attempt_number in range(1, args.attempts_per_frame + 1):
                    attempted_resid = protein.mc.choose_residue()
                    old_charge = protein.data[attempted_resid]["charge"]
                    info = protein.data[attempted_resid]
                    acid_base = -1 if info["type"] == "A" else 1
                    proposed_charge, _ = calculate_new_charge(
                        attempted_resid, acid_base, old_charge
                    )
                    charge_state, particle_info = protein.protonation_mc.attempt_charge_flip(
                        charge_state
                    )
                    last_accepted = particle_info is not None
                    accepted_count += int(last_accepted)
                    resulting_charge = protein.data[attempted_resid]["charge"]
                    attempts_writer.writerow(
                        [frame_number, attempt_number, f"{info['resname']}{attempted_resid}",
                         f"{old_charge:.1f}", f"{proposed_charge:.1f}", int(last_accepted),
                         f"{resulting_charge:.1f}"]
                    )

                state = dict(charge_state)
                label = f"{protein.data[attempted_resid]['resname']}{attempted_resid}"
                writer.writerow(
                    [frame_number, label, int(last_accepted),
                     *(f"{state[resid]:.1f}" for resid, _ in ionizable)]
                )
                frame_number += 1

    print(
        f"Done: {frame_number} frames, {frame_number * args.attempts_per_frame} "
        f"attempts, {accepted_count} accepted changes, "
        f"PBC={'+'.join(sorted(pbc_sources))}. No energies were calculated."
    )
    print(f"Electrostatic penalty: {'enabled' if args.electrostatic else 'disabled'}.")
    print(f"Polar penalty: {'enabled' if args.polar else 'disabled'}.")


if __name__ == "__main__":
    main()
