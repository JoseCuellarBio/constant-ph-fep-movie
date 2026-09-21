#!/usr/bin/env python3
"""Realiza intentos de protonacion Monte Carlo sobre una trayectoria DCD."""

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
    parser = argparse.ArgumentParser(
        description=(
            "Lee una trayectoria DCD, hace uno o más intentos Monte Carlo por "
            "frame y guarda el estado de carga de los residuos ionizables."
        )
    )
    parser.add_argument("--pdb", default="16_allatoms.pdb", help="Topologia PDB")
    parser.add_argument(
        "--dcd", default="16_allatoms_wrapped.dcd", help="Trayectoria DCD"
    )
    parser.add_argument(
        "--output", default="montecarlo_pH_out.csv", help="Archivo CSV de salida"
    )
    parser.add_argument(
        "--attempts-output",
        default="montecarlo_pH_attempts.csv",
        help="CSV con una fila por intento Monte Carlo",
    )
    parser.add_argument("--ph", type=float, default=7.0, help="pH (default: 7.0)")
    parser.add_argument(
        "--seed", type=int, default=None, help="Semilla aleatoria reproducible"
    )
    parser.add_argument(
        "--chunk", type=int, default=100, help="Frames leidos por bloque"
    )
    parser.add_argument(
        "--attempts-per-frame",
        type=int,
        default=10,
        help="Intentos Monte Carlo por frame (default: 10)",
    )
    add_pbc_arguments(parser)
    return parser.parse_args()


def topology_rows(topology: md.Topology, xyz: np.ndarray) -> list[dict]:
    """Convierte un frame de MDTraj al formato que espera Protein."""
    rows = []
    for atom, position in zip(topology.atoms, xyz):
        residue = atom.residue
        rows.append(
            {
                "residue_number": residue.resSeq,
                "residue_name": residue.name,
                "atom": atom.name,
                "posicion_xyz": position,
            }
        )
    return rows


def validate_topology(topology: md.Topology) -> None:
    residue_numbers = [residue.resSeq for residue in topology.residues]
    if len(residue_numbers) != len(set(residue_numbers)):
        raise ValueError(
            "El PDB repite numeros de residuo entre cadenas. Montecarlo_2.py usa "
            "el numero como identificador unico; renumere el PDB antes de continuar."
        )


def debye_huckel_energy_kcal(
    protein: Protein, box_vectors: np.ndarray | None = None
) -> float:
    """Energia total de pH_debyeHuckelTerms.py, expresada en kcal/mol.

    Usa longitud de apantallamiento de 1 nm, cutoff de 3 nm y el prefactor
    (5 * 4.184) * 0.1 de la fuerza OpenMM original.
    """
    charges = protein.representative_charges
    charged = np.flatnonzero(charges != 0.0)
    if charged.size < 2:
        return 0.0

    i, j = np.triu_indices(charged.size, k=1)
    selected_coords = protein.representative_coords[charged]
    displacements = selected_coords[j] - selected_coords[i]

    # Misma convencion periodica que CutoffPeriodic de OpenMM cuando el DCD
    # contiene una celda; en ausencia de celda se usan distancias directas.
    if box_vectors is not None and np.all(np.isfinite(box_vectors)):
        fractional = displacements @ np.linalg.inv(box_vectors)
        fractional -= np.rint(fractional)
        displacements = fractional @ box_vectors

    distances = np.linalg.norm(displacements, axis=1)
    valid = (distances > 0.0) & (distances < 3.0)
    if not np.any(valid):
        return 0.0

    pair_charges = charges[charged[i[valid]]] * charges[charged[j[valid]]]
    # El 4.184 convierte el prefactor original de kJ/mol a kcal/mol.
    prefactor_kcal_nm = (5.0 * 4.184) * 0.1 / 4.184
    return float(
        prefactor_kcal_nm
        * np.sum(pair_charges / distances[valid] * np.exp(-distances[valid]))
    )


def debye_huckel_interaction_matrix(
    coords: np.ndarray, box_vectors: np.ndarray | None = None
) -> np.ndarray:
    """Matriz simetrica para actualizar la energia tras cada cambio de carga."""
    displacement = coords[None, :, :] - coords[:, None, :]
    if box_vectors is not None:
        fractional = displacement @ np.linalg.inv(box_vectors)
        fractional -= np.rint(fractional)
        displacement = fractional @ box_vectors
    distances = np.linalg.norm(displacement, axis=2)
    matrix = np.zeros(distances.shape, dtype=float)
    valid = (distances > 0.0) & (distances < 3.0)
    matrix[valid] = 0.5 * np.exp(-distances[valid]) / distances[valid]
    return matrix


def main() -> None:
    args = parse_args()
    if args.chunk < 1:
        raise ValueError("--chunk debe ser mayor que cero")
    if args.attempts_per_frame < 1:
        raise ValueError("--attempts-per-frame debe ser mayor que cero")

    pdb_path = Path(args.pdb)
    dcd_path = Path(args.dcd)
    output_path = Path(args.output)
    attempts_output_path = Path(args.attempts_output)
    for path in (pdb_path, dcd_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    topology = md.load_topology(str(pdb_path))
    validate_topology(topology)

    # MDTraj y Montecarlo_2 trabajan aqui en nm. Se inicializa Protein con el
    # primer frame real del DCD, no con las coordenadas posiblemente distintas
    # del PDB.
    first = md.load_frame(str(dcd_path), 0, top=str(pdb_path))
    rows = topology_rows(topology, first.xyz[0])
    protein = Protein(rows, pH=args.ph)
    protein.neighborhood = PeriodicNeighborhood(protein)

    ionizable = [
        (resid, info["resname"])
        for resid, info in protein.data.items()
        if info["type"] in ("A", "B")
    ]
    if not ionizable:
        raise ValueError("El PDB no contiene residuos ionizables segun Montecarlo_2.py")

    charge_state = protein.list_charged_residues.copy()
    columns = [f"{resname}{resid}" for resid, resname in ionizable]

    frame_number = 0
    accepted_count = 0
    pbc_sources = set()
    with output_path.open("w", newline="") as output, attempts_output_path.open(
        "w", newline=""
    ) as attempts_output:
        writer = csv.writer(output, delimiter=",", lineterminator="\n")
        writer.writerow(
            [
                "frame",
                "attempted_residue",
                "accepted",
                "debye_huckel_energy_kcal_mol",
                "debye_huckel_visited_mean_kcal_mol",
                "visited_states",
                *columns,
            ]
        )
        attempts_writer = csv.writer(
            attempts_output, delimiter=",", lineterminator="\n"
        )
        attempts_writer.writerow(
            [
                "frame",
                "attempt",
                "residue",
                "old_charge",
                "proposed_charge",
                "accepted",
                "resulting_charge",
            ]
        )

        for chunk in md.iterload(
            str(dcd_path), top=str(pdb_path), chunk=args.chunk
        ):
            for local_frame, xyz in enumerate(chunk.xyz):
                for row, position in zip(rows, xyz):
                    row["posicion_xyz"] = position
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

                interaction = debye_huckel_interaction_matrix(
                    protein.representative_coords, box_vectors
                )
                energy_charges = protein.representative_charges.copy()
                current_energy = float(
                    0.5 * energy_charges @ interaction @ energy_charges
                )
                visited_energy_sum = 0.0

                # Los intentos son secuenciales: cada uno parte del estado dejado
                # por el anterior. Solo se registra el resultado del ultimo.
                for attempt_number in range(1, args.attempts_per_frame + 1):
                    attempted_resid = protein.mc.choose_residue()
                    old_state = dict(charge_state)
                    old_charge = old_state[attempted_resid]
                    residue_info = protein.data[attempted_resid]
                    acid_base = -1 if residue_info["type"] == "A" else 1
                    proposed_charge, _ = calculate_new_charge(
                        attempted_resid, acid_base, old_charge
                    )
                    charge_state, particle_info = (
                        protein.protonation_mc.attempt_charge_flip(charge_state)
                    )
                    accepted = particle_info is not None
                    accepted_count += int(accepted)
                    protein.refresh_charge_state(charge_state)
                    resulting_charge = dict(charge_state)[attempted_resid]
                    representative_index = (
                        protein._representative_resid_to_index.get(attempted_resid)
                    )
                    if representative_index is not None:
                        delta_charge = (
                            resulting_charge - energy_charges[representative_index]
                        )
                        if delta_charge:
                            current_energy += delta_charge * float(
                                interaction[representative_index] @ energy_charges
                            )
                            energy_charges[representative_index] = resulting_charge
                    visited_energy_sum += current_energy
                    attempts_writer.writerow(
                        [
                            frame_number,
                            attempt_number,
                            f"{residue_info['resname']}{attempted_resid}",
                            f"{old_charge:.1f}",
                            f"{proposed_charge:.1f}",
                            int(accepted),
                            f"{resulting_charge:.1f}",
                        ]
                    )

                state = dict(charge_state)
                dh_energy = debye_huckel_energy_kcal(protein, box_vectors)
                if not np.isclose(current_energy, dh_energy):
                    raise RuntimeError(
                        "La actualizacion incremental de energia no coincide "
                        "con el calculo final"
                    )
                visited_mean = visited_energy_sum / args.attempts_per_frame
                attempted_label = (
                    f"{protein.data[attempted_resid]['resname']}{attempted_resid}"
                    if attempted_resid is not None
                    else "NA"
                )
                writer.writerow(
                    [
                        frame_number,
                        attempted_label,
                        int(accepted and state != old_state),
                        f"{dh_energy:.8f}",
                        f"{visited_mean:.8f}",
                        args.attempts_per_frame,
                        *(f"{state[resid]:.1f}" for resid, _ in ionizable),
                    ]
                )
                frame_number += 1

    print(
        f"Listo: {frame_number} frames, "
        f"{frame_number * args.attempts_per_frame} intentos, "
        f"{accepted_count} cambios aceptados, "
        f"PBC={'+'.join(sorted(pbc_sources))}, "
        f"salidas: {output_path} y {attempts_output_path}"
    )


if __name__ == "__main__":
    main()
