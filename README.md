# Constant-pH Monte Carlo on a molecular trajectory

This directory contains a workflow for sampling protonation states along the
frames of a DCD trajectory. At every attempt, an ionizable residue is selected,
a charge flip is proposed, and the change is accepted or rejected using the
Metropolis criterion. Accepted states persist across attempts and frames. A
per-frame Debye--Huckel energy can also be calculated from those states.

The code **does not modify coordinates or write a new trajectory**. Here, a
"perturbation" means changing the charge state associated with a residue. The
results are written to CSV tables for later analysis or as input to another FEP
workflow.

## Contents

| File | Purpose |
|---|---|
| `Montecarlo_2.py` | Protein model, residue selection, energy terms, and Metropolis acceptance. |
| `periodic_neighborhood.py` | Minimum-image neighborhoods that apply the DCD periodic cell to Monte Carlo calculations. |
| `montecarlo_pH_attempt_md_dcd.py` | Single-stage workflow that samples protonation states and calculates final energies. |
| `montecarlo_questions_md_dcd.py` | Stage 1 of the split workflow: samples and saves attempts only. |
| `debye_huckel_from_mc_csv.py` | Stage 2 of the split workflow: reconstructs visited states and calculates energies. |
| `montecarlo_pH_out.csv` | Example final states and per-frame energies. |
| `montecarlo_pH_attempts.csv` | Example Monte Carlo attempt history. |

## Requirements

- Python 3.10 or later (the code uses `|` in type annotations).
- NumPy.
- MDTraj.
- A PDB file defining the topology.
- A DCD file with the same atom count and atom order as the PDB.

Minimal installation in a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy mdtraj
```

Residue numbers (`resSeq`) must be unique across the entire topology, including
different chains. The program stops if duplicate numbers are found. MDTraj
provides coordinates in nm, and every internal distance in the DCD workflow is
interpreted in that unit.

## Quick start: sampling and energy in one stage

Run from this directory:

```bash
python montecarlo_pH_attempt_md_dcd.py \
  --pdb structure.pdb \
  --dcd trajectory.dcd \
  --ph 7.0 \
  --attempts-per-frame 100 \
  --seed 123 \
  --pbc-mode none \
  --output final_states.csv \
  --attempts-output attempts.csv
```

This example uses `--pbc-mode none`, so it runs without periodic boundary
conditions and calculates direct Euclidean distances between atoms. Remove that
option to use the default `auto` mode, or select another mode as described
below.

The script reads the DCD in chunks so the entire trajectory is not loaded into
memory. Its options are:

| Option | Default | Description |
|---|---:|---|
| `--pdb` | `16_allatoms.pdb` | PDB topology. |
| `--dcd` | `16_allatoms_wrapped.dcd` | DCD trajectory. |
| `--ph` | `7.0` | pH used in the chemical term. |
| `--attempts-per-frame` | `10` | Sequential attempts per frame. |
| `--seed` | no seed | Python and NumPy seed for reproducibility. |
| `--chunk` | `100` | Number of frames read per chunk. |
| `--pbc-mode` | `auto` | Source of the periodic box. |
| `--box-lengths LX LY LZ` | none | User-supplied orthorhombic box in nm. |
| `--bounds-padding` | `0.1` | Per-side padding for a box estimated from coordinates, in nm. |
| `--output` | `montecarlo_pH_out.csv` | Final state and energy per frame. |
| `--attempts-output` | `montecarlo_pH_attempts.csv` | Complete proposal history. |

The default input files are not included in this directory, so `--pdb` and
`--dcd` will usually need to be provided.

## Selecting periodic boundary conditions

`--pbc-mode` controls how cell vectors are obtained. The same choice is applied
to both Monte Carlo attempts and the Debye--Huckel calculation.

| Mode | Behavior |
|---|---|
| `auto` | Uses the box stored in the DCD first, then `--box-lengths`, and finally estimates a box from the coordinate bounds. |
| `trajectory` | Requires a valid DCD box and fails if one is unavailable. |
| `box` | Requires `--box-lengths LX LY LZ` and builds a fixed orthorhombic box. |
| `bounds` | Calculates `L = max(xyz) - min(xyz) + 2*padding` for every frame. |
| `none` | Disables periodic boundary conditions and uses direct Euclidean distances. |

Examples:

```bash
# Use only the cell stored in the DCD
python montecarlo_pH_attempt_md_dcd.py --pdb structure.pdb \
  --dcd trajectory.dcd --pbc-mode trajectory

# Fixed 10 x 10 x 12 nm orthorhombic box
python montecarlo_pH_attempt_md_dcd.py --pdb structure.pdb \
  --dcd trajectory.dcd --pbc-mode box --box-lengths 10 10 12

# Infer a box per frame and add 0.2 nm on each side
python montecarlo_pH_attempt_md_dcd.py --pdb structure.pdb \
  --dcd trajectory.dcd --pbc-mode bounds --bounds-padding 0.2

# Ignore periodic boundary conditions entirely
python montecarlo_pH_attempt_md_dcd.py --pdb structure.pdb \
  --dcd trajectory.dcd --pbc-mode none
```

The `bounds` mode is an approximation: it measures the extent occupied by the
atoms, which is not necessarily the original physical box. It is more
reasonable for explicit solvent filling the cell and can produce artifacts if
the DCD contains only the protein. Padding prevents extreme atoms from being
identified as exact periodic copies.

## Recommended split sampling and energy workflow

Splitting the stages allows energies to be calculated after sampling and makes
it possible to analyze both the final state and the average of visited states.

### 1. Generate protonation states

```bash
python montecarlo_questions_md_dcd.py \
  --pdb structure.pdb \
  --dcd trajectory.dcd \
  --ph 7.0 \
  --attempts-per-frame 100 \
  --seed 123 \
  --output final_states.csv \
  --attempts-output attempts.csv
```

This program explicitly requires all four input/output paths. The `--no-polar`
and `--no-electrostatic` options independently disable those acceptance terms:

```bash
python montecarlo_questions_md_dcd.py \
  --pdb structure.pdb --dcd trajectory.dcd \
  --output final_states.csv --attempts-output attempts.csv \
  --no-electrostatic
```

### 2. Calculate energies of visited states

```bash
python debye_huckel_from_mc_csv.py \
  --pdb structure.pdb \
  --dcd trajectory.dcd \
  --charges-csv final_states.csv \
  --attempts-csv attempts.csv \
  --output energies.csv
```

The second stage checks that the DCD and both CSV files contain the same frames,
that attempts are ordered, and that each reconstructed state matches its saved
final state. Pass the same `--pbc-mode`, `--box-lengths`, and `--bounds-padding`
options used during sampling to evaluate the same periodic geometry.

## Protonation model

The allowed titratable residues and charges are:

| Type | Residues | Charged state | Neutral state |
|---|---|---:|---:|
| Acidic | ASP, GLU, CYS, TYR, CTR | -1 | 0 |
| Basic | ARG, HIS, LYS, NTR | +1 | 0 |

All these residues start in their charged state. Each attempt chooses one at
random and proposes switching between its charged and neutral states. The base
(reference) pKa values are ASP 4.0, GLU 4.5, HIS 6.4, CYS 8.3, TYR 11.0, LYS
10.6, ARG 12.0, NTR 7.5, and CTR 3.5. Effective pKa values can shift depending
on the local protein environment, including electrostatic interactions, solvent
exposure, hydrogen bonding, and nearby charged or polar residues.

The energy change of a proposal is

```text
Delta E = Delta E_pH + Delta E_electrostatic + Delta E_polar
```

with

```text
Delta E_pH = Delta q (pH - pKa) kB T ln(10)
```

where `T = 300 K` and `kB = 0.001987 kcal mol^-1 K^-1`. The electrostatic
contribution is a screened interaction with a 1 nm screening length; the polar
contribution depends on the number and type of neighbors. A proposal with
`Delta E < 0` is always accepted; otherwise, it is accepted with probability
`exp(-Delta E / kB T)`.

Each residue is represented geometrically by `CB`, falling back to `CA` and then
`O`. With valid DCD cell vectors, neighborhood distances use the minimum-image
convention. The Monte Carlo criterion and Debye--Huckel energy use the same
fractional-coordinate transformation. Without a cell, both use direct
distances.

## Reported Debye--Huckel energy

The post-sampling energy differs from `Delta E`: it is an output quantity for
the complete state,

```text
E_DH = 0.5 sum(i<j) [q_i q_j / r_ij] exp(-r_ij / 1 nm)
```

Only charged pairs with `0 < r_ij < 3 nm` are included. With valid DCD cell
vectors, this calculation uses the minimum-image convention; otherwise, it
uses direct distances. The result is labeled `kcal/mol`, following the fixed
prefactor implemented in the scripts.

## Output formats

### Final state per frame

The single-stage workflow produces:

```text
frame,attempted_residue,accepted,debye_huckel_energy_kcal_mol,debye_huckel_visited_mean_kcal_mol,visited_states,ASP1,GLU3,...
```

`debye_huckel_energy_kcal_mol` is the final-state energy, while
`debye_huckel_visited_mean_kcal_mol` is the mean energy after every attempt in
the frame. Rejected attempts are included and repeat the previous state's
energy. `visited_states` equals `--attempts-per-frame`; the initial state before
the first attempt is not an additional observation.

The first stage of the split workflow omits energy columns. In both workflows,
`attempted_residue` and `accepted` describe **only the final attempt in the
frame**. Columns such as `ASP1` and `GLU3` contain the final state of every
titratable residue after all attempts in that frame.

### Attempt history

```text
frame,attempt,residue,old_charge,proposed_charge,accepted,resulting_charge
0,1,ASP41,-1.0,0.0,0,-1.0
```

- `attempt` starts at 1 within each frame.
- `accepted` is 1 for an accepted proposal and 0 for a rejected proposal.
- On rejection, `resulting_charge` must equal `old_charge`.

### Split-workflow energy

```text
frame,debye_huckel_final_kcal_mol,debye_huckel_visited_mean_kcal_mol,visited_states
```

- `debye_huckel_final_kcal_mol`: energy at the end of the frame.
- `debye_huckel_visited_mean_kcal_mol`: mean energy after every attempt,
  including rejected attempts that repeat the previous state.
- `visited_states`: number of attempts included in the mean.

## Reproducibility and considerations

- Always use `--seed` to reproduce the exact Monte Carlo sequence with the same
  code version and input files.
- Increasing `--attempts-per-frame` improves sampling within each geometry but
  also lets the state evolve further before the next frame.
- States are not reset between frames: the first attempt in a frame starts from
  the final state of the previous frame.
- `NTR` and `CTR` are recognized only when they appear as residue names in the
  topology; the code does not create titratable termini automatically.
- The example CSV files can be large because they contain one column per
  titratable residue.

To inspect the exact command-line interface:

```bash
python montecarlo_pH_attempt_md_dcd.py --help
python montecarlo_questions_md_dcd.py --help
python debye_huckel_from_mc_csv.py --help
```
