import os
import shutil
import tempfile
import logging
import numpy as np
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Vina 是可选依赖
try:
    from vina import Vina as _Vina
    VINA_AVAILABLE = True
except ImportError:
    _Vina = None  # type: ignore[assignment]
    VINA_AVAILABLE = False
    logger.info("vina package not installed, docking will be unavailable")

# RDKit 是必须依赖，缺失时启动即报错
from rdkit import Chem
from rdkit.Chem import AllChem, rdPartialCharges, rdMolDescriptors
RDKIT_AVAILABLE = True


_ATOM_TYPE_MAP = {
    'C': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'CZ': 'C', 'CH': 'C',
    'CA': 'C',  # Cα carbon
    'N': 'NA', 'NZ': 'N', 'NE': 'N', 'NH': 'N', 'ND': 'N',
    'O': 'OA', 'OD': 'OA', 'OE': 'OA', 'OG': 'OA', 'OH': 'OA',
    'S': 'SA', 'SD': 'SA', 'SG': 'SA',
    'H': 'HD', 'HN': 'HD',
    'FE': 'Fe', 'ZN': 'Zn', 'MG': 'Mg', 'MN': 'Mn', 'CAL': 'Ca',  # CAL = calcium
}

_RECEPTOR_ATOM_TYPES = {
    'ALA': {'C': 'C', 'CA': 'C', 'CB': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'ARG': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'CZ': 'C', 'NE': 'NA', 'NH1': 'NA', 'NH2': 'NA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'ASN': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'ND2': 'NA', 'OD1': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'ASP': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'OD1': 'OA', 'OD2': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'CYS': {'C': 'C', 'CA': 'C', 'CB': 'C', 'SG': 'SA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'GLN': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'NE2': 'NA', 'OE1': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'GLU': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'OE1': 'OA', 'OE2': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'GLY': {'C': 'C', 'CA': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'HIS': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'ND1': 'NA', 'CD2': 'C', 'CE1': 'C', 'NE2': 'NA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'ILE': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG1': 'C', 'CG2': 'C', 'CD1': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'LEU': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD1': 'C', 'CD2': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'LYS': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'CE': 'C', 'NZ': 'N', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'MET': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'SD': 'SA', 'CE': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'PHE': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD1': 'C', 'CD2': 'C', 'CE1': 'C', 'CE2': 'C', 'CZ': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'PRO': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'SER': {'C': 'C', 'CA': 'C', 'CB': 'C', 'OG': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'THR': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG2': 'C', 'OG1': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'TRP': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD1': 'C', 'CD2': 'C', 'NE1': 'NA', 'CE2': 'C', 'CE3': 'C', 'CZ2': 'C', 'CZ3': 'C', 'CH2': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'TYR': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG': 'C', 'CD1': 'C', 'CD2': 'C', 'CE1': 'C', 'CE2': 'C', 'CZ': 'C', 'OH': 'OA', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
    'VAL': {'C': 'C', 'CA': 'C', 'CB': 'C', 'CG1': 'C', 'CG2': 'C', 'N': 'NA', 'O': 'OA', 'OXT': 'OA'},
}


def _ad4_type_for_receptor_atom(res_name: str, atom_name: str) -> str:
    res_map = _RECEPTOR_ATOM_TYPES.get(res_name, {})
    ad4 = res_map.get(atom_name)
    if ad4:
        return ad4
    if atom_name.startswith('C'):
        return 'C'
    if atom_name.startswith('N'):
        return 'NA'
    if atom_name.startswith('O'):
        return 'OA'
    if atom_name.startswith('S'):
        return 'SA'
    if atom_name.startswith('H'):
        return 'HD'
    return 'C'


def prepare_receptor(pdb_path: str, output_pdbqt: Optional[str] = None) -> str:
    """Convert a PDB file to PDBQT format (receptor).

    Steps:
      1. Read PDB, keep only ATOM records
      2. Remove water (HOH) and non-standard hetero-atoms
      3. Remove existing hydrogens (will be re-added as polar H)
      4. Assign AD4 atom types and partial charges (Gasteiger-like)
      5. Write PDBQT

    Returns:
        Path to the generated PDBQT file
    """
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_path}")

    atoms = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            res_name = line[17:20].strip()
            if res_name in ('HOH', 'WAT', 'TIP', 'H2O'):
                continue
            atom_name = line[12:16].strip()
            if atom_name.startswith('H') and len(atom_name) <= 2:
                continue
            chain_id = line[21]
            res_num = int(line[22:26].strip())
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            element = atom_name[0]
            ad4_type = _ad4_type_for_receptor_atom(res_name, atom_name)
            atoms.append({
                'serial': len(atoms) + 1,
                'atom_name': atom_name,
                'res_name': res_name,
                'chain_id': chain_id,
                'res_num': res_num,
                'x': x, 'y': y, 'z': z,
                'element': element,
                'ad4_type': ad4_type,
                'charge': 0.0,
            })

    _assign_polar_hydrogens(atoms)

    if output_pdbqt is None:
        base = os.path.splitext(pdb_path)[0]
        output_pdbqt = base + '.pdbqt'

    with open(output_pdbqt, 'w') as f:
        for a in atoms:
            name = a['atom_name']
            if len(name) <= 3:
                padded_name = ' ' + name.ljust(3)
            else:
                padded_name = name[:4].ljust(4)
            f.write(
                f"ATOM  {a['serial']:5d} {padded_name} "
                f"{a['res_name']:>3s} {a['chain_id']}{a['res_num']:4d}    "
                f"{a['x']:8.3f}{a['y']:8.3f}{a['z']:8.3f}"
                f"{1.0:6.2f}{0.0:6.2f}    "
                f"{a['charge']:+.4f} {a['ad4_type']}\n"
            )
        f.write("END\n")

    logger.info(f"Receptor PDBQT written: {output_pdbqt} ({len(atoms)} atoms)")
    return output_pdbqt


def _assign_polar_hydrogens(atoms: list):
    """Annotate polar hydrogen atoms on the receptor (adds HD type to backbone NH)."""
    backbone_n = [a for a in atoms if a['atom_name'] == 'N' and a['res_name'] != 'PRO']
    for n_atom in backbone_n:
        h_name = 'H'
        has_h = any(
            a['atom_name'] == h_name and a['res_num'] == n_atom['res_num'] and a['chain_id'] == n_atom['chain_id']
            for a in atoms
        )
        if not has_h:
            nx, ny, nz = n_atom['x'], n_atom['y'], n_atom['z']
            ca_candidates = [
                a for a in atoms
                if a['atom_name'] == 'CA' and a['res_num'] == n_atom['res_num'] and a['chain_id'] == n_atom['chain_id']
            ]
            if ca_candidates:
                ca = ca_candidates[0]
                dx = nx - ca['x']
                dy = ny - ca['y']
                dz = nz - ca['z']
                dist = (dx**2 + dy**2 + dz**2) ** 0.5
                if dist > 0:
                    bond_len = 1.01
                    hx = nx + dx / dist * bond_len
                    hy = ny + dy / dist * bond_len
                    hz = nz + dz / dist * bond_len
                    atoms.append({
                        'serial': len(atoms) + 1,
                        'atom_name': 'H   ',
                        'res_name': n_atom['res_name'],
                        'chain_id': n_atom['chain_id'],
                        'res_num': n_atom['res_num'],
                        'x': hx, 'y': hy, 'z': hz,
                        'element': 'H',
                        'ad4_type': 'HD',
                        'charge': 0.0,
                    })


def prepare_ligand(smiles: str, ligand_name: str = "ligand",
                   output_pdbqt: Optional[str] = None,
                   work_dir: Optional[str] = None) -> str:
    """Convert a SMILES string to a PDBQT file (ligand).

    Steps:
      1. RDKit: SMILES → Mol → add H → EmbedMolecule → MMFF optimize
      2. Calculate Gasteiger partial charges
      3. Write PDBQT with rotatable bonds marked

    Returns:
        Path to the generated PDBQT file
    """
    if not RDKIT_AVAILABLE:
        raise ImportError("RDKit is required for ligand preparation")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result != 0:
        # ETKDGv3 failed — fall back to distance geometry with random coords
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    if result != 0:
        raise ValueError(f"3D conformer generation failed for SMILES: {smiles}")
    ff_result = AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    if ff_result == -1:
        # MMFF not available for this molecule, try UFF
        AllChem.UFFOptimizeMolecule(mol, maxIters=2000)

    rdPartialCharges.ComputeGasteigerCharges(mol)

    if work_dir is None:
        work_dir = tempfile.mkdtemp(prefix='vina_lig_')
    os.makedirs(work_dir, exist_ok=True)

    if output_pdbqt is None:
        output_pdbqt = os.path.join(work_dir, f"{ligand_name}.pdbqt")

    _write_ligand_pdbqt(mol, output_pdbqt, ligand_name)
    logger.info(f"Ligand PDBQT written: {output_pdbqt}")
    return output_pdbqt


def _get_ad4_ligand_type(atom) -> str:
    symbol = atom.GetSymbol()
    if symbol == 'C':
        return 'A' if atom.GetIsAromatic() else 'C'
    if symbol == 'N':
        return 'NA'
    if symbol == 'O':
        return 'OA'
    if symbol == 'S':
        return 'SA'
    if symbol == 'H':
        return 'HD'
    if symbol == 'F':
        return 'F'
    if symbol == 'Cl':
        return 'Cl'
    if symbol == 'Br':
        return 'Br'
    if symbol == 'I':
        return 'I'
    if symbol == 'P':
        return 'P'
    return 'C'


def _find_torsions(mol) -> list:
    rot_bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() == 1.0 and not bond.IsInRing():
            a1 = bond.GetBeginAtomIdx()
            a2 = bond.GetEndAtomIdx()
            if mol.GetAtomWithIdx(a1).GetDegree() > 1 and mol.GetAtomWithIdx(a2).GetDegree() > 1:
                rot_bonds.append((a1, a2))
    return rot_bonds


def _get_torsion_tree(mol, rot_bonds: list) -> dict:
    from collections import defaultdict

    adj = defaultdict(set)
    for bond in mol.GetBonds():
        a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[a1].add(a2)
        adj[a2].add(a1)

    rot_set = set()
    for a1, a2 in rot_bonds:
        rot_set.add((min(a1, a2), max(a1, a2)))

    def is_rotatable(a, b):
        return (min(a, b), max(a, b)) in rot_set

    visited = set()

    def build_subtree(root_idx):
        visited.add(root_idx)
        node = {'atoms': [root_idx], 'branches': []}
        for neighbor in sorted(adj[root_idx]):
            if neighbor in visited:
                continue
            if is_rotatable(root_idx, neighbor):
                subtree = build_subtree(neighbor)
                node['branches'].append({
                    'bond': (root_idx, neighbor),
                    'tree': subtree,
                })
            else:
                sub = build_subtree(neighbor)
                node['atoms'].extend(sub['atoms'])
                node['branches'].extend(sub['branches'])
        return node

    return build_subtree(0)


def _write_ligand_pdbqt(mol, filepath: str, name: str = "ligand"):
    conf = mol.GetConformer()
    rot_bonds = _find_torsions(mol)

    lines = []
    lines.append(f"REMARK  Name: {name}")
    lines.append(f"REMARK  SMILES: {Chem.MolToSmiles(mol)}")
    lines.append(f"REMARK  Rotatable bonds: {len(rot_bonds)}")
    lines.append("")

    atom_info = []
    for i in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(i)
        pos = conf.GetAtomPosition(i)
        charge = float(atom.GetDoubleProp('_GasteigerCharge'))
        if charge != charge:  # NaN check
            charge = 0.0
        ad4 = _get_ad4_ligand_type(atom)
        symbol = atom.GetSymbol()
        atom_info.append({
            'idx': i,
            'name': f"{symbol}{i}",
            'symbol': symbol,
            'x': pos.x, 'y': pos.y, 'z': pos.z,
            'charge': charge,
            'ad4_type': ad4,
        })

    def write_atom_line(info, serial):
        return (
            f"ATOM  {serial:5d} {info['name']:<4s} "
            f"UNL     1    "
            f"{info['x']:8.3f}{info['y']:8.3f}{info['z']:8.3f}"
            f"{1.0:6.2f}{0.0:6.2f}    "
            f"{info['charge']:+.4f} {info['ad4_type']}"
        )

    if rot_bonds:
        tree = _get_torsion_tree(mol, rot_bonds)
        serial = [1]
        idx2serial = {}

        def emit_branch(node):
            out = []
            for idx in node['atoms']:
                idx2serial[idx] = serial[0]
                out.append(write_atom_line(atom_info[idx], serial[0]))
                serial[0] += 1
            for branch in node['branches']:
                parent_idx = branch['bond'][0]
                parent_serial = idx2serial[parent_idx]
                child_serial = serial[0]
                out.append(f"BRANCH {parent_serial:4d} {child_serial:4d}")
                out.extend(emit_branch(branch['tree']))
                out.append(f"ENDBRANCH {parent_serial:4d} {child_serial:4d}")
            return out

        lines.append("ROOT")
        for idx in tree['atoms']:
            idx2serial[idx] = serial[0]
            lines.append(write_atom_line(atom_info[idx], serial[0]))
            serial[0] += 1
        lines.append("ENDROOT")

        for branch in tree['branches']:
            parent_idx = branch['bond'][0]
            parent_serial = idx2serial[parent_idx]
            child_serial = serial[0]
            lines.append(f"BRANCH {parent_serial:4d} {child_serial:4d}")
            lines.extend(emit_branch(branch['tree']))
            lines.append(f"ENDBRANCH {parent_serial:4d} {child_serial:4d}")
    else:
        for i, info in enumerate(atom_info):
            lines.append(write_atom_line(info, i + 1))

    lines.append("TORSDOF %d" % len(rot_bonds))

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def calculate_docking_box(residue_coords: List[List[float]],
                          padding: float = 14.0,
                          min_size: float = 22.0,
                          max_size: float = 60.0) -> Dict:
    """Calculate docking box center and size from binding site residue coordinates.

    Args:
        residue_coords: List of [x, y, z] coordinate lists for each binding site residue
        padding: Extra space (Angstroms) around the bounding box. Default 14 Å accounts
                 for Cα-to-sidechain extension (~5 Å) plus ligand radius (~9 Å).
        min_size: Minimum box dimension
        max_size: Maximum box dimension

    Returns:
        Dict with 'center' [x,y,z] and 'box_size' [sx,sy,sz]
    """
    coords = np.array(residue_coords)
    if len(coords) == 0:
        raise ValueError("No residue coordinates provided")

    center = coords.mean(axis=0).tolist()

    if len(coords) == 1:
        size = np.array([padding * 2] * 3)
    else:
        mins = coords.min(axis=0) - padding
        maxs = coords.max(axis=0) + padding
        size = maxs - mins

    size = np.clip(size, min_size, max_size).tolist()

    return {
        "center": [round(c, 3) for c in center],
        "box_size": [round(s, 3) for s in size]
    }


def run_vina_docking(receptor_pdbqt: str,
                     ligand_pdbqt: str,
                     center: List[float],
                     box_size: List[float],
                     exhaustiveness: int = 8,
                     n_poses: int = 10,
                     cpu: int = 0,
                     output_poses_path: Optional[str] = None) -> Dict:
    """Run AutoDock Vina docking.

    Args:
        receptor_pdbqt: Path to receptor PDBQT file
        ligand_pdbqt: Path to ligand PDBQT file
        center: [x, y, z] center of the search box
        box_size: [sx, sy, sz] size of the search box
        exhaustiveness: Exhaustiveness of search
        n_poses: Maximum number of poses to output
        cpu: Number of CPUs (0 = all)

    Returns:
        Dict with docking results
    """
    if not VINA_AVAILABLE or _Vina is None:
        raise ImportError("vina package is not installed. Please install with: pip install vina")

    v = _Vina(sf_name='vina', cpu=cpu)
    v.set_receptor(receptor_pdbqt)
    v.set_ligand_from_file(ligand_pdbqt)

    v.compute_vina_maps(center=center, box_size=box_size)

    # Randomise ligand pose inside the box before scoring/docking so that
    # the initial conformation is guaranteed to be within the search space.
    v.randomize()

    v.dock(exhaustiveness=exhaustiveness, n_poses=n_poses)

    if output_poses_path:
        v.write_poses(output_poses_path, n_poses=n_poses, overwrite=True)

    energies = v.energies(n_poses=n_poses)
    all_scores = []
    if energies is not None and len(energies) > 0:
        for row in energies:
            all_scores.append({
                'total': float(row[0]),
                'inter': float(row[1]),
                'intra': float(row[2]),
                'torsional': float(row[3]),
            })

    best_score = all_scores[0]['total'] if all_scores else 0.0

    return {
        'success': True,
        'best_score': round(best_score, 3),
        'n_poses': len(all_scores),
        'poses': all_scores[:n_poses],
        'energies_raw': energies.tolist() if energies is not None else [],
    }


def vina_dock_single(pdb_path: str,
                     smiles: str,
                     ligand_name: str = "ligand",
                     center: Optional[List[float]] = None,
                     box_size: Optional[List[float]] = None,
                     binding_site_coords: Optional[List[List[float]]] = None,
                     padding: float = 14.0,
                     exhaustiveness: int = 8,
                     n_poses: int = 10) -> Dict:
    """All-in-one docking: prepare receptor + ligand + compute box + dock.

    Args:
        pdb_path: Path to receptor PDB file
        smiles: SMILES string for the ligand
        ligand_name: Name for the ligand
        center: Explicit box center [x,y,z] (overrides auto-calculation)
        box_size: Explicit box size [sx,sy,sz] (overrides auto-calculation)
        binding_site_coords: List of [x,y,z] for binding site residues (used to auto-compute box)
        padding: Padding around binding site (Angstroms)
        exhaustiveness: Vina exhaustiveness parameter
        n_poses: Number of poses to generate

    Returns:
        Dict with docking results
    """
    work_dir = tempfile.mkdtemp(prefix='vina_dock_')

    try:
        receptor_pdbqt = prepare_receptor(pdb_path, os.path.join(work_dir, 'receptor.pdbqt'))
        ligand_pdbqt = prepare_ligand(smiles, ligand_name,
                                       os.path.join(work_dir, f'{ligand_name}.pdbqt'),
                                       work_dir)

        if center is None or box_size is None:
            if binding_site_coords is None:
                binding_site_coords = _auto_detect_binding_site(pdb_path)
            if not binding_site_coords:
                shutil.rmtree(work_dir, ignore_errors=True)
                return {
                    'success': False,
                    'error': 'No binding site coordinates provided and auto-detection failed. '
                             'Please provide center/box_size or binding_site_coords.'
                }
            box = calculate_docking_box(binding_site_coords, padding=padding)
            if center is None:
                center = box['center']
            if box_size is None:
                box_size = box['box_size']

        assert center is not None and box_size is not None

        result = run_vina_docking(
            receptor_pdbqt, ligand_pdbqt,
            center=center, box_size=box_size,
            exhaustiveness=exhaustiveness, n_poses=n_poses,
            output_poses_path=os.path.join(work_dir, 'ligand_out.pdbqt')
        )

        result['receptor_pdb'] = pdb_path
        result['smiles'] = smiles
        result['center'] = center
        result['box_size'] = box_size
        result['work_dir'] = work_dir
        return result

    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {
            'success': False,
            'error': str(e)
        }


def _auto_detect_binding_site(pdb_path: str) -> List[List[float]]:
    """Extract binding site coordinates from PDB HETATM ligand atoms.

    Returns all heavy atom coordinates of the co-crystallised ligand so that
    calculate_docking_box can compute a tight bounding box around it.
    Returns empty list if no co-crystallised ligand is found — callers must
    then require the user to supply explicit center/box_size coordinates.
    """
    ligand_coords = []
    with open(pdb_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            atom_name = line[12:16].strip()
            res_name = line[17:20].strip()
            if res_name in ('HOH', 'WAT', 'TIP', 'H2O', 'UNK'):
                continue
            if atom_name.startswith('H'):
                continue
            try:
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                ligand_coords.append([x, y, z])
            except (ValueError, IndexError):
                continue

    return ligand_coords


def vina_dock_batch(pdb_path: str,
                    smiles_list: List[Dict],
                    center: Optional[List[float]] = None,
                    box_size: Optional[List[float]] = None,
                    binding_site_coords: Optional[List[List[float]]] = None,
                    padding: float = 14.0,
                    exhaustiveness: int = 8,
                    n_poses: int = 10,
                    progress_callback=None) -> Dict:
    """Batch docking: prepare receptor once, dock multiple ligands.

    Args:
        pdb_path: Path to receptor PDB file
        smiles_list: List of dicts with 'smiles' and optional 'name' keys
        center: Box center [x,y,z]
        box_size: Box size [sx,sy,sz]
        binding_site_coords: Binding site residue coordinates
        padding: Padding (Angstroms)
        exhaustiveness: Vina exhaustiveness
        n_poses: Number of poses per ligand
        progress_callback: callback(current, total, message)

    Returns:
        Dict with batch docking results
    """
    work_dir = tempfile.mkdtemp(prefix='vina_batch_')

    try:
        receptor_pdbqt = prepare_receptor(pdb_path, os.path.join(work_dir, 'receptor.pdbqt'))
    except Exception as e:
        shutil.rmtree(work_dir, ignore_errors=True)
        return {'success': False, 'error': f'Receptor preparation failed: {e}'}

    if center is None or box_size is None:
        if binding_site_coords is None:
            binding_site_coords = _auto_detect_binding_site(pdb_path)
        if binding_site_coords:
            box = calculate_docking_box(binding_site_coords, padding=padding)
            if center is None:
                center = box['center']
            if box_size is None:
                box_size = box['box_size']
        else:
            shutil.rmtree(work_dir, ignore_errors=True)
            return {'success': False, 'error': 'Cannot determine docking box. Provide center/box_size.'}

    assert center is not None and box_size is not None

    total = len(smiles_list)
    results = []
    success_count = 0

    if progress_callback:
        progress_callback(0, total, f'Preparing receptor PDBQT...')

    for i, mol_info in enumerate(smiles_list):
        smi = mol_info.get('smiles', '') if isinstance(mol_info, dict) else str(mol_info)
        name = mol_info.get('name', f'ligand_{i}') if isinstance(mol_info, dict) else f'ligand_{i}'

        if progress_callback:
            progress_callback(i, total, f'Docking {name} ({i+1}/{total})...')

        try:
            ligand_pdbqt = prepare_ligand(smi, name,
                                           os.path.join(work_dir, f'{name}.pdbqt'),
                                           work_dir)
            vina_result = run_vina_docking(
                receptor_pdbqt, ligand_pdbqt,
                center=center, box_size=box_size,
                exhaustiveness=exhaustiveness, n_poses=n_poses
            )
            vina_result['name'] = name
            vina_result['smiles'] = smi
            results.append(vina_result)
            success_count += 1
        except Exception as e:
            results.append({
                'success': False,
                'name': name,
                'smiles': smi,
                'error': str(e)
            })

    results.sort(key=lambda r: r.get('best_score', 0))

    if progress_callback:
        progress_callback(total, total, 'Docking complete!')

    return {
        'success': True,
        'total': total,
        'docked': success_count,
        'receptor': pdb_path,
        'center': center,
        'box_size': box_size,
        'results': results,
        'best': results[0] if results else None,
    }


class VinaDockingService:
    """AutoDock Vina 对接服务封装"""

    @staticmethod
    def is_available() -> bool:
        return VINA_AVAILABLE

    @staticmethod
    def prepare_receptor(pdb_path: str, output_pdbqt: Optional[str] = None) -> str:
        return prepare_receptor(pdb_path, output_pdbqt)

    @staticmethod
    def prepare_ligand(smiles: str, ligand_name: str = "ligand",
                       output_pdbqt: Optional[str] = None,
                       work_dir: Optional[str] = None) -> str:
        return prepare_ligand(smiles, ligand_name, output_pdbqt, work_dir)

    @staticmethod
    def calculate_docking_box(residue_coords: List[List[float]],
                              padding: float = 14.0,
                              min_size: float = 22.0,
                              max_size: float = 60.0) -> Dict:
        return calculate_docking_box(residue_coords, padding, min_size, max_size)

    @staticmethod
    def run_vina_docking(receptor_pdbqt: str, ligand_pdbqt: str,
                         center: List[float], box_size: List[float],
                         exhaustiveness: int = 8, n_poses: int = 10,
                         cpu: int = 0,
                         output_poses_path: Optional[str] = None) -> Dict:
        return run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, box_size,
                                exhaustiveness, n_poses, cpu, output_poses_path)

    @staticmethod
    def dock_single(pdb_path: str, smiles: str, ligand_name: str = "ligand",
                    center: Optional[List[float]] = None,
                    box_size: Optional[List[float]] = None,
                    binding_site_coords: Optional[List[List[float]]] = None,
                    padding: float = 14.0, exhaustiveness: int = 8,
                    n_poses: int = 10) -> Dict:
        return vina_dock_single(pdb_path, smiles, ligand_name, center, box_size,
                                binding_site_coords, padding, exhaustiveness, n_poses)

    @staticmethod
    def dock_batch(pdb_path: str, smiles_list: List[Dict],
                   center: Optional[List[float]] = None,
                   box_size: Optional[List[float]] = None,
                   binding_site_coords: Optional[List[List[float]]] = None,
                   padding: float = 14.0, exhaustiveness: int = 8,
                   n_poses: int = 10, progress_callback=None) -> Dict:
        return vina_dock_batch(pdb_path, smiles_list, center, box_size,
                               binding_site_coords, padding, exhaustiveness,
                               n_poses, progress_callback)


vina_docking_service = VinaDockingService()

