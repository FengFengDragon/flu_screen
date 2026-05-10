import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist

class ResidueFeatureExtractor:
    """残基特征提取器 - 用于传统机器学习"""
    
    def __init__(self):
        self.amino_acids = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
        
        self.aa_order = list(self.amino_acids.keys())
        
        self.physicochemical = {
            'ALA': {'hydrophobicity': 1.8, 'charge': 0, 'polarity': 0, 'volume': 67},
            'ARG': {'hydrophobicity': -4.5, 'charge': 1, 'polarity': 1, 'volume': 148},
            'ASN': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'volume': 96},
            'ASP': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'volume': 91},
            'CYS': {'hydrophobicity': 2.5, 'charge': 0, 'polarity': 0, 'volume': 86},
            'GLN': {'hydrophobicity': -3.5, 'charge': 0, 'polarity': 1, 'volume': 114},
            'GLU': {'hydrophobicity': -3.5, 'charge': -1, 'polarity': 1, 'volume': 109},
            'GLY': {'hydrophobicity': -0.4, 'charge': 0, 'polarity': 0, 'volume': 48},
            'HIS': {'hydrophobicity': -3.2, 'charge': 0.5, 'polarity': 1, 'volume': 118},
            'ILE': {'hydrophobicity': 4.5, 'charge': 0, 'polarity': 0, 'volume': 124},
            'LEU': {'hydrophobicity': 3.8, 'charge': 0, 'polarity': 0, 'volume': 124},
            'LYS': {'hydrophobicity': -3.9, 'charge': 1, 'polarity': 1, 'volume': 135},
            'MET': {'hydrophobicity': 1.9, 'charge': 0, 'polarity': 0, 'volume': 124},
            'PHE': {'hydrophobicity': 2.8, 'charge': 0, 'polarity': 0, 'volume': 135},
            'PRO': {'hydrophobicity': -1.6, 'charge': 0, 'polarity': 0, 'volume': 90},
            'SER': {'hydrophobicity': -0.8, 'charge': 0, 'polarity': 1, 'volume': 73},
            'THR': {'hydrophobicity': -0.7, 'charge': 0, 'polarity': 1, 'volume': 93},
            'TRP': {'hydrophobicity': -0.9, 'charge': 0, 'polarity': 0, 'volume': 163},
            'TYR': {'hydrophobicity': -1.3, 'charge': 0, 'polarity': 1, 'volume': 141},
            'VAL': {'hydrophobicity': 4.2, 'charge': 0, 'polarity': 0, 'volume': 105}
        }
        
        self.secondary_structure_map = {
            'H': 1,
            'E': 2,
            'C': 0
        }
    
    def encode_amino_acid(self, residue_name: str) -> np.ndarray:
        """One-hot编码氨基酸"""
        encoding = np.zeros(20)
        if residue_name in self.aa_order:
            idx = self.aa_order.index(residue_name)
            encoding[idx] = 1
        return encoding
    
    def get_physicochemical_features(self, residue_name: str) -> np.ndarray:
        """获取物理化学特征"""
        default = {'hydrophobicity': 0, 'charge': 0, 'polarity': 0, 'volume': 100}
        props = self.physicochemical.get(residue_name, default)
        return np.array([
            props['hydrophobicity'],
            props['charge'],
            props['polarity'],
            props['volume']
        ])
    
    def calculate_solvent_accessibility(self, residue_atoms: List[Dict], 
                                       probe_radius: float = 1.4) -> float:
        """计算溶剂可及性（简化版）"""
        if not residue_atoms:
            return 0.0
        
        coords = np.array([atom['coords'] for atom in residue_atoms])
        center = np.mean(coords, axis=0)
        
        distances = np.linalg.norm(coords - center, axis=1)
        avg_radius = np.mean(distances)
        
        return avg_radius
    
    def calculate_b_factor(self, residue_atoms: List[Dict]) -> float:
        """计算B因子平均值"""
        if not residue_atoms:
            return 0.0
        
        b_factors = [atom.get('b_factor', 0) for atom in residue_atoms]
        return np.mean(b_factors) if b_factors else 0.0
    
    def calculate_depth(self, residue_center: np.ndarray, 
                       all_centers: np.ndarray) -> float:
        """计算残基深度（距离蛋白中心的距离）"""
        protein_center = np.mean(all_centers, axis=0)
        return float(np.linalg.norm(residue_center - protein_center))
    
    def count_neighbors(self, residue_center: np.ndarray, 
                       all_centers: np.ndarray, 
                       cutoff: float = 5.0) -> int:
        """计算邻居数量"""
        distances = np.linalg.norm(all_centers - residue_center, axis=1)
        return int(np.sum(distances < cutoff) - 1)
    
    def calculate_neighbor_density(self, residue_center: np.ndarray,
                                   all_centers: np.ndarray,
                                   cutoff: float = 10.0) -> float:
        """计算邻居密度"""
        distances = np.linalg.norm(all_centers - residue_center, axis=1)
        neighbors = np.sum((distances < cutoff) & (distances > 0))
        
        volume = (4/3) * np.pi * (cutoff ** 3)
        return neighbors / volume if volume > 0 else 0
    
    def calculate_hydrophobic_neighbor_ratio(self, residue_center: np.ndarray,
                                             all_centers: np.ndarray,
                                             all_residue_names: List[str],
                                             cutoff: float = 5.0) -> float:
        """计算疏水邻居比例"""
        distances = np.linalg.norm(all_centers - residue_center, axis=1)
        neighbor_indices = np.where((distances < cutoff) & (distances > 0))[0]
        
        if len(neighbor_indices) == 0:
            return 0.0
        
        hydrophobic_count = 0
        for idx in neighbor_indices:
            res_name = all_residue_names[idx]
            props = self.physicochemical.get(res_name, {})
            if props.get('hydrophobicity', 0) > 0:
                hydrophobic_count += 1
        
        return hydrophobic_count / len(neighbor_indices)
    
    def calculate_charged_neighbor_ratio(self, residue_center: np.ndarray,
                                        all_centers: np.ndarray,
                                        all_residue_names: List[str],
                                        cutoff: float = 5.0) -> float:
        """计算带电邻居比例"""
        distances = np.linalg.norm(all_centers - residue_center, axis=1)
        neighbor_indices = np.where((distances < cutoff) & (distances > 0))[0]
        
        if len(neighbor_indices) == 0:
            return 0.0
        
        charged_count = 0
        for idx in neighbor_indices:
            res_name = all_residue_names[idx]
            props = self.physicochemical.get(res_name, {})
            if abs(props.get('charge', 0)) > 0:
                charged_count += 1
        
        return charged_count / len(neighbor_indices)
    
    def predict_secondary_structure(self, residue_name: str, 
                                   context: Optional[List[str]] = None) -> int:
        """预测二级结构（简化版）"""
        if context is None:
            context = []
        
        propensity = {
            'H': ['ALA', 'GLU', 'LEU', 'MET', 'GLN', 'LYS', 'ARG'],
            'E': ['VAL', 'ILE', 'TYR', 'PHE', 'TRP', 'LEU', 'CYS'],
            'C': ['GLY', 'SER', 'PRO', 'ASN', 'ASP', 'THR']
        }
        
        ss_scores = {'H': 0, 'E': 0, 'C': 0}
        
        for ss, aas in propensity.items():
            if residue_name in aas:
                ss_scores[ss] += 1
        
        for neighbor in context:
            for ss, aas in propensity.items():
                if neighbor in aas:
                    ss_scores[ss] += 0.5
        
        if ss_scores['H'] >= ss_scores['E'] and ss_scores['H'] >= ss_scores['C']:
            return self.secondary_structure_map['H']
        elif ss_scores['E'] >= ss_scores['H'] and ss_scores['E'] >= ss_scores['C']:
            return self.secondary_structure_map['E']
        else:
            return self.secondary_structure_map['C']
    
    def extract_residue_features(self, residue_atoms: List[Dict],
                                residue_name: str,
                                residue_id: int,
                                chain_id: str,
                                all_residue_centers: np.ndarray,
                                all_residue_names: List[str],
                                sequence_position: int,
                                total_length: int) -> np.ndarray:
        """提取单个残基的所有特征"""
        
        residue_center = np.mean([atom['coords'] for atom in residue_atoms], axis=0)
        
        features = []
        
        features.append(self.encode_amino_acid(residue_name))
        
        features.append(self.get_physicochemical_features(residue_name))
        
        asa = self.calculate_solvent_accessibility(residue_atoms)
        features.append([asa])
        
        b_factor = self.calculate_b_factor(residue_atoms)
        features.append([b_factor])
        
        depth = self.calculate_depth(residue_center, all_residue_centers)
        features.append([depth])
        
        neighbor_count = self.count_neighbors(residue_center, all_residue_centers)
        features.append([neighbor_count])
        
        neighbor_density = self.calculate_neighbor_density(residue_center, all_residue_centers)
        features.append([neighbor_density])
        
        hydro_ratio = self.calculate_hydrophobic_neighbor_ratio(
            residue_center, all_residue_centers, all_residue_names
        )
        features.append([hydro_ratio])
        
        charged_ratio = self.calculate_charged_neighbor_ratio(
            residue_center, all_residue_centers, all_residue_names
        )
        features.append([charged_ratio])
        
        pos_ratio = sequence_position / total_length if total_length > 0 else 0
        features.append([pos_ratio])
        
        features = np.concatenate(features)
        
        return features
    
    def extract_all_residues(self, pdb_file: str) -> Tuple[np.ndarray, List[Dict]]:
        """从PDB文件提取所有残基的特征"""
        
        atoms, residues = self._parse_pdb(pdb_file)
        
        if not residues:
            return np.array([]), []
        
        residue_list = []
        residue_centers = []
        residue_names = []
        
        for (chain, res_num), res_atoms in sorted(residues.items()):
            if 'CA' not in [a['atom_name'] for a in res_atoms]:
                continue
            
            ca_atom = next(a for a in res_atoms if a['atom_name'] == 'CA')
            residue_name = ca_atom['residue_name']
            
            residue_list.append({
                'chain_id': chain,
                'residue_num': res_num,
                'residue_name': residue_name,
                'atoms': res_atoms,
                'ca_coords': ca_atom['coords']
            })
            
            residue_centers.append(ca_atom['coords'])
            residue_names.append(residue_name)
        
        if not residue_list:
            return np.array([]), []
        
        all_centers = np.array(residue_centers)
        total_length = len(residue_list)
        
        features_list = []
        residue_info_list = []
        
        for i, residue in enumerate(residue_list):
            features = self.extract_residue_features(
                residue_atoms=residue['atoms'],
                residue_name=residue['residue_name'],
                residue_id=residue['residue_num'],
                chain_id=residue['chain_id'],
                all_residue_centers=all_centers,
                all_residue_names=residue_names,
                sequence_position=i + 1,
                total_length=total_length
            )
            
            features_list.append(features)
            
            residue_info_list.append({
                'chain_id': residue['chain_id'],
                'residue_num': residue['residue_num'],
                'residue_name': residue['residue_name'],
                'coordinates': residue['ca_coords'].tolist()
            })
        
        return np.array(features_list), residue_info_list
    
    def _parse_pdb(self, pdb_file: str) -> Tuple[List[Dict], Dict]:
        """解析PDB文件"""
        atoms = []
        residues = {}
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    try:
                        atom_name = line[12:16].strip()
                        residue_name = line[17:20].strip()
                        chain_id = line[21]
                        residue_num = int(line[22:26].strip())
                        x = float(line[30:38].strip())
                        y = float(line[38:46].strip())
                        z = float(line[46:54].strip())
                        b_factor = float(line[60:66].strip()) if line[60:66].strip() else 0.0
                        
                        atom = {
                            'atom_name': atom_name,
                            'residue_name': residue_name,
                            'chain_id': chain_id,
                            'residue_num': residue_num,
                            'coords': np.array([x, y, z]),
                            'b_factor': b_factor
                        }
                        
                        atoms.append(atom)
                        
                        key = (chain_id, residue_num)
                        if key not in residues:
                            residues[key] = []
                        residues[key].append(atom)
                        
                    except Exception as e:
                        continue
        
        return atoms, residues

residue_feature_extractor = ResidueFeatureExtractor()
