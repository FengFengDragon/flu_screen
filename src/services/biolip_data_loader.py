import os
import gzip
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class BioLiPDataLoader:
    """BioLiP 数据库解析器，用于生成真实训练数据"""

    def __init__(self, biolip_path: str = "BioLiP.txt.gz"):
        self.biolip_path = biolip_path

    def parse_biolip(self, max_entries: int = None) -> Dict[str, List[Dict]]:
        """解析 BioLiP.txt.gz，返回 {pdb_id: [binding_site_entries]}"""
        entries = defaultdict(list)
        count = 0

        opener = gzip.open if self.biolip_path.endswith('.gz') else open
        with opener(self.biolip_path, 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                fields = line.split('\t')
                if len(fields) < 9:
                    continue

                pdb_id = fields[0].strip().lower()
                chain_id = fields[1].strip()
                resolution = float(fields[2]) if fields[2].strip() else 0.0
                binding_site_residues_raw = fields[7].strip()
                binding_site_renumbered = fields[8].strip() if len(fields) > 8 else ''

                if not binding_site_residues_raw:
                    continue

                residue_set = self._parse_residue_string(binding_site_residues_raw)
                renumbered_positions = self._parse_positions_only(binding_site_renumbered)

                sequence = fields[20].strip() if len(fields) > 20 else ''

                entries[pdb_id].append({
                    'pdb_id': pdb_id,
                    'chain_id': chain_id,
                    'resolution': resolution,
                    'binding_residues': residue_set,
                    'renumbered_positions': renumbered_positions,
                    'sequence': sequence,
                    'sequence_length': len(sequence)
                })

                count += 1
                if max_entries and count >= max_entries:
                    break

        return dict(entries)

    def _parse_residue_string(self, residue_str: str) -> set:
        """解析 'F43 R45 V68 S92' 格式的残基字符串（单字母+编号）"""
        aa_1to3 = {
            'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
            'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
            'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'
        }
        residues = set()
        for token in residue_str.split():
            token = token.strip()
            if not token:
                continue
            try:
                aa_char = token[0].upper()
                res_num = int(token[1:])
                res_name = aa_1to3.get(aa_char, token[:3].upper())
                residues.add((res_name, res_num))
            except (ValueError, IndexError):
                continue
        return residues

    def _parse_positions_only(self, residue_str: str) -> set:
        """解析重编号后的残基字符串，返回位置编号集合"""
        positions = set()
        for token in residue_str.split():
            token = token.strip()
            if not token:
                continue
            try:
                pos = int(token[1:])
                positions.add(pos)
            except (ValueError, IndexError):
                continue
        return positions

    def get_binding_labels_for_pdb(self, pdb_entries: List[Dict],
                                   residue_info: List[Dict]) -> np.ndarray:
        """为 PDB 的残基列表生成标签 y

        Args:
            pdb_entries: 该 PDB 在 BioLiP 中的所有条目
            residue_info: 从 PDB 提取的残基信息列表
                         每项包含 chain_id, residue_num, residue_name

        Returns:
            y: numpy array, 1=结合位点, 0=非结合位点
        """
        all_binding = set()
        for entry in pdb_entries:
            for res_name, res_num in entry['binding_residues']:
                all_binding.add((entry['chain_id'], res_num, res_name))

        labels = np.zeros(len(residue_info), dtype=int)
        for i, info in enumerate(residue_info):
            key = (info['chain_id'], info['residue_num'], info['residue_name'])
            if key in all_binding:
                labels[i] = 1

        return labels

    def generate_training_data_from_features(self,
                                             biolip_entries: Dict[str, List[Dict]],
                                             pdb_dir: str,
                                             feature_extractor,
                                             max_pdbs: int = 500,
                                             min_binding_ratio: float = 0.01,
                                             max_binding_ratio: float = 0.8) -> Tuple[
        np.ndarray, np.ndarray, Dict]:
        """从 BioLiP 标注 + PDB 文件生成训练数据

        Args:
            biolip_entries: parse_biolip() 的返回结果
            pdb_dir: PDB 文件目录
            feature_extractor: ResidueFeatureExtractor 实例
            max_pdbs: 最多处理多少个 PDB
            min_binding_ratio: 结合残基占比过低则跳过
            max_binding_ratio: 结合残基占比过高则跳过（可能是非特异性）

        Returns:
            X: 特征矩阵 (N, 32)
            y: 标签 (N,)
            stats: 统计信息
        """
        all_features = []
        all_labels = []
        stats = {
            'total_pdbs_processed': 0,
            'total_pdbs_skipped': 0,
            'total_residues': 0,
            'total_binding': 0,
            'skipped_reasons': defaultdict(int)
        }

        for pdb_id in list(biolip_entries.keys())[:max_pdbs]:
            pdb_file = self._find_pdb_file(pdb_id, pdb_dir)
            if not pdb_file:
                stats['skipped_reasons']['pdb_not_found'] += 1
                stats['total_pdbs_skipped'] += 1
                continue

            try:
                features, residue_info = feature_extractor.extract_all_residues(pdb_file)
                if len(features) == 0:
                    stats['skipped_reasons']['no_features'] += 1
                    stats['total_pdbs_skipped'] += 1
                    continue

                labels = self.get_binding_labels_for_pdb(
                    biolip_entries[pdb_id], residue_info
                )

                binding_ratio = labels.sum() / len(labels) if len(labels) > 0 else 0
                if binding_ratio < min_binding_ratio or binding_ratio > max_binding_ratio:
                    stats['skipped_reasons']['bad_ratio'] += 1
                    stats['total_pdbs_skipped'] += 1
                    continue

                all_features.append(features)
                all_labels.append(labels)

                stats['total_pdbs_processed'] += 1
                stats['total_residues'] += len(labels)
                stats['total_binding'] += int(labels.sum())

            except Exception as e:
                stats['skipped_reasons'][f'error: {str(e)[:50]}'] += 1
                stats['total_pdbs_skipped'] += 1
                continue

        if not all_features:
            return np.array([]), np.array([]), dict(stats)

        X = np.vstack(all_features)
        y = np.concatenate(all_labels)

        stats['final_X_shape'] = list(X.shape)
        stats['final_y_shape'] = list(y.shape)
        stats['final_binding_ratio'] = float(y.sum() / len(y))
        stats['skipped_reasons'] = dict(stats['skipped_reasons'])

        return X, y, dict(stats)

    def generate_training_data_from_biolip_sequences(self,
                                                     biolip_entries: Dict[
                                                         str, List[Dict]],
                                                     max_entries: int = 5000,
                                                     min_binding_ratio: float = 0.01,
                                                     max_binding_ratio: float = 0.8) -> Tuple[
        np.ndarray, np.ndarray, Dict]:
        """仅从 BioLiP 序列数据生成训练数据（不需要 PDB 文件）

        使用序列级别的特征代替结构特征。
        特征: [20维 one-hot, 4维 物化性质, 8维 位置/统计特征] = 32维
        """
        all_features = []
        all_labels = []
        stats = {
            'total_entries_processed': 0,
            'total_residues': 0,
            'total_binding': 0,
            'total_entries_skipped': 0
        }

        aa_1to3 = _AA_1TO3
        aa_order = _AA_ORDER

        physicochemical = _PHYSICOCHEMICAL
        hydrophobic_set = _HYDROPHOBIC
        charged_set = _CHARGED
        polar_set = _POLAR

        processed_pdbs = set()
        entry_count = 0

        for pdb_id, entries in biolip_entries.items():
            if entry_count >= max_entries:
                break

            all_binding_positions = set()
            sequence = ''
            chain_id = ''

            for entry in entries:
                if not sequence and entry['sequence']:
                    sequence = entry['sequence']
                    chain_id = entry['chain_id']

                if entry.get('renumbered_positions'):
                    all_binding_positions.update(entry['renumbered_positions'])
                else:
                    for res_name, res_num in entry['binding_residues']:
                        all_binding_positions.add(res_num)

            if not sequence or len(sequence) < 10:
                stats['total_entries_skipped'] += 1
                continue

            binding_count = 0
            for pos in all_binding_positions:
                if 1 <= pos <= len(sequence):
                    binding_count += 1

            total = len(sequence)
            binding_ratio = binding_count / total if total > 0 else 0
            if binding_ratio < min_binding_ratio or binding_ratio > max_binding_ratio:
                stats['total_entries_skipped'] += 1
                continue

            if pdb_id in processed_pdbs:
                continue
            processed_pdbs.add(pdb_id)

            seq_upper = sequence.upper()
            window_small = 5
            window_large = 10

            for pos_idx, aa_char in enumerate(sequence):
                seq_pos = pos_idx + 1
                aa_upper = aa_char.upper()
                aa_three = aa_1to3.get(aa_upper)
                if not aa_three:
                    continue

                one_hot = np.zeros(20)
                if aa_three in aa_order:
                    one_hot[aa_order.index(aa_three)] = 1

                pc = physicochemical.get(aa_upper, [0, 0, 0, 100])
                phys_feat = np.array(pc, dtype=float)

                s_start = max(0, pos_idx - window_small)
                s_end = min(total, pos_idx + window_small + 1)
                small_window = seq_upper[s_start:s_end]
                sw_len = len(small_window)

                local_hydro = sum(1 for c in small_window if c in hydrophobic_set) / sw_len if sw_len else 0
                local_charge = sum(1 for c in small_window if c in charged_set) / sw_len if sw_len else 0
                local_polar = sum(1 for c in small_window if c in polar_set) / sw_len if sw_len else 0

                l_start = max(0, pos_idx - window_large)
                l_end = min(total, pos_idx + window_large + 1)
                large_window = seq_upper[l_start:l_end]
                lw_len = len(large_window)

                frac_hydro_w10 = sum(1 for c in large_window if c in hydrophobic_set) / lw_len if lw_len else 0
                frac_charged_w10 = sum(1 for c in large_window if c in charged_set) / lw_len if lw_len else 0
                neighbor_diversity = len(set(c for c in large_window if c.isalpha())) / 20.0

                gap_left = pos_idx / total
                gap_right = (total - pos_idx - 1) / total

                seq_features = np.array([
                    (pos_idx + 1) / total,
                    local_hydro,
                    local_charge,
                    local_polar,
                    frac_hydro_w10,
                    frac_charged_w10,
                    neighbor_diversity,
                    gap_left + gap_right
                ], dtype=float)

                feature = np.concatenate([one_hot, phys_feat, seq_features])

                is_binding = 1 if seq_pos in all_binding_positions else 0

                all_features.append(feature)
                all_labels.append(is_binding)

                stats['total_residues'] += 1
                if is_binding:
                    stats['total_binding'] += 1

            stats['total_entries_processed'] += 1
            entry_count += 1

        if not all_features:
            return np.array([]), np.array([]), dict(stats)

        X = np.array(all_features)
        y = np.array(all_labels)

        stats['final_X_shape'] = list(X.shape)
        stats['final_y_shape'] = list(y.shape)
        stats['final_binding_ratio'] = float(y.sum() / len(y))

        return X, y, dict(stats)

    def _find_pdb_file(self, pdb_id: str, pdb_dir: str) -> Optional[str]:
        """在目录中查找 PDB 文件"""
        possible_names = [
            f"{pdb_id}.pdb",
            f"{pdb_id.upper()}.pdb",
            f"pdb{pdb_id}.ent",
            f"pdb{pdb_id.upper()}.ent",
        ]
        for name in possible_names:
            path = os.path.join(pdb_dir, name)
            if os.path.exists(path):
                return path

        for root, dirs, files in os.walk(pdb_dir):
            for f in files:
                if pdb_id in f.lower() and (f.endswith('.pdb') or f.endswith('.ent')):
                    return os.path.join(root, f)

        return None

    def get_data_summary(self) -> Dict:
        """获取 BioLiP 数据摘要（不加载全部数据）"""
        entries = self.parse_biolip(max_entries=1000)

        total_pdbs = len(entries)
        total_entries = sum(len(v) for v in entries.values())

        ligand_types = defaultdict(int)
        for pdb_entries in entries.values():
            for entry in pdb_entries:
                ligand_types['all'] += 1

        return {
            'biolip_file': self.biolip_path,
            'file_exists': os.path.exists(self.biolip_path),
            'sample_pdbs_processed': total_pdbs,
            'sample_entries': total_entries,
            'sample_ligand_count': ligand_types.get('all', 0)
        }


biolip_data_loader = BioLiPDataLoader()


_AA_3TO1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}
_AA_1TO3 = {v: k for k, v in _AA_3TO1.items()}
_AA_ORDER = list(_AA_3TO1.keys())

_PHYSICOCHEMICAL = {
    'A': [1.8, 0, 0, 67], 'R': [-4.5, 1, 1, 148], 'N': [-3.5, 0, 1, 96],
    'D': [-3.5, -1, 1, 91], 'C': [2.5, 0, 0, 86], 'Q': [-3.5, 0, 1, 114],
    'E': [-3.5, -1, 1, 109], 'G': [-0.4, 0, 0, 48], 'H': [-3.2, 0.5, 1, 118],
    'I': [4.5, 0, 0, 124], 'L': [3.8, 0, 0, 124], 'K': [-3.9, 1, 1, 135],
    'M': [1.9, 0, 0, 124], 'F': [2.8, 0, 0, 135], 'P': [-1.6, 0, 0, 90],
    'S': [-0.8, 0, 1, 73], 'T': [-0.7, 0, 1, 93], 'W': [-0.9, 0, 0, 163],
    'Y': [-1.3, 0, 1, 141], 'V': [4.2, 0, 0, 105]
}

_HYDROPHOBIC = set('AILMFWVP')
_CHARGED = set('RDEKH')
_POLAR = set('STNQYC')


def extract_sequence_features_for_residues(residue_info: List[Dict]) -> np.ndarray:
    """从残基列表中提取序列级特征（与训练时完全一致）

    Args:
        residue_info: 包含 residue_name 字段的残基信息列表

    Returns:
        features: (N, 32) 的特征矩阵
    """
    total = len(residue_info)
    seq_str = ''
    for info in residue_info:
        aa = _AA_3TO1.get(info.get('residue_name', ''), 'X')
        seq_str += aa

    seq_upper = seq_str.upper()
    window_small = 5
    window_large = 10

    all_features = []
    for pos_idx in range(total):
        aa_upper = seq_upper[pos_idx]
        aa_three = _AA_1TO3.get(aa_upper)
        if not aa_three:
            all_features.append(np.zeros(32))
            continue

        one_hot = np.zeros(20)
        if aa_three in _AA_ORDER:
            one_hot[_AA_ORDER.index(aa_three)] = 1

        pc = _PHYSICOCHEMICAL.get(aa_upper, [0, 0, 0, 100])
        phys_feat = np.array(pc, dtype=float)

        s_start = max(0, pos_idx - window_small)
        s_end = min(total, pos_idx + window_small + 1)
        small_window = seq_upper[s_start:s_end]
        sw_len = len(small_window)

        local_hydro = sum(1 for c in small_window if c in _HYDROPHOBIC) / sw_len if sw_len else 0
        local_charge = sum(1 for c in small_window if c in _CHARGED) / sw_len if sw_len else 0
        local_polar = sum(1 for c in small_window if c in _POLAR) / sw_len if sw_len else 0

        l_start = max(0, pos_idx - window_large)
        l_end = min(total, pos_idx + window_large + 1)
        large_window = seq_upper[l_start:l_end]
        lw_len = len(large_window)

        frac_hydro_w10 = sum(1 for c in large_window if c in _HYDROPHOBIC) / lw_len if lw_len else 0
        frac_charged_w10 = sum(1 for c in large_window if c in _CHARGED) / lw_len if lw_len else 0
        neighbor_diversity = len(set(c for c in large_window if c.isalpha())) / 20.0

        gap_left = pos_idx / total
        gap_right = (total - pos_idx - 1) / total

        seq_features = np.array([
            (pos_idx + 1) / total,
            local_hydro,
            local_charge,
            local_polar,
            frac_hydro_w10,
            frac_charged_w10,
            neighbor_diversity,
            gap_left + gap_right
        ], dtype=float)

        feature = np.concatenate([one_hot, phys_feat, seq_features])
        all_features.append(feature)

    return np.array(all_features)
