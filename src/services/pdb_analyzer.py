import os
import numpy as np
from scipy.spatial.distance import cdist
from collections import defaultdict

class PDBAnalyzer:
    """PDB文件分析器 - 结合位点识别和分析"""
    
    def __init__(self):
        self.amino_acids = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
    
    def parse_pdb(self, pdb_file):
        """解析PDB文件，提取原子信息"""
        atoms = []
        residues = defaultdict(list)
        
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
                        element = line[76:78].strip()
                        
                        atom = {
                            'atom_name': atom_name,
                            'residue_name': residue_name,
                            'chain_id': chain_id,
                            'residue_num': residue_num,
                            'coords': np.array([x, y, z]),
                            'element': element
                        }
                        atoms.append(atom)
                        residues[(chain_id, residue_num)].append(atom)
                    except:
                        continue
        
        return atoms, residues
    
    def find_ligands(self, pdb_file):
        """识别PDB中的配体分子"""
        ligands = []
        
        with open(pdb_file, 'r') as f:
            for line in f:
                if line.startswith('HETATM'):
                    try:
                        residue_name = line[17:20].strip()
                        chain_id = line[21]
                        residue_num = int(line[22:26].strip())
                        
                        if residue_name not in ['HOH', 'WAT']:
                            if not any(l['residue_name'] == residue_name and 
                                     l['residue_num'] == residue_num for l in ligands):
                                ligands.append({
                                    'residue_name': residue_name,
                                    'chain_id': chain_id,
                                    'residue_num': residue_num
                                })
                    except:
                        continue
        
        return ligands
    
    def analyze_ligand_binding_site(self, pdb_file, ligand_residue=None, distance_cutoff=5.0):
        """基于配体分析结合位点"""
        atoms, residues = self.parse_pdb(pdb_file)
        ligands = self.find_ligands(pdb_file)
        
        if not ligands:
            return {
                "success": False,
                "error": "PDB文件中未发现配体，无法识别结合位点",
                "suggestion": "请上传包含配体的PDB文件，或使用几何方法识别口袋"
            }
        
        if ligand_residue:
            target_ligand = [l for l in ligands 
                           if l['residue_name'] == ligand_residue]
            if not target_ligand:
                return {"success": False, "error": f"未找到配体 {ligand_residue}"}
            ligands = target_ligand
        
        binding_sites = []
        
        for ligand in ligands:
            ligand_atoms = [a for a in atoms 
                          if a['residue_name'] == ligand['residue_name'] and 
                             a['residue_num'] == ligand['residue_num'] and
                             a['chain_id'] == ligand['chain_id']]
            
            if not ligand_atoms:
                continue
            
            ligand_center = np.mean([a['coords'] for a in ligand_atoms], axis=0)
            
            nearby_residues = set()
            for atom in atoms:
                if atom['residue_name'] in ['HOH', 'WAT']:
                    continue
                distance = np.linalg.norm(atom['coords'] - ligand_center)
                if distance <= distance_cutoff:
                    nearby_residues.add(
                        (atom['chain_id'], atom['residue_num'], atom['residue_name'])
                    )
            
            binding_site_residues = []
            for chain, res_num, res_name in sorted(nearby_residues):
                res_atoms = [a for a in atoms 
                           if a['chain_id'] == chain and a['residue_num'] == res_num]
                res_center = np.mean([a['coords'] for a in res_atoms], axis=0)
                
                binding_site_residues.append({
                    'chain_id': chain,
                    'residue_num': res_num,
                    'residue_name': res_name,
                    'residue_type': self.amino_acids.get(res_name, 'X'),
                    'center_coords': res_center.tolist(),
                    'num_atoms': len(res_atoms)
                })
            
            binding_sites.append({
                'ligand': ligand,
                'ligand_center': ligand_center.tolist(),
                'num_nearby_residues': len(binding_site_residues),
                'residues': binding_site_residues
            })
        
        return {
            "success": True,
            "pdb_file": pdb_file,
            "num_ligands": len(ligands),
            "binding_sites": binding_sites,
            "total_binding_residues": sum([site['num_nearby_residues'] for site in binding_sites])
        }
    
    def identify_geometric_pockets(self, pdb_file, min_pocket_size=5):
        """几何方法识别口袋（基于表面凹度）"""
        atoms, residues = self.parse_pdb(pdb_file)
        
        ca_atoms = [a for a in atoms if a['atom_name'] == 'CA']
        
        if len(ca_atoms) < min_pocket_size:
            return {
                "success": False,
                "error": "PDB文件太小，无法识别口袋"
            }
        
        coords = np.array([a['coords'] for a in ca_atoms])
        
        # 计算局部密度
        pockets = []
        for i, center_atom in enumerate(ca_atoms):
            distances = cdist([center_atom['coords']], coords)[0]
            neighbors = np.sum(distances < 8.0)
            
            # 计算局部曲率（简化版）
            if neighbors >= min_pocket_size:
                neighbor_coords = coords[distances < 8.0]
                local_center = np.mean(neighbor_coords, axis=0)
                distances_to_center = np.linalg.norm(neighbor_coords - local_center, axis=1)
                curvature = np.std(distances_to_center)
                
                if curvature > 1.0:
                    pocket_residues = []
                    for j in range(len(ca_atoms)):
                        if distances[j] < 8.0:
                            pocket_residues.append({
                                'chain_id': ca_atoms[j]['chain_id'],
                                'residue_num': ca_atoms[j]['residue_num'],
                                'residue_name': ca_atoms[j]['residue_name'],
                                'residue_type': self.amino_acids.get(
                                    ca_atoms[j]['residue_name'], 'X'
                                ),
                                'distance': float(distances[j])
                            })
                    
                    pockets.append({
                        'pocket_id': len(pockets) + 1,
                        'center': local_center.tolist(),
                        'curvature': float(curvature),
                        'num_residues': len(pocket_residues),
                        'residues': pocket_residues[:20]
                    })
        
        # 按曲率排序
        pockets.sort(key=lambda x: x['curvature'], reverse=True)
        
        return {
            "success": True,
            "pdb_file": pdb_file,
            "num_pockets": len(pockets),
            "pockets": pockets[:10]
        }
    
    def analyze_binding_sites(self, pdb_file, method='ligand', **kwargs):
        """分析结合位点 - 主入口函数"""
        
        if not os.path.exists(pdb_file):
            return {
                "success": False,
                "error": f"PDB文件不存在: {pdb_file}"
            }
        
        # 检查文件大小
        file_size = os.path.getsize(pdb_file)
        if file_size == 0:
            return {
                "success": False,
                "error": "PDB文件为空"
            }
        
        if method == 'ligand':
            result = self.analyze_ligand_binding_site(pdb_file, **kwargs)
        elif method == 'geometric':
            result = self.identify_geometric_pockets(pdb_file, **kwargs)
        else:
            return {
                "success": False,
                "error": f"未知的方法: {method}，支持的方法: ligand, geometric"
            }
        
        # 添加基本信息
        atoms, residues = self.parse_pdb(pdb_file)
        result['basic_info'] = {
            'num_atoms': len(atoms),
            'num_residues': len(residues),
            'chains': sorted(set(a['chain_id'] for a in atoms)),
            'file_size_kb': round(file_size / 1024, 2)
        }
        
        return result
    
    def get_pocket_summary(self, pdb_file):
        """获取口袋摘要信息"""
        ligand_result = self.analyze_binding_sites(pdb_file, method='ligand')
        geometric_result = self.analyze_binding_sites(pdb_file, method='geometric')
        
        summary = {
            "pdb_file": pdb_file,
            "ligand_based": {
                "has_ligands": ligand_result.get('success', False),
                "num_ligands": ligand_result.get('num_ligands', 0),
                "num_binding_sites": len(ligand_result.get('binding_sites', []))
            },
            "geometric_based": {
                "num_pockets": geometric_result.get('num_pockets', 0),
                "best_pocket": geometric_result.get('pockets', [{}])[0] if geometric_result.get('pockets') else None
            },
            "recommendation": self._get_recommendation(ligand_result, geometric_result)
        }
        
        return summary
    
    def _get_recommendation(self, ligand_result, geometric_result):
        """生成推荐建议"""
        if ligand_result.get('success'):
            return "基于配体的结合位点分析更准确，建议重点关注配体周围的残基"
        elif geometric_result.get('num_pockets', 0) > 0:
            return "使用几何方法识别到潜在口袋，建议进一步验证"
        else:
            return "未能识别到明确的结合位点，建议检查PDB文件或上传包含配体的结构"

pdb_analyzer = PDBAnalyzer()
