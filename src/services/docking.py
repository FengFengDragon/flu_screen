# e:\pycharmcode\flu-screen\src\services\docking.py
import os
import subprocess
import tempfile
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen, QED

class MolecularDocking:
    """分子对接服务 - 使用RDKit进行分子准备"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    def calculate_descriptors(self, smiles):
        """计算分子描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        descriptors = {
            "smiles": smiles,
            "molecular_weight": Descriptors.MolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": Descriptors.TPSA(mol),
            "num_rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "num_aromatic_rings": Lipinski.NumAromaticRings(mol),
            "num_hbd": Lipinski.NumHDonors(mol),
            "num_hba": Lipinski.NumHAcceptors(mol),
            "num_heavy_atoms": Descriptors.HeavyAtomCount(mol),
            "num_rings": Descriptors.RingCount(mol),
            "fraction_csp3": Descriptors.FractionCSP3(mol),
            "balaban_j": Descriptors.BalabanJ(mol) if Descriptors.RingCount(mol) > 0 else 0,
            "bertz_ct": Descriptors.BertzCT(mol),
        }
        
        return descriptors
    
    def screen_compounds(self, smiles_list, criteria=None):
        """筛选化合物"""
        if criteria is None:
            criteria = {
                "max_mw": 500,
                "max_logp": 5,
                "min_qed": 0.5
            }
        
        passed = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                qed_value = QED.qed(mol)
                
                if (mw <= criteria.get("max_mw", 500) and 
                    logp <= criteria.get("max_logp", 5) and 
                    qed_value >= criteria.get("min_qed", 0.5)):
                    passed.append({
                        "smiles": smiles,
                        "molecular_weight": mw,
                        "logp": logp,
                        "qed": qed_value
                    })
            except:
                continue
        
        return passed

molecular_docking = MolecularDocking()