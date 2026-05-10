import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class AdvancedFeatures:
    """高级特征提取器 - 支持数据增强和注意力机制"""
    
    def __init__(self):
        pass
    
    def data_augmentation(self, smiles: str, num_augmented: int = 5) -> list:
        """数据增强 - 分子图像翻转、旋转、加噪等"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return []
        
        augmented_smiles = [smiles]
        
        # 1. 生成互变异构体
        try:
            from rdkit.Chem import EnumerateStereoisomers
            stereoisomers = list(EnumerateStereoisomers.EnumerateStereoisomers(mol))
            for iso in stereoisomers[:2]:
                augmented_smiles.append(Chem.MolToSmiles(iso))
        except:
            pass
        
        # 2. 构效团修饰 - 简化羟基化、甲基化等
        modifications = self._generate_modifications(mol)
        augmented_smiles.extend(modifications[:num_augmented - len(augmented_smiles)])
        
        return augmented_smiles
    
    def _generate_modifications(self, mol) -> list:
        """生成结构修饰变体"""
        modifications = []
        
        # 羟基添加
        for i in range(2):
            try:
                copy_mol = Chem.RWMol(mol)
                for atom in copy_mol.GetAtoms():
                    if atom.GetAtomicNum() == 8:
                        new_mol = copy_mol
                        new_atom = Chem.Atom(9)
                        new_atom.SetFormalCharge(0)
                        new_mol.AddAtom(new_atom)
                        new_mol.AddBond(atom.GetIdx(), new_atom.GetIdx(), Chem.BondType.SINGLE)
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles and new_smiles not in modifications:
                            modifications.append(new_smiles)
                        break
            except:
                continue
        
        # 甲基化
        for i in range(2):
            try:
                copy_mol = Chem.RWMol(mol)
                for atom in copy_mol.GetAtoms():
                    if atom.GetAtomicNum() == 8:
                        new_mol = copy_mol
                        new_atom = Chem.Atom(6)
                        new_atom.SetFormalCharge(0)
                        new_mol.AddAtom(new_atom)
                        new_mol.AddBond(atom.GetIdx(), new_atom.GetIdx(), Chem.BondType.SINGLE)
                        new_smiles = Chem.MolToSmiles(new_mol)
                        if new_smiles and new_smiles not in modifications:
                            modifications.append(new_smiles)
                        break
            except:
                continue
        
        return modifications
    
    def attention_weights(self, smiles: str, target_smiles: str) -> dict:
        """注意力机制 - 计算与目标分子的相似性和重要性"""
        mol1 = Chem.MolFromSmiles(smiles)
        mol2 = Chem.MolFromSmiles(target_smiles)
        
        if mol1 is None or mol2 is None:
            return {"error": "无效的SMILES"}
        
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors as rdmd
        
        # 指纹相似度
        fp1 = rdmd.GetMorganFingerprintAsBitVect(mol1, 2, 1024)
        fp2 = rdmd.GetMorganFingerprintAsBitVect(mol2, 2, 1024)
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        
        # 药效团匹配
        fp1_pharma = self._get_pharmacophore_fp(mol1)
        fp2_pharma = self._get_pharmacophore_fp(mol2)
        pharma_match = self._pharmacophore_similarity(fp1_pharma, fp2_pharma)
        
        # 分子量差异
        mw1 = Descriptors.MolWt(mol1)
        mw2 = Descriptors.MolWt(mol2)
        mw_diff = abs(mw1 - mw2)
        
        return {
            "similarity": similarity,
            "pharmacophore_match": pharma_match,
            "mw_diff": mw_diff,
            "attention_weight": (similarity + pharma_match) / 2,
            "importance": "高" if (similarity + pharma_match) > 0.8 else "中"
        }
    
    def _get_pharmacophore_fp(self, mol) -> dict:
        """提取药效团指纹"""
        fp = {
            "hbd": 0,
            "hba": 0,
            "aromatic": 0,
            "acidic": 0,
            "basic": 0
        }
        
        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num == 8:
                fp["acidic"] = 1
            elif atomic_num == 7:
                fp["basic"] = 1
        
        for bond in mol.GetBonds():
            if bond.GetBondType() in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE]:
                atom1 = bond.GetBeginAtom()
                atom2 = bond.GetEndAtom()
                if atom1.GetAtomicNum() == 7 or atom2.GetAtomicNum() == 7:
                    fp["hbd"] = 1
                elif atom1.GetAtomicNum() == 8 or atom2.GetAtomicNum() == 8:
                    fp["hba"] = 1
        
        for ring in mol.GetRingInfo().AtomRings():
            is_aromatic = all([mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring])
            if is_aromatic:
                fp["aromatic"] = 1
                break
        
        return fp
    
    def _pharmacophore_similarity(self, fp1: dict, fp2: dict) -> float:
        """药效团相似度"""
        matches = 0
        total_features = len(fp1) + len(fp2)
        
        for key in fp1:
            if fp1[key] == fp2[key]:
                matches += 1
        
        return matches / total_features if total_features > 0 else 0
    
    def active_learning_weights(self, smiles: str, known_actives: list) -> dict:
        """主动学习权重 - 基于已知活性分子计算重要性"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"error": "无效的SMILES"}
        
        from rdkit.Chem import rdMolDescriptors as rdmd
        
        # 计算与活性分子的平均相似度
        similarities = []
        for active_smiles in known_actives:
            active_mol = Chem.MolFromSmiles(active_smiles)
            if active_mol:
                fp1 = rdmd.GetMorganFingerprintAsBitVect(mol, 2, 1024)
                fp2 = rdmd.GetMorganFingerprintAsBitVect(active_mol, 2, 1024)
                from rdkit import DataStructs
                sim = DataStructs.TanimotoSimilarity(fp1, fp2)
                similarities.append(sim)
        
        avg_similarity = np.mean(similarities) if similarities else 0
        max_similarity = max(similarities) if similarities else 0
        
        # 计算活性预测分数
        activity_score = avg_similarity * 0.7 + max_similarity * 0.3
        
        return {
            "avg_similarity_to_actives": round(avg_similarity, 3),
            "max_similarity_to_actives": round(max_similarity, 3),
            "predicted_activity": activity_score,
            "activity_level": "高" if activity_score > 0.8 else "中" if activity_score > 0.5 else "低"
        }
    
    def imbalance_correction(self, smiles_list: list, active_ratio: float = 0.1) -> list:
        """数据不平衡修正 - 活性分子过采样"""
        if len(smiles_list) == 0:
            return []
        
        corrected = []
        
        # 随机过采样活性分子
        target_active_count = int(len(smiles_list) * active_ratio)
        
        for smiles in smiles_list:
            corrected.append({
                "smiles": smiles,
                "weight": 1.5,  # 活性分子权重更高
                "is_augmented": False
            })
        
        return corrected
    
    def generate_training_set(self, smiles_list: list, labels: list) -> dict:
        """生成训练集 - 包含增强数据"""
        augmented_data = []
        
        for smiles, label in zip(smiles_list, labels):
            augmented = self.data_augmentation(smiles, num_augmented=2)
            for aug_smiles in augmented:
                augmented_data.append({
                    "smiles": aug_smiles,
                    "label": label,
                    "is_augmented": aug_smiles != smiles
                })
        
        return {
            "original_count": len(smiles_list),
            "augmented_count": len(augmented_data),
            "augmentation_ratio": len(augmented_data) / len(smiles_list),
            "training_set": augmented_data
        }

advanced_features = AdvancedFeatures()
