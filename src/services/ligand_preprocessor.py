import os
import tempfile
from typing import List, Dict, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski, Crippen, QED
from rdkit.Chem import rdMolDescriptors, Draw
from rdkit.Chem.rdMolDescriptors import CalcTPSA


class LigandPreprocessor:
    """小分子预处理服务 - 负责虚拟筛选前的分子准备和过滤"""

    def parse_smiles(self, smiles: str) -> Optional[object]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        return mol

    def parse_smiles_batch(self, smiles_list: List[str]) -> Dict:
        """批量解析 SMILES，返回有效/无效分类结果"""
        valid, invalid = [], []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                valid.append(smi)
            else:
                invalid.append(smi)
        return {"valid": valid, "invalid": invalid, "total": len(smiles_list),
                "valid_count": len(valid), "invalid_count": len(invalid)}

    def calculate_descriptors(self, smiles: str) -> Optional[Dict]:
        """计算单个分子的理化描述符"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return {
            "smiles": smiles,
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "tpsa": round(CalcTPSA(mol), 2),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "aromatic_rings": Lipinski.NumAromaticRings(mol),
            "heavy_atoms": rdMolDescriptors.CalcNumHeavyAtoms(mol),
            "qed": round(QED.qed(mol), 3),
            "fraction_csp3": round(Descriptors.FractionCSP3(mol), 3),
            "ring_count": Descriptors.RingCount(mol),
        }

    def lipinski_filter(self, smiles: str) -> Dict:
        """Lipinski 五规则过滤"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"pass": False, "error": "无效SMILES"}

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        violations = []
        if mw > 500:   violations.append(f"MW={mw:.1f} > 500")
        if logp > 5:   violations.append(f"LogP={logp:.2f} > 5")
        if hbd > 5:    violations.append(f"HBD={hbd} > 5")
        if hba > 10:   violations.append(f"HBA={hba} > 10")

        return {
            "smiles": smiles,
            "pass": len(violations) == 0,
            "violations": violations,
            "mw": round(mw, 2), "logp": round(logp, 2),
            "hbd": hbd, "hba": hba,
        }

    def admet_filter(self, smiles: str) -> Dict:
        """简化 ADMET 过滤（基于描述符规则）"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"pass": False, "error": "无效SMILES"}

        tpsa = CalcTPSA(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)

        flags = {
            "oral_absorption": tpsa < 140 and rotatable <= 10,
            "bbb_penetration": tpsa < 90 and mw < 450 and logp < 5,
            "solubility": logp < 4 and tpsa > 20,
            "metabolic_stability": rotatable <= 7,
        }
        issues = [k for k, v in flags.items() if not v]

        return {
            "smiles": smiles,
            "pass": len(issues) == 0,
            "flags": flags,
            "issues": issues,
            "tpsa": round(tpsa, 2),
            "rotatable_bonds": rotatable,
        }

    def pains_filter(self, smiles: str) -> Dict:
        """PAINS（泛筛选干扰化合物）过滤"""
        from rdkit.Chem import FilterCatalog
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        catalog = FilterCatalog.FilterCatalog(params)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"pass": False, "error": "无效SMILES"}

        entry = catalog.GetFirstMatch(mol)
        is_pains = entry is not None
        return {
            "smiles": smiles,
            "pass": not is_pains,
            "is_pains": is_pains,
            "pains_description": entry.GetDescription() if is_pains else None,
        }

    def full_preprocess(self, smiles_list: List[str], filters: Optional[Dict] = None) -> Dict:
        """
        完整预处理流程：解析 → Lipinski → ADMET → PAINS → 描述符计算
        filters 可控制开关各步骤，默认全开
        """
        if filters is None:
            filters = {"lipinski": True, "admet": True, "pains": True}

        results = []
        stats = {"total": len(smiles_list), "invalid_smiles": 0,
                 "lipinski_failed": 0, "admet_failed": 0,
                 "pains_failed": 0, "passed": 0}

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                stats["invalid_smiles"] += 1
                continue

            entry = {"smiles": smi, "filters": {}, "passed": True}

            if filters.get("lipinski", True):
                r = self.lipinski_filter(smi)
                entry["filters"]["lipinski"] = r
                if not r["pass"]:
                    stats["lipinski_failed"] += 1
                    entry["passed"] = False

            if filters.get("admet", True):
                r = self.admet_filter(smi)
                entry["filters"]["admet"] = r
                if not r["pass"]:
                    stats["admet_failed"] += 1
                    entry["passed"] = False

            if filters.get("pains", True):
                r = self.pains_filter(smi)
                entry["filters"]["pains"] = r
                if not r["pass"]:
                    stats["pains_failed"] += 1
                    entry["passed"] = False

            if entry["passed"]:
                entry["descriptors"] = self.calculate_descriptors(smi)
                stats["passed"] += 1

            results.append(entry)

        passed_molecules = [r for r in results if r["passed"]]
        return {
            "success": True,
            "stats": stats,
            "passed_molecules": passed_molecules,
            "all_results": results,
        }

    def generate_3d_conformer(self, smiles: str) -> Dict:
        """生成 3D 构象并返回原子坐标"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效SMILES"}

        mol = Chem.AddHs(mol)
        result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        if result != 0:
            return {"success": False, "error": "3D构象生成失败"}

        AllChem.MMFFOptimizeMolecule(mol)
        conf = mol.GetConformer()
        atoms = []
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            atoms.append({
                "idx": i, "symbol": atom.GetSymbol(),
                "x": round(pos.x, 4), "y": round(pos.y, 4), "z": round(pos.z, 4),
            })

        return {"success": True, "smiles": smiles,
                "num_atoms": mol.GetNumAtoms(), "atoms": atoms}

    def mol_to_pdb_block(self, smiles: str) -> Dict:
        """将 SMILES 转为 PDB 格式文本（用于后续对接）"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效SMILES"}

        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
        AllChem.MMFFOptimizeMolecule(mol)
        pdb_block = Chem.MolToPDBBlock(mol)

        return {"success": True, "smiles": smiles, "pdb_block": pdb_block}


ligand_preprocessor = LigandPreprocessor()
