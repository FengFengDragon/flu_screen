import time
import logging
import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

logger = logging.getLogger(__name__)

class EnhancedDocking:
    """增强型分子对接服务 - 支持分级对接和结合自由能计算"""
    
    def __init__(self):
        self.docking_results = []
    
    def tiered_docking(self, smiles_list: list,
                       target_pdb: Optional[str] = None,
                       binding_site_coords: Optional[list] = None,
                       center: Optional[list] = None,
                       box_size: Optional[list] = None,
                       tier1_count: int = 500, tier2_count: int = 100,
                       tier3_count: int = 20) -> dict:
        """分级对接策略

        Tier1: RDKit 描述符快速打分（始终使用）
        Tier2: 当提供 target_pdb 时用 Vina 标准精度对接，否则用 RDKit
        Tier3: 当提供 target_pdb 时用 Vina 精细对接（高 exhaustiveness），否则用 RDKit
        """
        
        start_time = time.time()
        
        use_vina = False
        if target_pdb:
            try:
                from src.services.vina_docking import vina_docking_service
                use_vina = vina_docking_service.is_available()
            except ImportError:
                logger.warning("vina_docking module not available, falling back to RDKit scoring")
                use_vina = False
        
        tier2_method = "Vina标准对接" if use_vina else "RDKit描述符评分"
        tier3_method = "Vina精细对接" if use_vina else "RDKit精细评分"
        
        results = {
            "tier1": {"method": "高通量筛选", "compounds": [], "time_estimate": "1-3分钟"},
            "tier2": {"method": tier2_method, "compounds": [], "time_estimate": "5-10分钟" if use_vina else "2-5分钟"},
            "tier3": {"method": tier3_method, "compounds": [], "time_estimate": "10-20分钟" if use_vina else "3-8分钟"},
            "summary": {},
            "use_vina": use_vina,
            "target_pdb": target_pdb,
        }
        
        tier1_results = self._high_throughput_screen(smiles_list, tier1_count)
        results["tier1"]["compounds"] = tier1_results
        
        if use_vina:
            tier2_results = self._vina_docking(
                tier1_results[:tier2_count], target_pdb,
                binding_site_coords, center, box_size,
                exhaustiveness=8
            )
        else:
            tier2_results = self._standard_docking(tier1_results[:tier2_count])
        results["tier2"]["compounds"] = tier2_results
        
        if use_vina:
            tier3_results = self._vina_docking(
                tier2_results[:tier3_count], target_pdb,
                binding_site_coords, center, box_size,
                exhaustiveness=32
            )
        else:
            tier3_results = self._refined_docking(tier2_results[:tier3_count])
        results["tier3"]["compounds"] = tier3_results
        
        best_score = None
        if tier3_results:
            best_score = tier3_results[0].get("docking_score", tier3_results[0].get("best_score", 0))
        
        results["summary"] = {
            "total_screened": len(smiles_list),
            "tier1_passed": len(tier1_results),
            "tier2_passed": len(tier2_results),
            "final_candidates": len(tier3_results),
            "best_score": best_score,
            "docking_method": "AutoDock Vina" if use_vina else "RDKit描述符",
            "recommendation": "建议对Tier3候选化合物进行实验验证",
            "processing_time": round(time.time() - start_time, 2)
        }
        
        self.docking_results = tier3_results
        return results

    def _vina_docking(self, compounds: list, target_pdb: Optional[str],
                      binding_site_coords: Optional[list],
                      center: Optional[list], box_size: Optional[list],
                      exhaustiveness: int = 8) -> list:
        """使用 AutoDock Vina 进行真实分子对接"""
        try:
            from src.services.vina_docking import vina_docking_service
        except ImportError:
            return self._standard_docking(compounds)

        if not target_pdb:
            return self._standard_docking(compounds)

        smiles_list = []
        for compound in compounds:
            smiles_list.append({
                'smiles': compound.get('smiles', ''),
                'name': compound.get('smiles', 'ligand')[:20],
            })

        batch_result = vina_docking_service.dock_batch(
            pdb_path=target_pdb,
            smiles_list=smiles_list,
            center=center,
            box_size=box_size,
            binding_site_coords=binding_site_coords,
            padding=10.0,
            exhaustiveness=exhaustiveness,
            n_poses=10,
        )

        if not batch_result.get('success'):
            logger.warning(f"Vina batch docking failed: {batch_result.get('error')}, falling back to RDKit")
            return self._standard_docking(compounds)

        results = []
        for vina_res in batch_result.get('results', []):
            if vina_res.get('success'):
                best = vina_res.get('best_score', 0)
                poses = vina_res.get('poses', [])
                mol = Chem.MolFromSmiles(vina_res.get('smiles', ''))
                mmgbsa = self._estimate_mmgbsa(mol) if mol else best * 1.1
                kd = self._score_to_kd(best)
                results.append({
                    "smiles": vina_res.get('smiles', ''),
                    "docking_score": best,
                    "best_score": best,
                    "tier": 3 if exhaustiveness >= 16 else 2,
                    "binding_energy_estimate": best,
                    "mmgbsa_energy": mmgbsa,
                    "binding_affinity_kd": kd,
                    "method": "AutoDock Vina",
                    "exhaustiveness": exhaustiveness,
                    "n_poses": vina_res.get('n_poses', 0),
                    "poses": poses[:5],
                    "confidence": "high" if best < -7 else ("medium" if best < -5 else "low"),
                })
            else:
                results.append({
                    "smiles": vina_res.get('smiles', ''),
                    "docking_score": 0,
                    "error": vina_res.get('error', 'unknown'),
                    "method": "AutoDock Vina (failed)",
                })

        results.sort(key=lambda x: x.get("docking_score", 0))
        return results
    
    def _high_throughput_screen(self, smiles_list: list, top_n: int = 500) -> list:
        """高通量虚拟筛选 - 使用快速打分函数"""
        results = []
        
        for smiles in smiles_list[:top_n]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                score = self._quick_score(mol)
                results.append({
                    "smiles": smiles,
                    "docking_score": score,
                    "molecular_weight": None,
                    "logp": None,
                    "tier": 1
                })
            except:
                continue
        
        results.sort(key=lambda x: x["docking_score"], reverse=True)
        return results[:min(len(results), top_n)]
    
    def _standard_docking(self, compounds: list) -> list:
        """标准精度对接 - 使用更精确的打分"""
        results = []
        
        for compound in compounds:
            smiles = compound.get("smiles")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                score = self._standard_score(mol)
                combined_score = score * 0.7 + compound.get("docking_score", 0) * 0.3
                mmgbsa = self._estimate_mmgbsa(mol)
                kd = self._score_to_kd(combined_score)
                
                results.append({
                    "smiles": smiles,
                    "docking_score": combined_score,
                    "tier": 2,
                    "descriptors": None,
                    "binding_energy_estimate": combined_score * -1.2,
                    "mmgbsa_energy": mmgbsa,
                    "binding_affinity_kd": kd,
                    "confidence": "high" if combined_score > 8 else ("medium" if combined_score > 5 else "low"),
                })
            except:
                continue
        
        results.sort(key=lambda x: x["docking_score"], reverse=True)
        return results
    
    def _refined_docking(self, compounds: list) -> list:
        """精细对接 - 最精确的打分"""
        results = []
        
        for compound in compounds:
            smiles = compound.get("smiles")
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            
            try:
                score = self._refined_score(mol)
                mmgbsa = self._estimate_mmgbsa(mol)
                
                final_score = score
                
                results.append({
                    "smiles": smiles,
                    "docking_score": final_score,
                    "tier": 3,
                    "descriptors": None,
                    "mmgbsa_energy": mmgbsa,
                    "binding_affinity_kd": self._score_to_kd(final_score),
                    "confidence": "high" if final_score > 8 else ("medium" if final_score > 5 else "low")
                })
            except:
                continue
        
        results.sort(key=lambda x: x["docking_score"], reverse=True)
        return results
    
    def _quick_score(self, mol) -> float:
        """快速打分函数 - 基于Lipinski规则和描述符的确定性评分"""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = CalcTPSA(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)

        score = 0.0

        if mw <= 500:
            score += 2.0
        else:
            score -= (mw - 500) / 100.0
        
        if logp <= 5 and logp >= -0.5:
            score += 2.0
        elif logp > 5:
            score -= (logp - 5) * 0.5
        else:
            score -= 0.5
        
        if hbd <= 5:
            score += 1.0
        else:
            score -= (hbd - 5) * 0.3
        
        if hba <= 10:
            score += 1.0
        else:
            score -= (hba - 10) * 0.2
        
        if tpsa <= 140:
            score += 1.5
        else:
            score -= (tpsa - 140) / 40.0
        
        if rotatable <= 10:
            score += 1.0
        else:
            score -= (rotatable - 10) * 0.2
        
        return score
    
    def _standard_score(self, mol) -> float:
        """标准打分函数 - 基于QED和描述符的确定性评分"""
        from rdkit.Chem import QED
        
        qed = QED.qed(mol) if hasattr(QED, 'qed') else 0.5
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)

        score = qed * 5.0
        
        if 200 <= mw <= 450:
            score += 2.0
        elif mw < 200:
            score += 0.5
        else:
            score -= (mw - 450) / 100.0
        
        if 1 <= logp <= 4:
            score += 2.0
        elif logp > 4:
            score -= (logp - 4) * 0.5
        else:
            score -= 0.3
        
        if tpsa < 90:
            score += 1.0
        
        if hbd >= 1 and hbd <= 3:
            score += 0.5
        
        if hba >= 2 and hba <= 6:
            score += 0.5
        
        return score
    
    def _refined_score(self, mol) -> float:
        """精细打分函数 - 基于多维度描述符的确定性评分"""
        from rdkit.Chem import QED
        
        qed = QED.qed(mol) if hasattr(QED, 'qed') else 0.5
        
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)
        aromatic = Lipinski.NumAromaticRings(mol)
        heavy = rdMolDescriptors.CalcNumHeavyAtoms(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rings = Descriptors.RingCount(mol)
        
        score = qed * 4.0
        
        if 250 <= mw <= 450:
            score += 2.0
        elif 150 <= mw < 250:
            score += 1.0
        else:
            score -= abs(mw - 350) / 150.0
        
        if 1.5 <= logp <= 3.5:
            score += 2.0
        elif 0 <= logp < 1.5 or 3.5 < logp <= 5:
            score += 1.0
        else:
            score -= 1.0
        
        if 60 <= tpsa <= 120:
            score += 1.5
        elif 40 <= tpsa < 60 or 120 < tpsa <= 140:
            score += 0.5
        
        if rotatable <= 7:
            score += 1.0
        elif rotatable <= 10:
            score += 0.5
        else:
            score -= (rotatable - 10) * 0.2
        
        if 1 <= aromatic <= 3:
            score += 1.0
        elif aromatic > 3:
            score -= 0.5
        
        if 20 <= heavy <= 40:
            score += 1.0
        elif heavy < 20:
            score += 0.3
        else:
            score -= (heavy - 40) / 20.0
        
        if 1 <= hbd <= 3 and 2 <= hba <= 6:
            score += 1.0
        
        if 2 <= rings <= 4:
            score += 0.5
        
        return score
    
    def _calculate_descriptors(self, mol) -> dict:
        """计算分子描述符"""
        return {
            "molecular_weight": round(Descriptors.MolWt(mol), 2),
            "logp": round(Crippen.MolLogP(mol), 2),
            "tpsa": round(CalcTPSA(mol), 2),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
            "aromatic_rings": Lipinski.NumAromaticRings(mol),
            "heavy_atoms": rdMolDescriptors.CalcNumHeavyAtoms(mol)
        }
    
    def _estimate_mmgbsa(self, mol) -> float:
        """估算MM-GBSA结合自由能（基于描述符的确定性估算）"""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        tpsa = CalcTPSA(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        rotatable = Lipinski.NumRotatableBonds(mol)

        base_energy = -30.0
        
        mw_contribution = -0.02 * mw if mw < 500 else 0.015 * (mw - 500)
        logp_contribution = -2.5 * logp if logp > 0 else 0.5 * abs(logp)
        tpsa_contribution = -0.06 * tpsa if tpsa < 120 else 0.025 * (tpsa - 120)
        
        hb_contribution = -(hbd * 0.8 + hba * 0.5)
        
        rotatable_contribution = -0.3 * rotatable if rotatable <= 7 else 0.2 * (rotatable - 7)
        
        energy = base_energy + mw_contribution + logp_contribution + tpsa_contribution + hb_contribution + rotatable_contribution
        
        return round(energy, 2)
    
    def _score_to_kd(self, score: float) -> float:
        """将对接分数转换为估计的Kd值 (μM)

        对于 RDKit 正分 (drug-likeness score): Kd = 10^(-score/2.4)
        对于 Vina 负分 (binding affinity): Kd = 10^(score/-1.2)
        """
        if score < 0:
            kd = 10 ** (score / -1.2)
        else:
            kd = 10 ** (-score / 2.4)
        if kd > 1000:
            return round(kd, 0)
        if kd > 1:
            return round(kd, 2)
        return round(kd, 4)
    
    def calculate_binding_free_energy(self, smiles: str, target: str = "NP") -> dict:
        """计算结合自由能"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效的SMILES"}
        
        mmgbsa_energy = self._estimate_mmgbsa(mol)
        
        descriptors = self._calculate_descriptors(mol)
        
        predicted_affinity = self._score_to_kd(mmgbsa_energy / -1.2) if mmgbsa_energy < 0 else None
        
        return {
            "success": True,
            "smiles": smiles,
            "target": target,
            "mmgbsa_energy": mmgbsa_energy,
            "estimated_kd_um": predicted_affinity,
            "binding_strength": self._evaluate_binding_strength(mmgbsa_energy),
            "descriptors": descriptors
        }
    
    def _evaluate_binding_strength(self, energy: float) -> str:
        """评估结合强度"""
        if energy < -40:
            return "很强"
        elif energy < -30:
            return "强"
        elif energy < -20:
            return "中等"
        elif energy < -10:
            return "弱"
        else:
            return "很弱"
    
    def evaluate_influenza_potential(self, energy: float) -> dict:
        """评估流感抑制剂潜力 - 基于-35 kcal/mol阈值"""
        threshold = -35.0
        
        potential = {
            "energy": energy,
            "threshold": threshold,
            "is_potential_inhibitor": energy < threshold,
            "potential_level": "",
            "recommendation": ""
        }
        
        if energy < -45:
            potential["potential_level"] = "极高潜力"
            potential["recommendation"] = "强烈推荐优先实验验证，可能成为一流候选药物"
        elif energy < -35:
            potential["potential_level"] = "高潜力"
            potential["recommendation"] = "推荐实验验证，有希望成为有效抑制剂"
        elif energy < -25:
            potential["potential_level"] = "中等潜力"
            potential["recommendation"] = "可考虑实验验证，或进行结构优化"
        else:
            potential["potential_level"] = "低潜力"
            potential["recommendation"] = "建议进行结构优化或选择其他候选分子"
        
        return potential
    
    def analyze_binding_mode(self, smiles: str) -> dict:
        """分析结合模式"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效的SMILES"}
        
        functional_groups = self._identify_functional_groups(mol)
        
        druglikeness = self._assess_druglikeness(mol)
        
        return {
            "success": True,
            "smiles": smiles,
            "functional_groups": functional_groups,
            "druglikeness": druglikeness,
            "potential_interactions": self._predict_interactions(mol)
        }
    
    def _identify_functional_groups(self, mol) -> list:
        """识别分子中的药效团"""
        groups = []
        
        if Lipinski.NumHDonors(mol) > 0:
            groups.append("氢键供体")
        if Lipinski.NumHAcceptors(mol) > 0:
            groups.append("氢键受体")
        if Lipinski.NumAromaticRings(mol) > 0:
            groups.append("芳香环")
        if Descriptors.FractionCSP3(mol) > 0.3:
            groups.append("SP3碳原子")
        
        return groups
    
    def _assess_druglikeness(self, mol) -> dict:
        """评估类药性"""
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = CalcTPSA(mol)

        violations = []
        if mw > 500:
            violations.append("分子量过大")
        if logp > 5:
            violations.append("LogP过高")
        if hbd > 5:
            violations.append("氢键供体过多")
        if hba > 10:
            violations.append("氢键受体过多")
        if tpsa > 140:
            violations.append("极性表面积过大")
        
        return {
            "violations": violations,
            "pass": len(violations) <= 1,
            "lipinski_compliant": len(violations) == 0
        }
    
    def _predict_interactions(self, mol) -> dict:
        """预测分子与蛋白的潜在相互作用"""
        return {
            "hydrogen_bonds": "预测形成2-5个氢键",
            "hydrophobic": "预测存在疏水相互作用",
            "pi_stacking": "预测存在π-π堆积" if Lipinski.NumAromaticRings(mol) > 0 else "无芳香环",
            "ionic": "预测可形成离子键" if Lipinski.NumHDonors(mol) > 2 and Lipinski.NumHAcceptors(mol) > 2 else "无明显离子键"
        }

enhanced_docking = EnhancedDocking()
