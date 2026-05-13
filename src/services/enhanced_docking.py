import time
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcTPSA

logger = logging.getLogger(__name__)


class AdaptiveCutoff:
    """基于核密度估计的自适应截断策略

    对每一级筛选的得分分布进行 KDE 拟合，通过检测密度函数的局部极小值
    （分布谷底）来确定自然截断点。当分布无明显双峰结构时，退化为基于
    统计量的百分位截断。
    """

    def __init__(self, min_pass_ratio: float = 0.05,
                 max_pass_ratio: float = 0.50,
                 fallback_percentile: float = 75.0):
        self.min_pass_ratio = min_pass_ratio
        self.max_pass_ratio = max_pass_ratio
        self.fallback_percentile = fallback_percentile

    def compute_cutoff(self, scores: np.ndarray,
                       higher_is_better: bool = True) -> Tuple[float, Dict]:
        """计算自适应截断阈值

        Args:
            scores: 得分数组
            higher_is_better: True 表示得分越高越好（RDKit），
                              False 表示得分越低越好（Vina）

        Returns:
            (cutoff_value, info_dict)
        """
        scores = np.asarray(scores, dtype=float)
        n = len(scores)
        info = {
            'n_total': n,
            'method': 'unknown',
            'score_mean': float(np.mean(scores)),
            'score_std': float(np.std(scores)),
        }

        if n < 5:
            cutoff = scores[0] if n > 0 else 0.0
            info['method'] = 'too_few_samples'
            info['n_passed'] = n
            info['cutoff_value'] = float(cutoff)
            return cutoff, info

        sorted_scores = np.sort(scores)
        if not higher_is_better:
            sorted_scores = sorted_scores[::-1]

        n_pass_min = max(int(n * self.min_pass_ratio), 1)
        n_pass_max = max(int(n * self.max_pass_ratio), n_pass_min + 1)

        kde_cutoff = self._kde_valley_cutoff(sorted_scores, n_pass_min, n_pass_max)
        if kde_cutoff is not None:
            cutoff_value = kde_cutoff['cutoff']
            n_passed = kde_cutoff['n_passed']
            info['method'] = 'kde_valley'
            info['kde_bandwidth'] = kde_cutoff['bandwidth']
            info['kde_valley_position'] = float(kde_cutoff['cutoff'])
        else:
            fallback_idx = max(n_pass_min - 1,
                               min(int(n * self.fallback_percentile / 100.0), n - 1))
            if higher_is_better:
                cutoff_value = sorted_scores[fallback_idx]
            else:
                cutoff_value = sorted_scores[fallback_idx]
            n_passed = fallback_idx + 1
            info['method'] = 'percentile_fallback'

        info['cutoff_value'] = float(cutoff_value)
        info['n_passed'] = int(n_passed)
        info['pass_ratio'] = float(n_passed / n)
        return cutoff_value, info

    def _kde_valley_cutoff(self, sorted_scores: np.ndarray,
                           n_pass_min: int,
                           n_pass_max: int) -> Optional[Dict]:
        """使用 KDE 检测得分分布中的谷底作为截断点

        步骤:
        1. 用 Silverman 法则计算带宽 h
        2. 在得分范围内等距采样，计算 KDE 密度
        3. 检测密度曲线的局部极小值
        4. 选择最靠近"高分区域边缘"的谷底作为截断点
        """
        n = len(sorted_scores)
        if n < 10:
            return None

        score_range = sorted_scores[-1] - sorted_scores[0]
        if score_range < 1e-8:
            return None

        std = np.std(sorted_scores)
        h = 1.06 * std * (n ** (-1.0 / 5.0))
        if h < score_range * 0.01:
            h = score_range * 0.05
        if h > score_range * 0.5:
            h = score_range * 0.2

        n_eval = min(200, n * 2)
        eval_points = np.linspace(sorted_scores[0], sorted_scores[-1], n_eval)

        density = np.zeros(n_eval)
        for i, x in enumerate(eval_points):
            u = (x - sorted_scores) / h
            density[i] = np.sum(np.exp(-0.5 * u ** 2)) / (n * h * np.sqrt(2 * np.pi))

        local_minima = []
        for i in range(1, n_eval - 1):
            if density[i] < density[i - 1] and density[i] < density[i + 1]:
                local_minima.append(i)

        if not local_minima:
            return None

        peak_idx = int(np.argmax(density[:n_eval // 2 + n_eval // 4]))

        best_valley = None
        for valley_idx in local_minima:
            n_above = int(np.searchsorted(sorted_scores, eval_points[valley_idx]))
            if n_pass_min <= (n - n_above) <= n_pass_max:
                if best_valley is None:
                    best_valley = valley_idx
                    break

        if best_valley is None:
            for valley_idx in local_minima:
                n_above = int(np.searchsorted(sorted_scores, eval_points[valley_idx]))
                n_passed = n - n_above
                if n_passed >= n_pass_min:
                    best_valley = valley_idx
                    break

        if best_valley is None:
            return None

        cutoff_score = eval_points[best_valley]
        n_passed = int(np.sum(sorted_scores >= cutoff_score))

        n_passed = max(n_pass_min, min(n_passed, n_pass_max))

        return {
            'cutoff': float(cutoff_score),
            'n_passed': n_passed,
            'bandwidth': float(h),
            'valley_density': float(density[best_valley]),
        }

    def apply_cutoff(self, compounds: list, score_key: str = 'docking_score',
                     higher_is_better: bool = True) -> Tuple[list, Dict]:
        """对化合物列表应用自适应截断

        Args:
            compounds: 化合物结果列表，每个元素包含 score_key 对应的得分
            score_key: 得分字段名
            higher_is_better: True 表示得分越高越好

        Returns:
            (passed_compounds, cutoff_info)
        """
        if len(compounds) <= 3:
            return compounds, {
                'n_total': len(compounds),
                'n_passed': len(compounds),
                'method': 'too_few_samples',
                'cutoff_value': None,
            }

        scores = np.array([c.get(score_key, 0) for c in compounds])
        cutoff_value, info = self.compute_cutoff(scores, higher_is_better)

        if higher_is_better:
            passed = [c for c in compounds if c.get(score_key, 0) >= cutoff_value]
        else:
            passed = [c for c in compounds if c.get(score_key, 0) <= cutoff_value]

        n_target = info['n_passed']
        if len(passed) != n_target and len(compounds) > n_target:
            sorted_compounds = sorted(compounds,
                                      key=lambda c: c.get(score_key, 0),
                                      reverse=higher_is_better)
            passed = sorted_compounds[:n_target]

        return passed, info


adaptive_cutoff = AdaptiveCutoff()

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
                       tier3_count: int = 20,
                       adaptive: bool = False) -> dict:
        """分级对接策略

        Tier1: RDKit 描述符快速打分（始终使用）
        Tier2: 当提供 target_pdb 时用 Vina 标准精度对接，否则用 RDKit
        Tier3: 当提供 target_pdb 时用 Vina 精细对接（高 exhaustiveness），否则用 RDKit

        Args:
            adaptive: 是否启用自适应截断模式。启用后，每一级的通过数量
                      不再使用固定阈值，而是基于得分分布的 KDE 谷底检测
                      自动确定。
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
            "adaptive": adaptive,
        }

        adaptive_info = {}
        
        tier1_results = self._high_throughput_screen(smiles_list, tier1_count)
        results["tier1"]["compounds"] = tier1_results

        if adaptive and len(tier1_results) > 5:
            tier1_cutoff, tier1_info = adaptive_cutoff.compute_cutoff(
                np.array([c.get("docking_score", 0) for c in tier1_results]),
                higher_is_better=True
            )
            adaptive_info["tier1"] = tier1_info
            tier1_passed = [c for c in tier1_results if c.get("docking_score", 0) >= tier1_cutoff]
            n_t1 = max(tier1_info['n_passed'], 5)
            if len(tier1_passed) < n_t1:
                tier1_passed = sorted(tier1_results, key=lambda c: c.get("docking_score", 0), reverse=True)[:n_t1]
            tier2_input = tier1_passed
        else:
            tier2_input = tier1_results[:tier2_count]
        
        if use_vina:
            tier2_results = self._vina_docking(
                tier2_input, target_pdb,
                binding_site_coords, center, box_size,
                exhaustiveness=8
            )
        else:
            tier2_results = self._standard_docking(tier2_input)
        results["tier2"]["compounds"] = tier2_results

        if adaptive and len(tier2_results) > 5:
            score_key = "docking_score"
            tier2_cutoff, tier2_info = adaptive_cutoff.compute_cutoff(
                np.array([c.get(score_key, 0) for c in tier2_results]),
                higher_is_better=not use_vina
            )
            adaptive_info["tier2"] = tier2_info
            if use_vina:
                tier2_passed = [c for c in tier2_results if c.get(score_key, 0) <= tier2_cutoff]
            else:
                tier2_passed = [c for c in tier2_results if c.get(score_key, 0) >= tier2_cutoff]
            n_t2 = max(tier2_info['n_passed'], 3)
            if len(tier2_passed) < n_t2:
                tier2_passed = sorted(tier2_results,
                                      key=lambda c: c.get(score_key, 0),
                                      reverse=(not use_vina))[:n_t2]
            tier3_input = tier2_passed
        else:
            tier3_input = tier2_results[:tier3_count]
        
        if use_vina:
            tier3_results = self._vina_docking(
                tier3_input, target_pdb,
                binding_site_coords, center, box_size,
                exhaustiveness=32
            )
        else:
            tier3_results = self._refined_docking(tier3_input)
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
            "processing_time": round(time.time() - start_time, 2),
            "cutoff_mode": "adaptive_kde" if adaptive else "fixed_threshold",
        }

        if adaptive and adaptive_info:
            results["summary"]["adaptive_cutoff_details"] = adaptive_info
        
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
