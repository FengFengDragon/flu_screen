import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors

class MolecularDynamics:
    """分子动力学模拟服务 - 评估结合稳定性"""
    
    def __init__(self):
        pass
    
    def simulate_binding_stability(self, smiles: str, simulation_steps: int = 1000) -> dict:
        """模拟结合稳定性 - 简化的动力学模拟"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效的SMILES"}
        
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 生成3D构象
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        
        # 模拟分子柔性
        conformers = []
        for i in range(min(10, simulation_steps // 100)):
            try:
                conf_id = AllChem.EmbedMolecule(mol, randomSeed=i)
                if conf_id >= 0:
                    conf = AllChem.EmbedMultipleConfs(mol, numConfs=10, randomSeed=i)[0]
                    conformers.append({
                        "conformer_id": i,
                        "energy": conf.GetDoubleProp('Energy'),
                        "rmsd": conf.GetDoubleProp('RMSD') if hasattr(conf, 'GetDoubleProp') else 0
                    })
            except:
                pass
        
        # 分析构象能量分布
        energies = [c["energy"] for c in conformers if c["energy"] is not None]
        rmsds = [c["rmsd"] for c in conformers]
        
        energy_range = max(energies) - min(energies) if energies else 0
        energy_variance = np.var(energies) if energies else 0
        avg_rmsd = np.mean(rmsds) if rmsds else 0
        
        stability_score = self._calculate_stability_score(energy_range, energy_variance, avg_rmsd)
        
        return {
            "success": True,
            "smiles": smiles,
            "simulation_steps": simulation_steps,
            "num_conformers": len(conformers),
            "energy_range": round(energy_range, 2),
            "energy_variance": round(energy_variance, 2),
            "avg_rmsd": round(avg_rmsd, 3),
            "stability_score": stability_score,
            "stability_level": self._evaluate_stability(stability_score),
            "recommendation": self._get_stability_recommendation(stability_score)
        }
    
    def _calculate_stability_score(self, energy_range: float, energy_variance: float, avg_rmsd: float) -> float:
        """计算稳定性分数"""
        score = 100.0
        
        # 能量范围越小越稳定
        if energy_range < 5:
            score += 30
        elif energy_range < 10:
            score += 20
        elif energy_range < 20:
            score += 10
        
        # 方差越小越稳定
        if energy_variance < 2:
            score += 20
        elif energy_variance < 5:
            score += 10
        
        # RMSD越小越稳定
        if avg_rmsd < 0.5:
            score += 15
        elif avg_rmsd < 1.0:
            score += 10
        
        return score
    
    def _evaluate_stability(self, score: float) -> str:
        """评估稳定性等级"""
        if score >= 80:
            return "很稳定"
        elif score >= 60:
            return "稳定"
        elif score >= 40:
            return "中等稳定"
        elif score >= 20:
            return "不稳定"
        else:
            return "很稳定"
    
    def _get_stability_recommendation(self, score: float) -> str:
        """获取稳定性建议"""
        if score >= 80:
            return "分子构象稳定，推荐进行实验验证"
        elif score >= 60:
            return "分子构象较稳定，可考虑实验验证"
        elif score >= 40:
            return "分子构象不稳定，建议优化或选择其他分子"
        else:
            return "分子构象很稳定，强烈建议重新设计"
    
    def analyze_binding_pocket(self, smiles: str, target_pocket: str = "NP") -> dict:
        """分析结合口袋特征"""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {"success": False, "error": "无效的SMILES"}
        
        # 计算分子性质
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        rotatable = Descriptors.NumRotatableBonds(mol)
        
        # 评估结合口袋兼容性
        pocket_compatibility = 100.0
        
        # 分子量适中
        if 300 <= mw <= 450:
            pocket_compatibility += 20
        
        # LogP适中
        if 2 <= logp <= 4:
            pocket_compatibility += 20
        
        # TPSA适中
        if 80 <= tpsa <= 120:
            pocket_compatibility += 20
        
        # 柔性适中
        if rotatable <= 5:
            pocket_compatibility += 20
        
        # 分子形状分析
        from rdkit.Chem import rdMolDescriptors
        shape_complement = self._analyze_shape(mol)
        pocket_compatibility += shape_complement * 0.2
        
        return {
            "success": True,
            "smiles": smiles,
            "target_pocket": target_pocket,
            "molecular_properties": {
                "mw": round(mw, 2),
                "logp": round(logp, 2),
                "tpsa": round(tpsa, 2),
                "rotatable_bonds": rotatable
            },
            "pocket_compatibility": round(pocket_compatibility, 2),
            "compatibility_level": self._evaluate_compatibility(pocket_compatibility),
            "shape_analysis": shape_complement,
            "recommendation": self._get_pocket_recommendation(pocket_compatibility)
        }
    
    def _analyze_shape(self, mol) -> float:
        """分析分子形状互补性"""
        try:
            from rdkit.Chem import rdMolDescriptors
            
            # 计算形状描述符
            asphericity = rdMolDescriptors.CalcAsphericity(mol)
            eccentricity = rdMolDescriptors.CalcEccentricity(mol)
            spherocity_index = rdMolDescriptors.CalcSpherocityIndex(mol)
            
            # 形状得分（归一化）
            shape_score = (1.0 - asphericity + (1.0 - eccentricity)) / 2.0
            
            return shape_score
        except:
            return 0.5
    
    def _evaluate_compatibility(self, score: float) -> str:
        """评估兼容性等级"""
        if score >= 90:
            return "高度兼容"
        elif score >= 70:
            return "良好兼容"
        elif score >= 50:
            return "中等兼容"
        elif score >= 30:
            return "低兼容"
        else:
            return "不兼容"
    
    def _get_pocket_recommendation(self, score: float) -> str:
        """获取结合口袋建议"""
        if score >= 90:
            return "分子与靶点口袋高度匹配，推荐优先实验"
        elif score >= 70:
            return "分子与靶点口袋匹配良好，建议实验验证"
        elif score >= 50:
            return "分子与靶点口袋匹配一般，建议优化结构"
        else:
            return "分子与靶点口袋匹配较差，建议选择其他分子"
    
    def batch_stability_simulation(self, smiles_list: list) -> dict:
        """批量稳定性模拟"""
        results = []
        
        for i, smiles in enumerate(smiles_list):
            if i < 3:  # 只对前3个分子进行详细模拟
                result = self.simulate_binding_stability(smiles, simulation_steps=500)
                results.append(result)
            else:
                # 其他分子快速评估
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    result = {
                        "success": True,
                        "smiles": smiles,
                        "stability_score": 60.0,
                        "stability_level": "快速评估",
                        "note": "简化评估"
                    }
                    results.append(result)
        
        return {
            "total": len(smiles_list),
            "detailed_simulations": len([r for r in results if "simulation_steps" in r]),
            "results": results,
            "summary": self._summarize_stability(results)
        }
    
    def _summarize_stability(self, results: list) -> dict:
        """总结稳定性模拟结果"""
        stable_count = len([r for r in results if r.get("stability_level") in ["很稳定", "稳定"]])
        unstable_count = len([r for r in results if r.get("stability_level") in ["不稳定", "很稳定"]])
        
        return {
            "stable_molecules": stable_count,
            "unstable_molecules": unstable_count,
            "stability_rate": round(stable_count / len(results) * 100, 2) if results else 0,
            "recommendation": "建议对稳定分子进行实验验证" if stable_count > 0 else "需要重新设计候选分子"
        }

molecular_dynamics = MolecularDynamics()
