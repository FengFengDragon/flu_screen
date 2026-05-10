import json
from datetime import datetime
from typing import Dict, List, Optional
from src import db
from src.models.experiment import Experiment, ExperimentResult

class ExperimentTracker:
    """实验追踪与模型迭代服务"""
    
    def __init__(self):
        self.feedback_history = []
    
    def create_experiment_plan(self, candidate_compounds: List[Dict], 
                                experiment_type: str = "in_vitro") -> Dict:
        """生成实验计划"""
        return self._default_experiment_plan(candidate_compounds, experiment_type)
    
    def _default_experiment_plan(self, candidates: List[Dict], experiment_type: str) -> Dict:
        """默认实验计划"""
        return {
            "plan_name": "蛋白质分子动力学模拟验证实验",
            "methodology": "MD模拟验证",
            "cell_lines": ["MDCK", "A549"],
            "virus_strains": ["H1N1", "H3N2"],
            "concentration_range": {"min": 0.1, "max": 100, "unit": "μM"},
            "controls": {"positive": "奥司他韦", "negative": "DMSO"},
            "readouts": ["结合自由能", "RMSD", "氢键数量", "疏水相互作用"],
            "success_criteria": "MM-GBSA能量 < -35 kcal/mol 为有效化合物"
        }
    
    def record_experiment_result(self, compound_smiles: str, 
                                   experiment_data: Dict) -> Dict:
        """记录实验结果"""
        result_entry = {
            "smiles": compound_smiles,
            "experiment_date": datetime.utcnow().isoformat(),
            "data": experiment_data,
            "status": "recorded"
        }
        
        self.feedback_history.append(result_entry)
        
        return {
            "success": True,
            "record_id": len(self.feedback_history),
            "message": "实验结果已记录"
        }
    
    def analyze_experiment_results(self, results: List[Dict]) -> Dict:
        """分析实验结果"""
        if not results:
            return {"error": "无实验结果"}
        
        valid_results = [r for r in results if r.get("ic50") is not None or r.get("binding_energy") is not None]
        
        if not valid_results:
            return {"message": "无有效数据"}
        
        ic50_values = [r["ic50"] for r in valid_results if r.get("ic50") is not None]
        energy_values = [r["binding_energy"] for r in valid_results if r.get("binding_energy") is not None]
        
        active_compounds = [r for r in valid_results if r.get("ic50", float('inf')) < 50]
        high_affinity = [r for r in valid_results if r.get("binding_energy", 0) < -35]
        
        analysis = {
            "total_experiments": len(results),
            "valid_results": len(valid_results),
            "active_compounds": len(active_compounds),
            "high_affinity_compounds": len(high_affinity),
            "statistics": {
                "mean_ic50": round(sum(ic50_values) / len(ic50_values), 2) if ic50_values else None,
                "min_ic50": min(ic50_values) if ic50_values else None,
                "mean_energy": round(sum(energy_values) / len(energy_values), 2) if energy_values else None,
                "min_energy": min(energy_values) if energy_values else None
            },
            "active_list": [
                {
                    "smiles": c.get("smiles"),
                    "ic50": c.get("ic50"),
                    "binding_energy": c.get("binding_energy"),
                    "si": c.get("si")
                }
                for c in active_compounds
            ]
        }
        
        return analysis
    
    def generate_feedback(self, experiment_results: List[Dict]) -> Dict:
        """生成模型优化反馈"""
        analysis = self.analyze_experiment_results(experiment_results)
        
        if "error" in analysis:
            return {"error": analysis["error"]}
        
        feedback = {
            "experiment_analysis": analysis,
            "model_evaluation": {
                "prediction_accuracy": "基于实验数据分析",
                "false_positives": [],
                "false_negatives": []
            },
            "improvements": [
                "增加轨迹采样密度以提高精度",
                "使用更精确的力场参数",
                "延长MD模拟时间"
            ],
            "next_round_suggestions": {
                "focus_on": "结合自由能低于-35 kcal/mol的化合物",
                "exclude": "能量高于-20 kcal/mol的化合物",
                "new_requirements": "增加结合口袋分析"
            },
            "sar_analysis": "基于分子动力学模拟结果的结构-活性关系分析"
        }
        
        return feedback
    
    def update_model_with_feedback(self, feedback: Dict) -> Dict:
        """使用反馈更新模型参数"""
        analysis = feedback.get("experiment_analysis", {})
        
        model_updates = {
            "timestamp": datetime.utcnow().isoformat(),
            "updates": [],
            "new_parameters": {}
        }
        
        if analysis.get("active_compounds"):
            active = analysis["active_compounds"]
            ic50_values = [c["ic50"] for c in active if c.get("ic50")]
            
            if ic50_values:
                avg_ic50 = sum(ic50_values) / len(ic50_values)
                model_updates["updates"].append(f"发现{len(active)}个活性化合物，平均IC50: {avg_ic50:.2f}μM")
                model_updates["new_parameters"]["ic50_threshold"] = max(avg_ic50 * 1.5, 10)
        
        if analysis.get("high_affinity_compounds"):
            energy_values = [r["binding_energy"] for r in analysis["active_list"] if r.get("binding_energy")]
            
            if energy_values:
                avg_energy = sum(energy_values) / len(energy_values)
                model_updates["updates"].append(f"发现{len(energy_values)}个高亲和力化合物，平均能量: {avg_energy:.2f} kcal/mol")
                model_updates["new_parameters"]["energy_threshold"] = max(avg_energy * 0.9, -40)
        
        improvements = feedback.get("improvements", [])
        for imp in improvements:
            model_updates["updates"].append(imp)
        
        return model_updates
    
    def generate_final_report(self, workflow_results: Dict, 
                              experiment_results: List[Dict] = None) -> str:
        """生成最终筛选报告"""
        
        report_parts = [
            "# 蛋白质分子动力学模拟与AI分析报告",
            "",
            f"生成时间: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## 1. 执行摘要",
            "",
            "本报告总结了基于分子动力学模拟和AI分析的蛋白质-配体相互作用研究。",
            "",
            "## 2. 模拟参数",
            "",
            f"模拟时间: {workflow_results.get('simulation_time', 'N/A')}",
            f"温度: {workflow_results.get('temperature', '310K')}",
            f"压力: {workflow_results.get('pressure', '1 atm')}",
            f"力场: {workflow_results.get('forcefield', 'AMBER99SB-ILDN')}",
            "",
            "## 3. 轨迹分析结果",
            "",
            f"总帧数: {workflow_results.get('total_frames', 'N/A')}",
            f"关键帧数: {workflow_results.get('key_frames', 'N/A')}",
            "",
            "## 4. 结合自由能分析",
            "",
            f"最佳结合自由能: {workflow_results.get('best_energy', 'N/A')} kcal/mol",
            f"平均结合自由能: {workflow_results.get('avg_energy', 'N/A')} kcal/mol",
            "",
            "## 5. 候选化合物排名",
            ""
        ]
        
        if experiment_results:
            report_parts.extend([
                "## 6. 实验验证结果",
                "",
                f"总实验数: {len(experiment_results)}",
                f"有效结果: {len([r for r in experiment_results if r.get('binding_energy')])}",
                ""
            ])
        
        report_parts.extend([
            "## 7. 结论与建议",
            "",
            "基于分子动力学模拟和轨迹分析，建议选择结合自由能低于-35 kcal/mol的化合物进行实验验证。",
            "",
            "## 8. 下一步计划",
            "",
            "1. 对高亲和力化合物进行体外实验验证",
            "2. 基于实验结果优化MD模拟参数",
            "3. 扩大化合物库进行新一轮筛选",
            ""
        ])
        
        return "\n".join(report_parts)
    
    def get_workflow_status(self) -> Dict:
        """获取工作流状态"""
        return {
            "feedback_records": len(self.feedback_history),
            "last_update": self.feedback_history[-1]["experiment_date"] if self.feedback_history else None,
            "status": "ready"
        }

experiment_tracker = ExperimentTracker()
