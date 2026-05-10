import os
import numpy as np
from typing import Dict, List, Optional
from src.services.residue_feature_extractor import residue_feature_extractor
from src.services.ml_algorithms import (
    MLAlgorithmFactory,
    HeuristicBindingPredictor,
    train_model_on_synthetic_data,
    train_model_on_real_data
)


class MLBindingPredictor:
    """传统机器学习结合位点预测器"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.feature_extractor = residue_feature_extractor
        self.heuristic_predictor = HeuristicBindingPredictor()
        self.current_model = None
        self.current_algorithm = None
    
    def predict_binding_sites(self, pdb_file: str, 
                            algorithm: str = 'random_forest',
                            threshold: float = 0.5) -> Dict:
        """预测结合位点
        
        Args:
            pdb_file: PDB文件路径
            algorithm: 使用的算法 (random_forest, svm, gradient_boosting, heuristic)
            threshold: 预测阈值
            
        Returns:
            预测结果
        """
        
        if not os.path.exists(pdb_file):
            return {
                "success": False,
                "error": f"PDB文件不存在: {pdb_file}"
            }
        
        structure_features, residue_info = self.feature_extractor.extract_all_residues(pdb_file)
        
        if len(residue_info) == 0:
            return {
                "success": False,
                "error": "无法从PDB文件提取残基特征"
            }
        
        if algorithm == 'heuristic':
            if len(structure_features) == 0:
                return {
                    "success": False,
                    "error": "无法从PDB文件提取残基特征"
                }
            return self.heuristic_predictor.predict(structure_features, residue_info)
        
        model_path = os.path.join(self.model_dir, f'{algorithm}_binding_model.pkl')
        
        if not os.path.exists(model_path):
            train_result = train_model_on_synthetic_data(algorithm, self.model_dir)
            if not train_result.get('success'):
                return {
                    "success": False,
                    "error": f"模型不存在且训练失败: {train_result.get('error', '未知错误')}",
                    "model_path": model_path
                }
        
        try:
            predictor = MLAlgorithmFactory.create(algorithm, model_dir=self.model_dir)
            predictor.load(model_path)
            
            feature_type = getattr(predictor, 'feature_type', None)
            if feature_type == 'sequence':
                from src.services.biolip_data_loader import extract_sequence_features_for_residues
                features = extract_sequence_features_for_residues(residue_info)
            else:
                features = structure_features
            
            if len(features) == 0:
                return {
                    "success": False,
                    "error": "无法提取预测特征"
                }
            
            model_threshold = getattr(predictor, 'optimal_threshold', 0.5)
            if threshold == 0.5 and model_threshold != 0.5:
                threshold = model_threshold
            
            X_scaled = predictor.scaler.transform(features)
            proba_result = predictor.predict_proba(X_scaled)
            if proba_result.shape[1] >= 2:
                probabilities = proba_result[:, 1]
            else:
                probabilities = proba_result[:, 0]
            predictions = (probabilities >= threshold).astype(int)

            binding_count = int(predictions.sum())
            binding_ratio = binding_count / len(predictions) if len(predictions) > 0 else 0
            percentile_fallback = False

            if binding_ratio < 0.03:
                target_ratio = min(max(binding_ratio, 0.10), 0.20)
                top_n = max(int(len(probabilities) * target_ratio), 5)
                top_indices = np.argsort(probabilities)[-top_n:]
                predictions = np.zeros(len(probabilities), dtype=int)
                predictions[top_indices] = 1
                threshold = float(np.sort(probabilities)[-top_n])
                percentile_fallback = True

            binding_residues = []
            for i, (prob, pred, info) in enumerate(zip(probabilities, predictions, residue_info)):
                if pred == 1:
                    binding_residues.append({
                        **info,
                        'probability': float(prob),
                        'is_binding': True,
                        'algorithm': algorithm
                    })
            
            binding_residues.sort(key=lambda x: x['probability'], reverse=True)
            
            feature_importance = None
            if predictor.feature_importance is not None:
                feature_importance = predictor.feature_importance.tolist()
            
            return {
                "success": True,
                "algorithm": algorithm,
                "pdb_file": pdb_file,
                "total_residues": len(residue_info),
                "predicted_binding": len(binding_residues),
                "binding_residues": binding_residues[:50],
                "threshold": threshold,
                "model_optimal_threshold": model_threshold,
                "percentile_fallback": percentile_fallback,
                "model_path": model_path,
                "feature_importance": feature_importance,
                "all_probabilities": probabilities.tolist()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": f"预测失败: {str(e)}",
                "algorithm": algorithm
            }
    
    def batch_predict(self, pdb_files: List[str],
                     algorithm: str = 'random_forest',
                     threshold: float = 0.5) -> Dict:
        """批量预测多个PDB文件"""
        
        results = []
        successful = 0
        failed = 0
        
        for pdb_file in pdb_files:
            result = self.predict_binding_sites(pdb_file, algorithm, threshold)
            result['pdb_file'] = pdb_file
            
            if result.get('success'):
                successful += 1
            else:
                failed += 1
            
            results.append(result)
        
        return {
            "success": True,
            "total_files": len(pdb_files),
            "successful": successful,
            "failed": failed,
            "algorithm": algorithm,
            "results": results
        }
    
    def get_feature_info(self, pdb_file: str) -> Dict:
        """获取特征信息（不进行预测）"""
        
        if not os.path.exists(pdb_file):
            return {
                "success": False,
                "error": f"PDB文件不存在: {pdb_file}"
            }
        
        features, residue_info = self.feature_extractor.extract_all_residues(pdb_file)
        
        if len(features) == 0:
            return {
                "success": False,
                "error": "无法从PDB文件提取残基特征"
            }
        
        feature_names = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
            'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO',
            'SER', 'THR', 'TRP', 'TYR', 'VAL',
            'hydrophobicity', 'charge', 'polarity', 'volume',
            'solvent_accessibility', 'b_factor', 'depth',
            'neighbor_count', 'neighbor_density',
            'hydrophobic_neighbor_ratio', 'charged_neighbor_ratio',
            'position_ratio'
        ]
        
        feature_stats = {
            'mean': features.mean(axis=0).tolist(),
            'std': features.std(axis=0).tolist(),
            'min': features.min(axis=0).tolist(),
            'max': features.max(axis=0).tolist()
        }
        
        return {
            "success": True,
            "pdb_file": pdb_file,
            "num_residues": len(residue_info),
            "num_features": features.shape[1],
            "feature_names": feature_names,
            "feature_statistics": feature_stats,
            "residue_summary": [
                {
                    'residue_id': r['residue_num'],
                    'residue_name': r['residue_name'],
                    'chain_id': r['chain_id']
                }
                for r in residue_info[:20]
            ]
        }
    
    def train_model(self, algorithm: str = 'random_forest',
                   model_dir: Optional[str] = None,
                   data_source: str = 'synthetic',
                   biolip_path: str = 'BioLiP.txt.gz',
                   max_entries: int = 5000) -> Dict:
        """训练模型

        Args:
            algorithm: 算法名称
            model_dir: 模型保存目录
            data_source: 'synthetic' 或 'real'
            biolip_path: BioLiP 文件路径（仅 data_source='real' 时使用）
            max_entries: 最大处理条目数（仅 data_source='real' 时使用）
        """
        
        if model_dir is None:
            model_dir = self.model_dir
        
        try:
            if data_source == 'real':
                result = train_model_on_real_data(
                    algorithm=algorithm,
                    model_dir=model_dir,
                    biolip_path=biolip_path,
                    max_entries=max_entries,
                    use_sequence_only=True
                )
            else:
                result = train_model_on_synthetic_data(algorithm, model_dir)
            
            if result.get('success'):
                result['message'] = f"成功训练{algorithm}模型（数据源: {'真实BioLiP数据' if data_source == 'real' else '合成数据'}）"
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"训练失败: {str(e)}"
            }
    
    def get_supported_algorithms(self) -> Dict:
        """获取支持的算法列表"""
        algorithms = MLAlgorithmFactory.get_supported_algorithms()
        
        algorithms.append({
            'name': 'heuristic',
            'display_name': '启发式方法',
            'description': '基于规则的快速预测方法，不需要训练模型',
            'advantages': ['无需训练', '预测速度快', '可解释性强'],
            'default_params': {}
        })
        
        return {
            "success": True,
            "algorithms": algorithms
        }
    
    def get_model_info(self, algorithm: str = 'random_forest') -> Dict:
        """获取模型信息"""
        
        model_path = os.path.join(self.model_dir, f'{algorithm}_binding_model.pkl')
        
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"模型不存在: {model_path}",
                "suggestion": "请先训练模型或使用启发式方法"
            }
        
        try:
            predictor = MLAlgorithmFactory.create(algorithm, model_dir=self.model_dir)
            predictor.load(model_path)
            
            file_size = os.path.getsize(model_path) / 1024
            
            info = {
                "success": True,
                "algorithm": algorithm,
                "model_path": model_path,
                "file_size_kb": round(file_size, 2),
                "is_fitted": predictor.is_fitted,
                "has_feature_importance": predictor.feature_importance is not None
            }
            
            if predictor.feature_importance is not None:
                info['feature_importance'] = predictor.feature_importance.tolist()
            
            return info
            
        except Exception as e:
            return {
                "success": False,
                "error": f"加载模型失败: {str(e)}"
            }
    
    def delete_model(self, algorithm: str) -> Dict:
        """删除模型"""
        
        model_path = os.path.join(self.model_dir, f'{algorithm}_binding_model.pkl')
        
        if not os.path.exists(model_path):
            return {
                "success": False,
                "error": f"模型不存在: {model_path}"
            }
        
        try:
            os.remove(model_path)
            return {
                "success": True,
                "message": f"成功删除 {algorithm} 模型",
                "deleted_file": model_path
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"删除模型失败: {str(e)}"
            }
    
    def list_models(self) -> Dict:
        """列出所有已训练的模型"""
        
        models = []
        
        if os.path.exists(self.model_dir):
            for filename in os.listdir(self.model_dir):
                if filename.endswith('_binding_model.pkl'):
                    algorithm = filename.replace('_binding_model.pkl', '')
                    model_path = os.path.join(self.model_dir, filename)
                    file_size = os.path.getsize(model_path) / 1024
                    
                    models.append({
                        'algorithm': algorithm,
                        'filename': filename,
                        'path': model_path,
                        'size_kb': round(file_size, 2)
                    })
        
        return {
            "success": True,
            "total_models": len(models),
            "models": models
        }

ml_binding_predictor = MLBindingPredictor()
