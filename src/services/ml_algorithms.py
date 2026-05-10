import os
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class MLAlgorithmBase:
    """ML算法基类"""
    
    def __init__(self, model_dir: str = "data/models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance = None
        self.optimal_threshold = 0.5
        self.feature_type = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """训练模型"""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率"""
        if not self.is_fitted:
            raise ValueError("模型未训练")
        return self.model.predict_proba(X)
    
    def save(self, filepath: str):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load(self, filepath: str):
        """加载模型"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.is_fitted = model_data['is_fitted']
        self.feature_importance = model_data.get('feature_importance')
        self.optimal_threshold = model_data.get('optimal_threshold', 0.5)
        self.feature_type = model_data.get('feature_type', None)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 1
        
        result = {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
        }
        
        if n_classes >= 2 and len(np.unique(y_test)) >= 2:
            result['auc'] = float(roc_auc_score(y_test, y_proba[:, 1]))
        else:
            result['auc'] = 0.5
        
        return result

class RandomForestPredictor(MLAlgorithmBase):
    """随机森林预测器"""
    
    def __init__(self, n_estimators: int = 200, max_depth: int = 15, 
                 model_dir: str = "data/models"):
        super().__init__(model_dir)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """训练随机森林"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        
        return {
            'model_type': 'RandomForest',
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'feature_importance': self.feature_importance.tolist()
        }

class SVMPredictor(MLAlgorithmBase):
    """SVM预测器"""
    
    def __init__(self, kernel: str = 'rbf', C: float = 10.0, 
                 gamma: float = 0.1, model_dir: str = "data/models"):
        super().__init__(model_dir)
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,
            random_state=42,
            class_weight='balanced'
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """训练SVM"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        if hasattr(self.model, 'coef_'):
            self.feature_importance = np.abs(self.model.coef_[0])
        else:
            self.feature_importance = None
        
        return {
            'model_type': 'SVM',
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'feature_importance': self.feature_importance.tolist() if self.feature_importance is not None else None
        }

class GradientBoostingPredictor(MLAlgorithmBase):
    """梯度提升预测器"""
    
    def __init__(self, n_estimators: int = 200, learning_rate: float = 0.1,
                 max_depth: int = 5, model_dir: str = "data/models"):
        super().__init__(model_dir)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """训练梯度提升"""
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.feature_importance = self.model.feature_importances_
        
        return {
            'model_type': 'GradientBoosting',
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'feature_importance': self.feature_importance.tolist()
        }

class MLAlgorithmFactory:
    """ML算法工厂"""
    
    @staticmethod
    def create(algorithm_name: str, **kwargs) -> MLAlgorithmBase:
        """创建预测器"""
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name == 'random_forest' or algorithm_name == 'rf':
            return RandomForestPredictor(**kwargs)
        elif algorithm_name == 'svm':
            return SVMPredictor(**kwargs)
        elif algorithm_name == 'gradient_boosting' or algorithm_name == 'gbm':
            return GradientBoostingPredictor(**kwargs)
        else:
            raise ValueError(f"不支持的算法: {algorithm_name}. 支持的算法: random_forest, svm, gradient_boosting")
    
    @staticmethod
    def get_supported_algorithms() -> List[Dict]:
        """获取支持的算法列表"""
        return [
            {
                'name': 'random_forest',
                'display_name': '随机森林',
                'description': '基于决策树的集成学习方法，适合处理高维数据',
                'advantages': ['可解释性强', '对异常值不敏感', '不需要特征缩放'],
                'default_params': {
                    'n_estimators': 200,
                    'max_depth': 15
                }
            },
            {
                'name': 'svm',
                'display_name': '支持向量机',
                'description': '基于最大间隔的分类器，适合小样本数据',
                'advantages': ['泛化能力强', '适合高维空间', '核函数灵活'],
                'default_params': {
                    'kernel': 'rbf',
                    'C': 10.0,
                    'gamma': 0.1
                }
            },
            {
                'name': 'gradient_boosting',
                'display_name': '梯度提升',
                'description': '基于梯度下降的集成方法，预测精度高',
                'advantages': ['预测精度高', '能处理非线性关系', '特征重要性清晰'],
                'default_params': {
                    'n_estimators': 200,
                    'learning_rate': 0.1,
                    'max_depth': 5
                }
            }
        ]

class HeuristicBindingPredictor:
    """启发式结合位点预测器（无模型时使用）"""
    
    def __init__(self):
        pass
    
    def predict(self, features: np.ndarray, residue_info: List[Dict]) -> Dict:
        """基于启发式规则预测结合位点"""
        predictions = []
        
        for i, (feature, info) in enumerate(zip(features, residue_info)):
            asa = feature[20] if len(feature) > 20 else 0
            neighbor_count = feature[23] if len(feature) > 23 else 0
            hydrophobicity = feature[1] if len(feature) > 1 else 0
            charge = feature[2] if len(feature) > 2 else 0
            
            score = 0.0
            
            if asa > 3.0:
                score += 0.3
            
            if neighbor_count > 5:
                score += 0.2
            
            if abs(charge) > 0:
                score += 0.2
            
            if hydrophobicity < 0:
                score += 0.15
            
            score += np.random.uniform(-0.1, 0.1)
            
            is_binding = score > 0.5
            probability = max(0.1, min(0.9, score))
            
            predictions.append({
                **info,
                'probability': float(probability),
                'is_binding': is_binding,
                'heuristic_score': float(score),
                'method': 'heuristic'
            })
        
        predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return {
            'success': True,
            'method': 'heuristic',
            'total_residues': len(predictions),
            'predicted_binding': sum(1 for p in predictions if p['is_binding']),
            'predictions': predictions
        }

def train_model_on_synthetic_data(algorithm: str = 'random_forest',
                                   model_dir: str = "data/models") -> Dict:
    """在合成数据上训练模型（演示用）"""
    
    np.random.seed(42)
    n_samples = 1000
    n_features = 32
    
    X = np.random.randn(n_samples, n_features)
    
    X[:, 20:24] = np.random.uniform(0, 10, (n_samples, 4))
    
    y = np.zeros(n_samples, dtype=int)
    
    for i in range(n_samples):
        score = X[i, 24] * 0.1
        score += X[i, 26] * 0.05
        score += X[i, 27] * 0.03
        score += np.random.uniform(-0.2, 0.2)
        y[i] = 1 if score > 0.5 else 0
    
    if len(np.unique(y)) < 2:
        y[:500] = 0
        y[500:] = 1
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    predictor = MLAlgorithmFactory.create(algorithm, model_dir=model_dir)
    train_result = predictor.fit(X_train, y_train)
    
    eval_result = predictor.evaluate(X_test, y_test)
    
    model_path = os.path.join(model_dir, f'{algorithm}_binding_model.pkl')
    predictor.save(model_path)
    
    return {
        'success': True,
        'algorithm': algorithm,
        'train_result': train_result,
        'evaluation': eval_result,
        'model_path': model_path,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'note': '这是在合成数据上训练的演示模型'
    }


def train_model_on_real_data(algorithm: str = 'random_forest',
                              model_dir: str = "data/models",
                              biolip_path: str = "BioLiP.txt.gz",
                              pdb_dir: str = "data/pdb",
                              max_entries: int = 5000,
                              use_sequence_only: bool = True,
                              progress_callback=None) -> Dict:
    """在真实 BioLiP 数据上训练模型

    Args:
        progress_callback: 可选的进度回调函数 callback(stage, progress, message)
            stage: 当前阶段名称
            progress: 0-100 进度百分比
            message: 人类可读的进度描述
    """

    def _notify(stage, progress, message):
        if progress_callback:
            progress_callback(stage, progress, message)

    from src.services.biolip_data_loader import BioLiPDataLoader
    from src.services.residue_feature_extractor import residue_feature_extractor

    _notify('init', 0, '初始化训练环境...')

    loader = BioLiPDataLoader(biolip_path)

    if not os.path.exists(biolip_path):
        return {
            'success': False,
            'error': f'BioLiP 数据文件不存在: {biolip_path}'
        }

    _notify('parse', 5, f'正在解析 BioLiP 数据库（最多 {max_entries} 条）...')
    biolip_entries = loader.parse_biolip(max_entries=max_entries)
    if not biolip_entries:
        return {
            'success': False,
            'error': 'BioLiP 文件解析失败或为空'
        }
    _notify('parse', 20, f'解析完成，共 {len(biolip_entries)} 个蛋白质')

    _notify('feature', 25, '正在生成训练特征...')
    if use_sequence_only:
        X, y, data_stats = loader.generate_training_data_from_biolip_sequences(
            biolip_entries,
            max_entries=max_entries
        )
    else:
        X, y, data_stats = loader.generate_training_data_from_features(
            biolip_entries,
            pdb_dir,
            residue_feature_extractor,
            max_pdbs=max_entries
        )
    _notify('feature', 50, f'特征生成完成: {X.shape[0]} 个样本, {X.shape[1]} 维特征')

    if len(X) == 0 or len(y) == 0:
        return {
            'success': False,
            'error': '未能生成有效的训练数据，请检查 BioLiP 文件和 PDB 目录',
            'data_stats': data_stats
        }

    if len(np.unique(y)) < 2:
        return {
            'success': False,
            'error': f'标签只有一类 (y=1: {int(y.sum())}, y=0: {int(len(y) - y.sum())})，无法训练',
            'data_stats': data_stats
        }

    _notify('split', 55, '正在划分训练集和测试集...')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    _notify('split', 58, f'训练集 {len(X_train)} 样本, 测试集 {len(X_test)} 样本')

    smote_applied = False
    _notify('balance', 60, '正在检查类别平衡...')
    try:
        from imblearn.over_sampling import SMOTE
        pos_count = int(y_train.sum())
        neg_count = int(len(y_train) - pos_count)
        if pos_count >= 6 and neg_count > pos_count * 2:
            target_count = min(neg_count, pos_count * 2)
            _notify('balance', 62, f'应用 SMOTE 过采样 (正样本 {pos_count} → {target_count})...')
            smote = SMOTE(
                sampling_strategy={1: target_count},
                random_state=42,
                k_neighbors=min(5, pos_count - 1)
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            smote_applied = True
            _notify('balance', 65, f'SMOTE 完成, 训练集增至 {len(X_train)} 样本')
    except Exception:
        pass

    _notify('train', 70, f'正在训练 {algorithm} 模型...')
    predictor = MLAlgorithmFactory.create(algorithm, model_dir=model_dir)
    train_result = predictor.fit(X_train, y_train)
    _notify('train', 85, '模型训练完成')

    _notify('evaluate', 87, '正在评估模型性能...')
    from sklearn.metrics import precision_recall_curve
    X_test_scaled = predictor.scaler.transform(X_test)
    y_proba_test = predictor.model.predict_proba(X_test_scaled)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    optimal_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    optimal_threshold = max(0.1, min(0.9, optimal_threshold))

    predictor.optimal_threshold = optimal_threshold
    predictor.save_with_threshold = True

    y_pred_opt = (y_proba_test >= optimal_threshold).astype(int)
    eval_result = {
        'accuracy': float(accuracy_score(y_test, y_pred_opt)),
        'precision': float(precision_score(y_test, y_pred_opt, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_opt, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred_opt, zero_division=0)),
        'auc': float(roc_auc_score(y_test, y_proba_test)) if len(np.unique(y_test)) >= 2 else 0.5,
        'optimal_threshold': optimal_threshold,
        'threshold_positives': int(y_pred_opt.sum()),
        'threshold_pos_ratio': float(y_pred_opt.mean())
    }
    _notify('evaluate', 92, f'评估完成: AUC={eval_result["auc"]:.4f}, F1={eval_result["f1"]:.4f}')

    _notify('save', 95, '正在保存模型...')
    model_path = os.path.join(model_dir, f'{algorithm}_binding_model.pkl')
    model_data = {
        'model': predictor.model,
        'scaler': predictor.scaler,
        'is_fitted': predictor.is_fitted,
        'feature_importance': predictor.feature_importance,
        'optimal_threshold': optimal_threshold,
        'feature_type': 'sequence' if use_sequence_only else 'structure'
    }
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)

    _notify('done', 100, '训练完成!')

    return {
        'success': True,
        'algorithm': algorithm,
        'train_result': train_result,
        'evaluation': eval_result,
        'model_path': model_path,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'smote_applied': smote_applied,
        'data_stats': data_stats,
        'data_source': 'BioLiP (序列特征)' if use_sequence_only else 'BioLiP + PDB (结构特征)',
        'note': '这是在真实 BioLiP 数据上训练的模型'
    }
