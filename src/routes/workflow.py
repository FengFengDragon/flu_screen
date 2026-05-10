from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
from src.services.target_service import target_service
from src.services.docking import molecular_docking
from src.services.enhanced_docking import enhanced_docking
from src.services.experiment_tracker import experiment_tracker
from src.services.advanced_features import advanced_features
from src.services.molecular_dynamics import molecular_dynamics
from src.services.pdb_analyzer import pdb_analyzer

bp = Blueprint('workflow', __name__, url_prefix='/api/workflow')

@bp.route('/targets', methods=['GET'])
def list_targets():
    """获取可用的病毒靶点"""
    targets = target_service.list_targets()
    return jsonify({
        "targets": targets,
        "recommendation": "NP（核蛋白）是高度保守的靶点，推荐作为首选"
    })

@bp.route('/targets/<target_code>', methods=['GET'])
def get_target(target_code):
    """获取特定靶点详情"""
    target = target_service.get_target_info(target_code.upper())
    if target:
        return jsonify(target)
    return jsonify({"error": "未找到该靶点"}), 404

@bp.route('/targets/<target_code>/download', methods=['POST'])
def download_target_pdb(target_code):
    """下载靶点PDB结构"""
    result = target_service.download_pdb_structure(target_code.upper())
    return jsonify(result)

@bp.route('/docking/tiered', methods=['POST'])
def tiered_docking():
    """分级对接"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    target = data.get('target', 'NP')
    
    result = enhanced_docking.tiered_docking(
        smiles_list, 
        target,
        tier1_count=data.get('tier1_count', 500),
        tier2_count=data.get('tier2_count', 100),
        tier3_count=data.get('tier3_count', 20)
    )
    return jsonify(result)

@bp.route('/docking/binding-energy', methods=['POST'])
def calculate_binding_energy():
    """计算结合自由能"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    target = data.get('target', 'NP')
    
    result = enhanced_docking.calculate_binding_free_energy(smiles, target)
    return jsonify(result)

@bp.route('/docking/binding-mode', methods=['POST'])
def analyze_binding_mode():
    """分析结合模式"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    
    result = enhanced_docking.analyze_binding_mode(smiles)
    return jsonify(result)

@bp.route('/docking/evaluate-potential', methods=['POST'])
def evaluate_influenza_potential():
    """评估抑制剂潜力"""
    data = request.get_json() or {}
    energy = data.get('energy', 0.0)
    
    result = enhanced_docking.evaluate_influenza_potential(energy)
    return jsonify(result)

@bp.route('/docking/full-analysis', methods=['POST'])
def full_analysis():
    """完整分析 - 对接+稳定性+潜力评估"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    target = data.get('target', 'NP')
    
    # 分子对接
    docking_result = enhanced_docking.calculate_binding_free_energy(smiles, target)
    
    # 稳定性模拟
    stability_result = molecular_dynamics.simulate_binding_stability(smiles, simulation_steps=500)
    
    # 潜力评估
    if docking_result.get("success") and "mmgbsa_energy" in docking_result:
        potential_result = enhanced_docking.evaluate_influenza_potential(
            docking_result["mmgbsa_energy"]
        )
    else:
        potential_result = {"error": "无法评估潜力"}
    
    # 结合口袋分析
    pocket_result = molecular_dynamics.analyze_binding_pocket(smiles, target)
    
    return jsonify({
        "smiles": smiles,
        "target": target,
        "docking": docking_result,
        "stability": stability_result,
        "potential": potential_result,
        "pocket_analysis": pocket_result,
        "overall_recommendation": _get_overall_recommendation(docking_result, stability_result, potential_result)
    })

def _get_overall_recommendation(docking: dict, stability: dict, potential: dict) -> str:
    """综合推荐"""
    scores = []
    
    if docking.get("success"):
        if docking.get("binding_strength") in ["很强", "强"]:
            scores.append(3)
        elif docking.get("binding_strength") == "中等":
            scores.append(2)
        else:
            scores.append(1)
    
    if stability.get("stability_level") in ["很稳定", "稳定"]:
        scores.append(3)
    elif stability.get("stability_level") == "中等稳定":
        scores.append(2)
    else:
        scores.append(1)
    
    if potential.get("is_potential_inhibitor"):
        scores.append(3)
    elif potential.get("potential_level") == "中等潜力":
        scores.append(2)
    else:
        scores.append(1)
    
    avg_score = sum(scores) / len(scores) if scores else 0
    
    if avg_score >= 2.7:
        return "强烈推荐进行实验验证，该分子具有优秀的结合能力、稳定性和抑制潜力"
    elif avg_score >= 2.0:
        return "推荐进行实验验证，该分子具有良好的综合性能"
    elif avg_score >= 1.5:
        return "建议进行结构优化后实验验证"
    else:
        return "建议重新设计或选择其他候选分子"

@bp.route('/experiment/plan', methods=['POST'])
def create_experiment_plan():
    """生成实验计划"""
    data = request.get_json() or {}
    candidates = data.get('candidates', [])
    exp_type = data.get('experiment_type', 'in_vitro')
    
    result = experiment_tracker.create_experiment_plan(candidates, exp_type)
    return jsonify(result)

@bp.route('/experiment/record', methods=['POST'])
def record_experiment():
    """记录实验结果"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    exp_data = data.get('experiment_data', {})
    
    result = experiment_tracker.record_experiment_result(smiles, exp_data)
    return jsonify(result)

@bp.route('/experiment/analyze', methods=['POST'])
def analyze_experiment():
    """分析实验结果"""
    data = request.get_json() or {}
    results = data.get('results', [])
    
    result = experiment_tracker.analyze_experiment_results(results)
    return jsonify(result)

@bp.route('/feedback/generate', methods=['POST'])
def generate_feedback():
    """生成模型反馈"""
    data = request.get_json() or {}
    results = data.get('experiment_results', [])
    
    result = experiment_tracker.generate_feedback(results)
    return jsonify(result)

@bp.route('/feedback/update-model', methods=['POST'])
def update_model():
    """更新模型参数"""
    data = request.get_json() or {}
    feedback = data.get('feedback', {})
    
    result = experiment_tracker.update_model_with_feedback(feedback)
    return jsonify(result)

@bp.route('/report/generate', methods=['POST'])
def generate_report():
    """生成筛选报告"""
    data = request.get_json() or {}
    workflow_results = data.get('workflow_results', {})
    experiment_results = data.get('experiment_results', [])
    
    report = experiment_tracker.generate_final_report(workflow_results, experiment_results)
    return jsonify({"report": report})

@bp.route('/advanced/data-augmentation', methods=['POST'])
def data_augmentation():
    """数据增强 - 分子结构变体生成"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    num_augmented = data.get('num_augmented', 5)
    
    result = advanced_features.data_augmentation(smiles, num_augmented)
    return jsonify({
        "original_smiles": smiles,
        "augmented_count": len(result),
        "augmented_smiles": result
    })

@bp.route('/advanced/attention-weights', methods=['POST'])
def attention_weights():
    """注意力机制 - 计算分子相似性和重要性"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    target_smiles = data.get('target_smiles', '')
    
    result = advanced_features.attention_weights(smiles, target_smiles)
    return jsonify(result)

@bp.route('/advanced/active-learning', methods=['POST'])
def active_learning_weights():
    """主动学习权重 - 基于已知活性分子"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    known_actives = data.get('known_actives', [])
    
    result = advanced_features.active_learning_weights(smiles, known_actives)
    return jsonify(result)

@bp.route('/advanced/generate-training-set', methods=['POST'])
def generate_training_set():
    """生成训练集 - 包含增强数据"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    labels = data.get('labels', [])
    
    result = advanced_features.generate_training_set(smiles_list, labels)
    return jsonify(result)

@bp.route('/dynamics/stability', methods=['POST'])
def simulate_stability():
    """模拟结合稳定性"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    simulation_steps = data.get('simulation_steps', 1000)
    
    result = molecular_dynamics.simulate_binding_stability(smiles, simulation_steps)
    return jsonify(result)

@bp.route('/dynamics/batch-stability', methods=['POST'])
def batch_stability_simulation():
    """批量稳定性模拟"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    
    result = molecular_dynamics.batch_stability_simulation(smiles_list)
    return jsonify(result)

@bp.route('/dynamics/binding-pocket', methods=['POST'])
def analyze_binding_pocket():
    """分析结合口袋特征"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    target_pocket = data.get('target_pocket', 'NP')
    
    result = molecular_dynamics.analyze_binding_pocket(smiles, target_pocket)
    return jsonify(result)

@bp.route('/pdb/upload', methods=['POST'])
def upload_pdb():
    """上传PDB文件"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    if not file.filename.lower().endswith('.pdb'):
        return jsonify({"success": False, "error": "只支持PDB格式文件"}), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "size_kb": round(os.path.getsize(filepath) / 1024, 2)
    })

@bp.route('/pdb/analyze', methods=['POST'])
def analyze_pdb():
    """分析PDB文件"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    method = data.get('method', 'ligand')
    ligand_residue = data.get('ligand_residue')
    distance_cutoff = data.get('distance_cutoff', 5.0)
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件路径"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    result = pdb_analyzer.analyze_binding_sites(
        pdb_file,
        method=method,
        ligand_residue=ligand_residue,
        distance_cutoff=distance_cutoff
    )
    
    return jsonify(result)

@bp.route('/pdb/binding-sites', methods=['POST'])
def get_binding_sites():
    """获取结合位点摘要"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件路径"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    result = pdb_analyzer.get_pocket_summary(pdb_file)
    return jsonify(result)

@bp.route('/pdb/analyze-uploaded', methods=['POST'])
def analyze_uploaded_pdb():
    """上传并分析PDB文件（一步完成）"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    if not file.filename.lower().endswith('.pdb'):
        return jsonify({"success": False, "error": "只支持PDB格式文件"}), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    # 分析PDB文件
    method = request.form.get('method', 'ligand')
    result = pdb_analyzer.analyze_binding_sites(filepath, method=method)
    result['filename'] = filename
    
    return jsonify(result)
