from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask import current_app
from src.services.deep_learning_models import deep_learning_models

bp = Blueprint('deep_learning', __name__, url_prefix='/api/deep-learning')

@bp.route('/create-graph', methods=['POST'])
def create_protein_graph():
    """从PDB文件创建蛋白质图"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    threshold = data.get('threshold', 8.0)
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    graph_data = deep_learning_models.create_protein_graph(pdb_file, threshold)
    
    if graph_data is None:
        return jsonify({"success": False, "error": "无法创建蛋白质图"}), 400
    
    return jsonify({
        "success": True,
        "pdb_file": pdb_file,
        "threshold": threshold,
        "num_nodes": graph_data.x.shape[0],
        "num_edges": graph_data.edge_index.shape[1] // 2,
        "node_features": graph_data.x.shape[1],
        "message": "蛋白质图创建成功"
    })

@bp.route('/predict-key-residues', methods=['POST'])
def predict_key_residues():
    """预测关键残基"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    result = deep_learning_models.predict_key_residues(pdb_file)
    return jsonify(result)

@bp.route('/upload-pdb', methods=['POST'])
def upload_pdb_for_dl():
    """上传PDB文件用于深度学习分析"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    if not file.filename.lower().endswith('.pdb'):
        return jsonify({"success": False, "error": "只支持PDB格式文件"}), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'deep_learning')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "size_kb": round(os.path.getsize(filepath) / 1024, 2),
        "message": "PDB文件上传成功，可以开始深度学习分析"
    })

@bp.route('/models', methods=['GET'])
def list_models():
    """列出所有训练好的模型"""
    model_dir = deep_learning_models.model_dir
    
    if not os.path.exists(model_dir):
        return jsonify({
            "success": True,
            "models": [],
            "message": "模型目录不存在"
        })
    
    models = []
    for filename in os.listdir(model_dir):
        if filename.endswith('.pth'):
            model_path = os.path.join(model_dir, filename)
            models.append({
                "name": filename,
                "path": model_path,
                "size_mb": round(os.path.getsize(model_path) / 1024 / 1024, 2)
            })
    
    return jsonify({
        "success": True,
        "models": models,
        "total": len(models)
    })

@bp.route('/models/<model_name>', methods=['DELETE'])
def delete_model(model_name):
    """删除模型"""
    model_path = os.path.join(deep_learning_models.model_dir, model_name)
    
    if not model_path.endswith('.pth'):
        model_path += '.pth'
    
    if not os.path.exists(model_path):
        return jsonify({"error": "模型不存在"}), 404
    
    try:
        os.remove(model_path)
        return jsonify({
            "success": True,
            "message": f"模型 {model_name} 已删除"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """批量预测关键残基"""
    data = request.get_json() or {}
    pdb_files = data.get('pdb_files', [])
    
    if not pdb_files:
        return jsonify({"success": False, "error": "未提供PDB文件列表"}), 400
    
    results = []
    for pdb_file in pdb_files:
        if os.path.exists(pdb_file):
            result = deep_learning_models.predict_key_residues(pdb_file)
            result["pdb_file"] = pdb_file
            results.append(result)
    
    return jsonify({
        "success": True,
        "total_files": len(pdb_files),
        "successful_predictions": sum(1 for r in results if r.get("success")),
        "results": results
    })

@bp.route('/model-info', methods=['GET'])
def get_model_info():
    """获取模型信息"""
    return jsonify({
        "success": True,
        "model_types": {
            "gcn": {
                "name": "Graph Convolutional Network",
                "description": "图卷积网络，用于识别蛋白质关键残基（推理模式）",
                "input": "PDB文件",
                "output": "关键残基列表",
                "mode": "inference_only",
                "supported_formats": [".pth", ".onnx"]
            }
        },
        "available_models": {
            "gcn": True,
            "onnx": True
        },
        "model_directory": deep_learning_models.model_dir,
        "note": "仅支持推理功能，训练请使用外部程序"
    })

@bp.route('/convert-onnx', methods=['POST'])
def convert_model_to_onnx():
    """将PyTorch模型转换为ONNX格式"""
    result = deep_learning_models.convert_to_onnx()
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify(result), 400

@bp.route('/predict-onnx', methods=['POST'])
def predict_with_onnx():
    """使用ONNX模型进行快速推理"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    
    if not pdb_file:
        return jsonify({
            "success": False,
            "error": "未提供PDB文件"
        }), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({
            "success": False,
            "error": "PDB文件不存在"
        }), 404
    
    result = deep_learning_models.predict_key_residues_onnx(pdb_file)
    
    if result["success"]:
        return jsonify(result)
    else:
        return jsonify(result), 400
