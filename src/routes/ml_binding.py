from flask import Blueprint, request, jsonify, Response
from werkzeug.utils import secure_filename
import os
import json
import queue
import threading
import numpy as np
from flask import current_app
from src.services.ml_binding_predictor import ml_binding_predictor


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _sse(data_obj):
    return f"data: {json.dumps(data_obj, ensure_ascii=False, cls=_NumpyEncoder)}\n\n"

bp = Blueprint('ml_binding', __name__, url_prefix='/api/ml-binding')

@bp.route('/predict', methods=['POST'])
def predict_binding_sites():
    """预测结合位点"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    algorithm = data.get('algorithm', 'random_forest')
    threshold = data.get('threshold', 0.5)
    
    if not pdb_file:
        return jsonify({
            "success": False,
            "error": "未提供PDB文件路径"
        }), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({
            "success": False,
            "error": f"PDB文件不存在: {pdb_file}"
        }), 404
    
    result = ml_binding_predictor.predict_binding_sites(pdb_file, algorithm, threshold)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400

@bp.route('/batch-predict', methods=['POST'])
def batch_predict():
    """批量预测结合位点"""
    data = request.get_json() or {}
    pdb_files = data.get('pdb_files', [])
    algorithm = data.get('algorithm', 'random_forest')
    threshold = data.get('threshold', 0.5)
    
    if not pdb_files:
        return jsonify({
            "success": False,
            "error": "未提供PDB文件列表"
        }), 400
    
    result = ml_binding_predictor.batch_predict(pdb_files, algorithm, threshold)
    return jsonify(result)

@bp.route('/upload-pdb', methods=['POST'])
def upload_pdb():
    """上传PDB文件用于ML分析"""
    if 'file' not in request.files:
        return jsonify({
            "success": False,
            "error": "未上传文件"
        }), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({
            "success": False,
            "error": "未选择文件"
        }), 400
    
    if not file.filename.lower().endswith('.pdb'):
        return jsonify({
            "success": False,
            "error": "只支持PDB格式文件"
        }), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'ml_binding')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "size_kb": round(os.path.getsize(filepath) / 1024, 2),
        "message": "PDB文件上传成功，可以开始ML预测"
    })

@bp.route('/train-model', methods=['POST'])
def train_model():
    """训练模型"""
    data = request.get_json() or {}
    algorithm = data.get('algorithm', 'random_forest')
    data_source = data.get('data_source', 'synthetic')
    biolip_path = data.get('biolip_path', 'BioLiP.txt.gz')
    max_entries = data.get('max_entries', 5000)
    
    result = ml_binding_predictor.train_model(
        algorithm=algorithm,
        data_source=data_source,
        biolip_path=biolip_path,
        max_entries=max_entries
    )
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400

@bp.route('/train-model-stream', methods=['POST'])
def train_model_stream():
    """SSE 流式训练模型，实时推送训练进度"""
    data = request.get_json() or {}
    algorithm = data.get('algorithm', 'random_forest')
    data_source = data.get('data_source', 'synthetic')
    biolip_path = data.get('biolip_path', 'BioLiP.txt.gz')
    max_entries = data.get('max_entries', 5000)

    q = queue.Queue()

    def progress_callback(stage, progress, message):
        q.put({
            'type': 'progress',
            'stage': stage,
            'progress': progress,
            'message': message
        })

    def run_training():
        try:
            if data_source == 'real':
                from src.services.ml_algorithms import train_model_on_real_data
                result = train_model_on_real_data(
                    algorithm=algorithm,
                    biolip_path=biolip_path,
                    max_entries=max_entries,
                    use_sequence_only=True,
                    progress_callback=progress_callback
                )
            else:
                progress_callback('init', 0, '生成合成数据...')
                progress_callback('train', 50, '正在训练模型...')
                result = ml_binding_predictor.train_model(
                    algorithm=algorithm,
                    data_source='synthetic'
                )
                progress_callback('done', 100, '训练完成!')
            q.put({'type': 'result', **result})
        except Exception as e:
            q.put({'type': 'result', 'success': False, 'error': str(e)})

    train_thread = threading.Thread(target=run_training)
    train_thread.daemon = True
    train_thread.start()

    def generate():
        while True:
            try:
                item = q.get(timeout=0.5)
                yield _sse(item)
                if item.get('type') == 'result':
                    break
            except queue.Empty:
                if not train_thread.is_alive():
                    break

        train_thread.join(timeout=5)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})

@bp.route('/models', methods=['GET'])
def list_models():
    """列出所有模型"""
    result = ml_binding_predictor.list_models()
    return jsonify(result)

@bp.route('/models/<algorithm>', methods=['GET'])
def get_model_info(algorithm):
    """获取模型信息"""
    result = ml_binding_predictor.get_model_info(algorithm)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 404

@bp.route('/models/<algorithm>', methods=['DELETE'])
def delete_model(algorithm):
    """删除模型"""
    result = ml_binding_predictor.delete_model(algorithm)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400

@bp.route('/algorithms', methods=['GET'])
def get_algorithms():
    """获取支持的算法列表"""
    result = ml_binding_predictor.get_supported_algorithms()
    return jsonify(result)

@bp.route('/feature-info', methods=['POST'])
def get_feature_info():
    """获取特征信息"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    
    if not pdb_file:
        return jsonify({
            "success": False,
            "error": "未提供PDB文件路径"
        }), 400
    
    result = ml_binding_predictor.get_feature_info(pdb_file)
    
    if result.get('success'):
        return jsonify(result)
    else:
        return jsonify(result), 400
