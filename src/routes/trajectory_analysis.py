from flask import Blueprint, request, jsonify, send_file
from werkzeug.utils import secure_filename
import os
import numpy as np
import io
import tempfile
import zipfile
import shutil
from flask import current_app
from src.services.trajectory_analyzer import trajectory_analyzer

bp = Blueprint('trajectory_analysis', __name__, url_prefix='/api/trajectory')

@bp.route('/load', methods=['POST'])
def load_trajectory():
    """加载轨迹文件"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    if not os.path.exists(topology_file):
        return jsonify({"success": False, "error": "拓扑文件不存在"}), 404
    
    if not os.path.exists(trajectory_file):
        return jsonify({"success": False, "error": "轨迹文件不存在"}), 404
    
    result = trajectory_analyzer.load_trajectory(topology_file, trajectory_file)
    return jsonify(result)

@bp.route('/extract-key-frames', methods=['POST'])
def extract_key_frames():
    """智能抽帧"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    num_frames = data.get('num_frames', 100)
    method = data.get('method', 'rmsd')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.extract_key_frames(
        topology_file,
        trajectory_file,
        num_frames=num_frames,
        method=method
    )
    return jsonify(result)

@bp.route('/pca', methods=['POST'])
def pca_analysis():
    """PCA降维分析"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    n_components = data.get('n_components', 2)
    atom_selection = data.get('atom_selection', 'protein')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.pca_analysis(
        topology_file,
        trajectory_file,
        n_components=n_components,
        atom_selection=atom_selection
    )
    return jsonify(result)

@bp.route('/tsne', methods=['POST'])
def tsne_analysis():
    """t-SNE非线性降维分析"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    n_components = data.get('n_components', 2)
    perplexity = data.get('perplexity', 30.0)
    n_iter = data.get('n_iter', 1000)
    atom_selection = data.get('atom_selection', 'protein')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.tsne_analysis(
        topology_file,
        trajectory_file,
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        atom_selection=atom_selection
    )
    return jsonify(result)

@bp.route('/cluster', methods=['POST'])
def cluster_conformations():
    """构象聚类分析"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    n_clusters = data.get('n_clusters', 5)
    method = data.get('method', 'kmeans')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.cluster_conformations(
        topology_file,
        trajectory_file,
        n_clusters=n_clusters,
        method=method
    )
    return jsonify(result)

@bp.route('/rmsd-rmsf', methods=['POST'])
def analyze_rmsd_rmsf():
    """分析RMSD和RMSF"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.analyze_rmsd_rmsf(topology_file, trajectory_file)
    return jsonify(result)

@bp.route('/full-analysis', methods=['POST'])
def full_analysis():
    """完整轨迹分析"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.full_trajectory_analysis(topology_file, trajectory_file)
    return jsonify(result)

@bp.route('/upload', methods=['POST'])
def upload_trajectory():
    """上传轨迹文件"""
    if 'topology' not in request.files or 'trajectory' not in request.files:
        return jsonify({"success": False, "error": "未上传拓扑或轨迹文件"}), 400
    
    topology_file = request.files['topology']
    trajectory_file = request.files['trajectory']
    
    if topology_file.filename == '' or trajectory_file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    topology_filename = secure_filename(topology_file.filename)
    trajectory_filename = secure_filename(trajectory_file.filename)
    
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'trajectories')
    os.makedirs(upload_dir, exist_ok=True)
    
    topology_path = os.path.join(upload_dir, topology_filename)
    trajectory_path = os.path.join(upload_dir, trajectory_filename)
    
    topology_file.save(topology_path)
    trajectory_file.save(trajectory_path)
    
    return jsonify({
        "success": True,
        "topology_file": topology_path,
        "trajectory_file": trajectory_path,
        "topology_filename": topology_filename,
        "trajectory_filename": trajectory_filename,
        "topology_size_kb": round(os.path.getsize(topology_path) / 1024, 2),
        "trajectory_size_mb": round(os.path.getsize(trajectory_path) / 1024 / 1024, 2),
        "message": "文件上传成功，可以开始分析"
    })

@bp.route('/compare', methods=['POST'])
def compare_trajectories():
    """比较两个轨迹"""
    data = request.get_json() or {}
    topology_file1 = data.get('topology_file1')
    trajectory_file1 = data.get('trajectory_file1')
    topology_file2 = data.get('topology_file2')
    trajectory_file2 = data.get('trajectory_file2')
    
    if not all([topology_file1, trajectory_file1, topology_file2, trajectory_file2]):
        return jsonify({"success": False, "error": "未提供完整的轨迹文件对"}), 400
    
    try:
        result1 = trajectory_analyzer.load_trajectory(topology_file1, trajectory_file1)
        result2 = trajectory_analyzer.load_trajectory(topology_file2, trajectory_file2)
        
        if not result1["success"] or not result2["success"]:
            return jsonify({
                "success": False,
                "error": "轨迹加载失败",
                "details": {
                    "trajectory1": result1,
                    "trajectory2": result2
                }
            })
        
        # PCA比较
        pca1 = trajectory_analyzer.pca_analysis(topology_file1, trajectory_file1, n_components=2)
        pca2 = trajectory_analyzer.pca_analysis(topology_file2, trajectory_file2, n_components=2)
        
        return jsonify({
            "success": True,
            "comparison": {
                "trajectory1": {
                    "n_frames": result1["n_frames"],
                    "n_atoms": result1["n_atoms"],
                    "pc1_variance": pca1["explained_variance"][0] if pca1["success"] else None
                },
                "trajectory2": {
                    "n_frames": result2["n_frames"],
                    "n_atoms": result2["n_atoms"],
                    "pc1_variance": pca2["explained_variance"][0] if pca2["success"] else None
                },
                "difference": {
                    "frames_diff": result1["n_frames"] - result2["n_frames"],
                    "atoms_diff": result1["n_atoms"] - result2["n_atoms"]
                }
            }
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "comparison_failed"
        })

@bp.route('/download-key-frames', methods=['POST'])
def download_key_frames():
    """下载关键帧PDB文件"""
    data = request.get_json() or {}
    topology_file = data.get('topology_file')
    trajectory_file = data.get('trajectory_file')
    num_frames = data.get('num_frames', 100)
    
    if not topology_file or not trajectory_file:
        return jsonify({"success": False, "error": "未提供拓扑文件或轨迹文件"}), 400
    
    result = trajectory_analyzer.extract_key_frames(
        topology_file,
        trajectory_file,
        num_frames=num_frames,
        method="rmsd"
    )
    
    if not result["success"]:
        return jsonify(result)
    
    # 创建临时目录保存PDB文件
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    output_files = []
    base_name = os.path.splitext(os.path.basename(trajectory_file))[0]
    
    for i, frame_info in enumerate(result["key_frames"]):
        output_filename = f"{base_name}_keyframe_{i+1}.pdb"
        output_path = os.path.join(temp_dir, output_filename)
        
        # 写入PDB文件（简化版本）
        coords = np.array(frame_info["coordinates"])
        with open(output_path, 'w') as f:
            for j, coord in enumerate(coords):
                f.write(f"ATOM  {j+1:5d}  CA  ALA A{j+1:4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}  1.00  0.00\n")
            f.write("END\n")
        
        output_files.append({
            "filename": output_filename,
            "path": output_path,
            "frame_index": frame_info["frame_index"],
            "time": frame_info["time"]
        })
    
    # 创建ZIP文件
    import zipfile
    zip_filename = f"{base_name}_keyframes.zip"
    zip_path = os.path.join(temp_dir, zip_filename)
    
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for output_file in output_files:
            zipf.write(output_file['path'], output_file['filename'])
    
    # 读取ZIP文件内容
    with open(zip_path, 'rb') as f:
        zip_data = f.read()
    
    # 清理临时目录
    import shutil
    shutil.rmtree(temp_dir)
    
    # 返回ZIP文件下载
    from flask import send_file
    return send_file(
        io.BytesIO(zip_data),
        mimetype='application/zip',
        as_attachment=True,
        download_name=zip_filename
    )
