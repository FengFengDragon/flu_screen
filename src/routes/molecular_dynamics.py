from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import os
from flask import current_app
from src.services.gromacs_runner import gromacs_runner
from src.services.format_converter import format_converter

bp = Blueprint('molecular_dynamics', __name__, url_prefix='/api/md')

@bp.route('/check-installation', methods=['GET'])
def check_gromacs():
    """检查GROMACS安装状态"""
    result = gromacs_runner.check_gromacs_installation()
    return jsonify(result)

@bp.route('/verify-gpu', methods=['GET'])
def verify_gpu():
    """验证GPU是否正在使用"""
    result = gromacs_runner.verify_gpu_usage()
    return jsonify(result)

@bp.route('/prepare', methods=['POST'])
def prepare_system():
    """准备MD模拟系统"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    forcefield = data.get('forcefield', 'amber99sb-ildn')
    water_model = data.get('water_model', 'tip3p')
    box_type = data.get('box_type', 'dodecahedron')
    distance = data.get('distance', 1.0)
    work_dir = data.get('work_dir')
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    result = gromacs_runner.prepare_system(
        pdb_file,
        forcefield=forcefield,
        water_model=water_model,
        box_type=box_type,
        distance=distance,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/minimize-energy', methods=['POST'])
def minimize_energy():
    """能量最小化"""
    data = request.get_json() or {}
    gro_file = data.get('gro_file')
    top_file = data.get('top_file')
    nsteps = data.get('nsteps', 50000)
    emtol = data.get('emtol', 1000.0)
    work_dir = data.get('work_dir')
    
    if not gro_file or not top_file:
        return jsonify({"success": False, "error": "未提供GRO或TOP文件"}), 400
    
    result = gromacs_runner.energy_minimize(
        gro_file,
        top_file,
        nsteps=nsteps,
        emtol=emtol,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/equilibrate', methods=['POST'])
def equilibrate():
    """NVT/NPT平衡"""
    data = request.get_json() or {}
    gro_file = data.get('gro_file')
    top_file = data.get('top_file')
    temperature = data.get('temperature', 310.0)
    pressure = data.get('pressure', 1.0)
    nvt_steps = data.get('nvt_steps', 50000)
    npt_steps = data.get('npt_steps', 100000)
    work_dir = data.get('work_dir')
    
    if not gro_file or not top_file:
        return jsonify({"success": False, "error": "未提供GRO或TOP文件"}), 400
    
    result = gromacs_runner.equilibrate_system(
        gro_file,
        top_file,
        temperature=temperature,
        pressure=pressure,
        nvt_steps=nvt_steps,
        npt_steps=npt_steps,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/production', methods=['POST'])
def run_production():
    """生产运行"""
    data = request.get_json() or {}
    gro_file = data.get('gro_file')
    top_file = data.get('top_file')
    time_ns = data.get('time_ns', 100.0)
    temperature = data.get('temperature', 310.0)
    pressure = data.get('pressure', 1.0)
    dt = data.get('dt', 0.002)
    work_dir = data.get('work_dir')
    
    if not gro_file or not top_file:
        return jsonify({"success": False, "error": "未提供GRO或TOP文件"}), 400
    
    result = gromacs_runner.run_production_md(
        gro_file,
        top_file,
        time_ns=time_ns,
        temperature=temperature,
        pressure=pressure,
        dt=dt,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/analyze', methods=['POST'])
def analyze_trajectory():
    """分析轨迹"""
    data = request.get_json() or {}
    trajectory_file = data.get('trajectory_file')
    top_file = data.get('top_file')
    work_dir = data.get('work_dir')
    
    if not trajectory_file or not top_file:
        return jsonify({"success": False, "error": "未提供轨迹或拓扑文件"}), 400
    
    if not os.path.exists(trajectory_file):
        return jsonify({"success": False, "error": "轨迹文件不存在"}), 404
    
    result = gromacs_runner.analyze_trajectory(
        trajectory_file,
        top_file,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/full-workflow', methods=['POST'])
def run_full_workflow():
    """运行完整MD工作流"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file')
    time_ns = data.get('time_ns', 100.0)
    temperature = data.get('temperature', 310.0)
    pressure = data.get('pressure', 1.0)
    work_dir = data.get('work_dir')
    
    if not pdb_file:
        return jsonify({"success": False, "error": "未提供PDB文件"}), 400
    
    if not os.path.exists(pdb_file):
        return jsonify({"success": False, "error": "PDB文件不存在"}), 404
    
    result = gromacs_runner.run_full_workflow(
        pdb_file,
        time_ns=time_ns,
        temperature=temperature,
        pressure=pressure,
        work_dir=work_dir
    )
    
    return jsonify(result)

@bp.route('/upload-pdb', methods=['POST'])
def upload_pdb_for_md():
    """上传文件用于MD模拟 - 支持多种格式"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    supported_formats = format_converter.get_supported_formats()
    if not any(file.filename.lower().endswith(ext) for ext in supported_formats):
        return jsonify({
            "success": False, 
            "error": f"不支持的文件格式。支持的格式: {', '.join(supported_formats)}"
        }), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    # 获取绝对路径，避免相对路径问题
    abs_filepath = os.path.abspath(filepath)
    
    result = {
        "success": True,
        "filename": filename,
        "filepath": abs_filepath,
        "size_kb": round(os.path.getsize(filepath) / 1024, 2),
        "original_format": os.path.splitext(filename)[1].lower(),
        "message": "文件上传成功"
    }
    
    if not filepath.lower().endswith('.gro'):
        converted = format_converter.convert_to_gro(filepath)
        if converted[0]:
            gro_file = os.path.splitext(filepath)[0] + '.gro'
            result["gro_file"] = gro_file
            result["converted"] = True
            result["message"] = f"文件上传成功，已转换为GRO格式"
            result["conversion_message"] = converted[1]
        else:
            result["converted"] = False
            result["conversion_error"] = converted[1]
    
    return jsonify(result)

@bp.route('/fasta-info', methods=['POST'])
def get_fasta_info():
    """获取FASTA序列信息"""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "未选择文件"}), 400
    
    if not file.filename.lower().endswith('.fasta'):
        return jsonify({"success": False, "error": "仅支持FASTA格式文件"}), 400
    
    filename = secure_filename(file.filename)
    upload_dir = os.path.join(current_app.root_path, '..', '..', 'data', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    sequence_info = format_converter.get_fasta_info(filepath)
    
    return jsonify({
        "success": True,
        "filename": filename,
        "filepath": filepath,
        "sequence_info": sequence_info,
        "prediction_methods": format_converter.get_prediction_methods()
    })

@bp.route('/jobs', methods=['GET'])
def list_jobs():
    """列出所有模拟任务"""
    return jsonify({
        "jobs": gromacs_runner.simulation_jobs,
        "total": len(gromacs_runner.simulation_jobs)
    })

@bp.route('/jobs/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """获取特定任务状态"""
    job = gromacs_runner.simulation_jobs.get(job_id)
    if job:
        return jsonify(job)
    return jsonify({"error": "未找到该任务"}), 404

@bp.route('/cleanup', methods=['POST'])
def cleanup_temp_files():
    """清理临时文件"""
    data = request.get_json() or {}
    work_dir = data.get('work_dir')
    base_name = data.get('base_name')
    keep_files = data.get('keep_files')
    
    if not work_dir or not base_name:
        return jsonify({"success": False, "error": "未提供工作目录或基础文件名"}), 400
    
    if keep_files and not isinstance(keep_files, list):
        return jsonify({"success": False, "error": "keep_files参数必须是列表"}), 400
    
    result = gromacs_runner.cleanup_temp_files(
        work_dir,
        base_name,
        keep_files=keep_files
    )
    
    return jsonify(result)

@bp.route('/cleanup/workflow', methods=['POST'])
def cleanup_workflow():
    """清理整个工作流的临时文件"""
    data = request.get_json() or {}
    work_dir = data.get('work_dir')
    base_name = data.get('base_name')
    cleanup_level = data.get('cleanup_level', 'moderate')
    
    if not work_dir or not base_name:
        return jsonify({"success": False, "error": "未提供工作目录或基础文件名"}), 400
    
    if cleanup_level not in ['minimal', 'moderate', 'aggressive']:
        return jsonify({"success": False, "error": "清理级别必须是minimal、moderate或aggressive"}), 400
    
    result = gromacs_runner.cleanup_workflow(
        work_dir,
        base_name,
        cleanup_level=cleanup_level
    )
    
    return jsonify(result)

@bp.route('/cleanup/all', methods=['POST'])
def cleanup_all_temp_files():
    """清理所有临时文件（在data/gromacs目录下）"""
    data = request.get_json() or {}
    work_dir = data.get('work_dir')
    cleanup_level = data.get('cleanup_level', 'moderate')
    
    if not work_dir:
        work_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', '..', 'data', 'gromacs')
    
    if cleanup_level not in ['minimal', 'moderate', 'aggressive']:
        return jsonify({"success": False, "error": "清理级别必须是minimal、moderate或aggressive"}), 400
    
    if not os.path.exists(work_dir):
        return jsonify({"success": False, "error": "工作目录不存在"}), 404
    
    try:
        import glob
        total_deleted = 0
        total_freed = 0
        cleaned_projects = []
        
        # 扫描所有项目的临时文件
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            
            # 只处理文件和子目录
            if os.path.isdir(item_path):
                # 清理子目录中的临时文件
                sub_result = gromacs_runner.cleanup_workflow(
                    item_path,
                    os.path.splitext(item)[0],
                    cleanup_level=cleanup_level
                )
                if sub_result.get("success"):
                    total_deleted += sub_result.get("deleted_count", 0)
                    total_freed += sub_result.get("freed_space_bytes", 0)
                    cleaned_projects.append(item)
        
        return jsonify({
            "success": True,
            "cleanup_level": cleanup_level,
            "total_deleted": total_deleted,
            "total_freed_mb": round(total_freed / (1024 * 1024), 2),
            "cleaned_projects": cleaned_projects,
            "work_dir": work_dir
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "error"
        })

@bp.route('/list-files', methods=['POST'])
def list_project_files():
    """列出项目目录中的所有文件"""
    data = request.get_json() or {}
    work_dir = data.get('work_dir')
    
    if not work_dir:
        return jsonify({"success": False, "error": "未提供工作目录"}), 400
    
    if not os.path.exists(work_dir):
        return jsonify({"success": False, "error": "工作目录不存在"}), 404
    
    try:
        import os
        
        files = []
        total_size = 0
        
        for item in os.listdir(work_dir):
            item_path = os.path.join(work_dir, item)
            if os.path.isfile(item_path):
                size = os.path.getsize(item_path)
                total_size += size
                files.append({
                    "name": item,
                    "size_bytes": size,
                    "size_mb": round(size / (1024 * 1024), 2),
                    "extension": os.path.splitext(item)[1]
                })
        
        # 按文件大小排序
        files.sort(key=lambda x: x["size_bytes"], reverse=True)
        
        return jsonify({
            "success": True,
            "work_dir": work_dir,
            "total_files": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "files": files
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "error"
        })
