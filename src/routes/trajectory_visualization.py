from flask import Blueprint, request, jsonify, current_app, send_file
import os
import json
from werkzeug.utils import secure_filename
from src.services.trajectory_visualizer import trajectory_visualizer

bp = Blueprint('trajectory_visualization', __name__, url_prefix='/api/trajectory-visualization')


@bp.route('/status', methods=['GET'])
def get_status():
    """获取可视化系统状态"""
    return jsonify({
        "status": "active",
        "matplotlib_available": trajectory_visualizer.MATPLOTLIB_AVAILABLE,
        "mpl_3d_available": trajectory_visualizer.MPL_3D_AVAILABLE,
        "output_dir": trajectory_visualizer.output_dir,
        "message": "轨迹可视化系统正常运行"
    })


@bp.route('/create-3d-plot', methods=['POST'])
def create_3d_plot():
    """创建3D轨迹图"""
    try:
        data = request.get_json()
        
        trajectory_data = data.get('trajectory_data', {})
        key_residues = data.get('key_residues', [])
        frame_indices = data.get('frame_indices', [])
        show_backbone = data.get('show_backbone', True)
        
        if not trajectory_data:
            return jsonify({
                "success": False,
                "error": "缺少轨迹数据"
            }), 400
        
        result = trajectory_visualizer.create_3d_trajectory_plot(
            trajectory_data=trajectory_data,
            key_residues=key_residues,
            frame_indices=frame_indices,
            show_backbone=show_backbone
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "visualization_failed"
        }), 500


@bp.route('/create-animation', methods=['POST'])
def create_animation():
    """创建轨迹动画"""
    try:
        data = request.get_json()
        
        trajectory_data = data.get('trajectory_data', {})
        output_format = data.get('output_format', 'gif')
        frame_interval = data.get('frame_interval', 100)
        key_residues = data.get('key_residues', [])
        
        if not trajectory_data:
            return jsonify({
                "success": False,
                "error": "缺少轨迹数据"
            }), 400
        
        if output_format not in ['gif', 'mp4', 'avi']:
            return jsonify({
                "success": False,
                "error": "不支持的输出格式，请使用gif、mp4或avi"
            }), 400
        
        result = trajectory_visualizer.create_trajectory_animation(
            trajectory_data=trajectory_data,
            output_format=output_format,
            frame_interval=frame_interval,
            key_residues=key_residues
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "animation_failed"
        }), 500


@bp.route('/create-rmsd-plot', methods=['POST'])
def create_rmsd_plot():
    """创建RMSD图表"""
    try:
        data = request.get_json()
        
        rmsd_data = data.get('rmsd_data', {})
        
        if not rmsd_data:
            return jsonify({
                "success": False,
                "error": "缺少RMSD数据"
            }), 400
        
        result = trajectory_visualizer.create_rmsd_plot(rmsd_data=rmsd_data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "plot_failed"
        }), 500


@bp.route('/create-rmsf-plot', methods=['POST'])
def create_rmsf_plot():
    """创建RMSF图表"""
    try:
        data = request.get_json()
        
        rmsf_data = data.get('rmsf_data', {})
        
        if not rmsf_data:
            return jsonify({
                "success": False,
                "error": "缺少RMSF数据"
            }), 400
        
        result = trajectory_visualizer.create_rmsf_plot(rmsf_data=rmsf_data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "plot_failed"
        }), 500


@bp.route('/create-conformation-heatmap', methods=['POST'])
def create_conformation_heatmap():
    """创建构象分布热图"""
    try:
        data = request.get_json()
        
        pca_data = data.get('pca_data', {})
        tsne_data = data.get('tsne_data', {})
        
        if not pca_data and not tsne_data:
            return jsonify({
                "success": False,
                "error": "缺少PCA或t-SNE数据"
            }), 400
        
        result = trajectory_visualizer.create_conformation_heatmap(
            pca_data=pca_data,
            tsne_data=tsne_data
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "heatmap_failed"
        }), 500


@bp.route('/create-interaction-network', methods=['POST'])
def create_interaction_network():
    """创建相互作用网络图"""
    try:
        data = request.get_json()
        
        interaction_data = data.get('interaction_data', {})
        
        if not interaction_data:
            return jsonify({
                "success": False,
                "error": "缺少相互作用数据"
            }), 400
        
        result = trajectory_visualizer.create_interaction_network(
            interaction_data=interaction_data
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "network_failed"
        }), 500


@bp.route('/create-dashboard', methods=['POST'])
def create_dashboard():
    """创建综合仪表板"""
    try:
        data = request.get_json()
        
        trajectory_data = data.get('trajectory_data', {})
        analysis_data = data.get('analysis_data', {})
        output_format = data.get('output_format', 'html')
        
        if not trajectory_data or not analysis_data:
            return jsonify({
                "success": False,
                "error": "缺少轨迹数据或分析数据"
            }), 400
        
        result = trajectory_visualizer.create_comprehensive_dashboard(
            trajectory_data=trajectory_data,
            analysis_data=analysis_data,
            output_format=output_format
        )
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "dashboard_failed"
        }), 500


@bp.route('/visualize/<viz_type>', methods=['POST'])
def visualize(viz_type):
    """通用可视化接口"""
    try:
        data = request.get_json()
        
        viz_functions = {
            '3d': trajectory_visualizer.create_3d_trajectory_plot,
            'animation': trajectory_visualizer.create_trajectory_animation,
            'rmsd': trajectory_visualizer.create_rmsd_plot,
            'rmsf': trajectory_visualizer.create_rmsf_plot,
            'heatmap': trajectory_visualizer.create_conformation_heatmap,
            'network': trajectory_visualizer.create_interaction_network,
            'dashboard': trajectory_visualizer.create_comprehensive_dashboard
        }
        
        if viz_type not in viz_functions:
            return jsonify({
                "success": False,
                "error": f"不支持的可视化类型: {viz_type}",
                "supported_types": list(viz_functions.keys())
            }), 400
        
        result = viz_functions[viz_type](**data)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "status": "visualization_failed"
        }), 500


@bp.route('/download/<filename>', methods=['GET'])
def download_visualization(filename):
    """下载可视化结果文件"""
    try:
        filename = secure_filename(filename)
        file_path = os.path.join(trajectory_visualizer.output_dir, filename)
        
        if not os.path.exists(file_path):
            return jsonify({
                "success": False,
                "error": "文件不存在"
            }), 404
        
        return send_file(file_path, as_attachment=True)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route('/list-visualizations', methods=['GET'])
def list_visualizations():
    """列出所有可视化文件"""
    try:
        files = []
        
        if os.path.exists(trajectory_visualizer.output_dir):
            for filename in os.listdir(trajectory_visualizer.output_dir):
                file_path = os.path.join(trajectory_visualizer.output_dir, filename)
                if os.path.isfile(file_path):
                    files.append({
                        "filename": filename,
                        "size": os.path.getsize(file_path),
                        "modified": os.path.getmtime(file_path)
                    })
        
        files.sort(key=lambda x: x['modified'], reverse=True)
        
        return jsonify({
            "success": True,
            "output_dir": trajectory_visualizer.output_dir,
            "files": files,
            "total_count": len(files)
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route('/clear-visualizations', methods=['DELETE'])
def clear_visualizations():
    """清除所有可视化文件"""
    try:
        count = 0
        
        if os.path.exists(trajectory_visualizer.output_dir):
            for filename in os.listdir(trajectory_visualizer.output_dir):
                file_path = os.path.join(trajectory_visualizer.output_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    count += 1
        
        return jsonify({
            "success": True,
            "deleted_count": count,
            "message": f"已清除 {count} 个可视化文件"
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@bp.route('/batch-visualize', methods=['POST'])
def batch_visualize():
    """批量可视化"""
    try:
        data = request.get_json()
        
        visualization_tasks = data.get('tasks', [])
        results = []
        
        for task in visualization_tasks:
            viz_type = task.get('type')
            viz_data = task.get('data', {})
            
            viz_functions = {
                '3d': trajectory_visualizer.create_3d_trajectory_plot,
                'animation': trajectory_visualizer.create_trajectory_animation,
                'rmsd': trajectory_visualizer.create_rmsd_plot,
                'rmsf': trajectory_visualizer.create_rmsf_plot,
                'heatmap': trajectory_visualizer.create_conformation_heatmap,
                'network': trajectory_visualizer.create_interaction_network
            }
            
            if viz_type in viz_functions:
                result = viz_functions[viz_type](**viz_data)
                results.append({
                    "type": viz_type,
                    "result": result
                })
            else:
                results.append({
                    "type": viz_type,
                    "result": {
                        "success": False,
                        "error": f"不支持的可视化类型: {viz_type}"
                    }
                })
        
        success_count = sum(1 for r in results if r['result'].get('success', False))
        
        return jsonify({
            "success": True,
            "total_tasks": len(visualization_tasks),
            "successful_tasks": success_count,
            "results": results
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500
