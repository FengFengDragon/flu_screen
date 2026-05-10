import os
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import base64

try:
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle, FancyArrowPatch
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib未安装，将使用基本可视化")

try:
    from mpl_toolkits.mplot3d import Axes3D
    MPL_3D_AVAILABLE = True
except ImportError:
    MPL_3D_AVAILABLE = False
    print("Matplotlib 3D工具未安装")


class TrajectoryVisualizer:
    """轨迹可视化增强服务
    
    支持多种可视化方式：
    - 3D轨迹动画
    - 轨迹统计图表
    - 关键残基高亮
    - 相互作用网络
    - 构象分布热图
    """
    
    def __init__(self, output_dir: str = "data/visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.MATPLOTLIB_AVAILABLE = MATPLOTLIB_AVAILABLE
        self.MPL_3D_AVAILABLE = MPL_3D_AVAILABLE
        
        self.color_map = {
            'ALA': '#FF6B6B', 'ARG': '#4ECDC4', 'ASN': '#45B7D1',
            'ASP': '#96CEB4', 'CYS': '#FFEAA7', 'GLN': '#DDA0DD',
            'GLU': '#98D8C8', 'GLY': '#F7DC6F', 'HIS': '#FF9FF3',
            'ILE': '#B4A7D6', 'LEU': '#A8E6CF', 'LYS': '#FD79A8',
            'MET': '#FDCB6E', 'PHE': '#6C5CE7', 'PRO': '#FF85A1',
            'SER': '#FFA502', 'THR': '#A29BFE', 'TRP': '#FF6348',
            'TYR': '#009432', 'VAL': '#00D2D3', 'UNK': '#95A5A6'
        }
    
    def create_3d_trajectory_plot(self, trajectory_data: Dict, 
                               key_residues: List[int] = None,
                               frame_indices: List[int] = None,
                               show_backbone: bool = True) -> Dict:
        """创建3D轨迹图
        
        Args:
            trajectory_data: 轨迹数据
            key_residues: 关键残基列表
            frame_indices: 要显示的帧索引
            show_backbone: 是否显示主链
            
        Returns:
            可视化结果
        """
        if not MATPLOTLIB_AVAILABLE or not MPL_3D_AVAILABLE:
            return self._mock_3d_plot(trajectory_data)
        
        try:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            coordinates = trajectory_data.get('coordinates', [])
            residue_names = trajectory_data.get('residue_names', [])
            residue_ids = trajectory_data.get('residue_ids', [])
            
            if frame_indices:
                frame_coords = [coordinates[i] for i in frame_indices if i < len(coordinates)]
            else:
                frame_coords = coordinates
            
            n_frames = len(frame_coords)
            colors = ['blue', 'green', 'red', 'orange', 'purple']
            
            for frame_idx, frame in enumerate(frame_coords):
                color = colors[frame_idx % len(colors)]
                alpha = 0.8 if frame_idx == 0 else 0.3
                
                coords = np.array(frame)
                
                if show_backbone and len(coords) > 1:
                    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                           color=color, alpha=alpha, linewidth=1, 
                           label=f'Frame {frame_idx + 1}')
                    
                    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2],
                             color=color, alpha=alpha, s=20)
                
                if key_residues and frame_idx == 0:
                    for key_res in key_residues:
                        if key_res < len(coords):
                            ax.scatter([coords[key_res, 0]], [coords[key_res, 1]], [coords[key_res, 2]],
                                     color='red', s=100, marker='*', 
                                     label=f'Key Residue {key_res}')
            
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title('3D Trajectory Visualization')
            ax.legend()
            
            output_file = os.path.join(self.output_dir, '3d_trajectory.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "3d_trajectory",
                "output_file": output_file,
                "n_frames": n_frames,
                "key_residues": key_residues or [],
                "message": "3D轨迹图创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "visualization_failed"
            }
    
    def create_trajectory_animation(self, trajectory_data: Dict,
                               output_format: str = 'gif',
                               frame_interval: int = 100,
                               key_residues: List[int] = None) -> Dict:
        """创建轨迹动画
        
        Args:
            trajectory_data: 轨迹数据
            output_format: 输出格式
            frame_interval: 帧间隔
            key_residues: 关键残基列表
            
        Returns:
            动画创建结果
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_animation(trajectory_data)
        
        try:
            coordinates = trajectory_data.get('coordinates', [])
            residue_names = trajectory_data.get('residue_names', [])
            
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            coords = np.array(coordinates[0])
            
            scatter = ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
                            color='blue', s=50)
            line, = ax.plot(coords[:, 0], coords[:, 1], coords[:, 2],
                             color='blue', alpha=0.5, linewidth=1)
            
            key_scatters = []
            if key_residues:
                for key_res in key_residues:
                    if key_res < len(coords):
                        scat = ax.scatter([coords[key_res, 0]], [coords[key_res, 1]], [coords[key_res, 2]],
                                       color='red', s=100, marker='*')
                        key_scatters.append(scat)
            
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_title('Trajectory Animation')
            ax.set_xlim(np.min([c[:, 0] for c in coordinates]) - 2,
                         np.max([c[:, 0] for c in coordinates]) + 2)
            ax.set_ylim(np.min([c[:, 1] for c in coordinates]) - 2,
                         np.max([c[:, 1] for c in coordinates]) + 2)
            ax.set_zlim(np.min([c[:, 2] for c in coordinates]) - 2,
                         np.max([c[:, 2] for c in coordinates]) + 2)
            
            def update(frame):
                coords = np.array(coordinates[frame])
                
                scatter._offsets3d = (coords[:, 0], coords[:, 1], coords[:, 2])
                line.set_data(coords[:, 0], coords[:, 1])
                line.set_3d_properties(coords[:, 2])
                
                if key_residues:
                    for i, key_res in enumerate(key_residues):
                        if key_res < len(coords):
                            key_scatters[i]._offsets3d = ([coords[key_res, 0]], 
                                                           [coords[key_res, 1]], 
                                                           [coords[key_res, 2]])
                
                ax.set_title(f'Trajectory Animation - Frame {frame + 1}/{len(coordinates)}')
            
            anim = animation.FuncAnimation(fig, update, frames=len(coordinates),
                                      interval=frame_interval, blit=False)
            
            output_file = os.path.join(self.output_dir, f'trajectory_animation.{output_format}')
            
            if output_format == 'gif':
                anim.save(output_file, writer='pill', fps=10)
            else:
                anim.save(output_file, writer='ffmpeg', fps=10)
            
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "animation",
                "output_file": output_file,
                "format": output_format,
                "n_frames": len(coordinates),
                "frame_interval": frame_interval,
                "key_residues": key_residues or [],
                "message": f"轨迹动画创建成功 ({output_format}格式)"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "animation_failed"
            }
    
    def create_rmsd_plot(self, rmsd_data: Dict) -> Dict:
        """创建RMSD图表
        
        Args:
            rmsd_data: RMSD数据
            
        Returns:
            可视化结果
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_plot("RMSD Plot")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            time_points = rmsd_data.get('time_points', [])
            rmsd_values = rmsd_data.get('rmsd_values', [])
            rmsd_std = rmsd_data.get('rmsd_std', [])
            
            ax1.plot(time_points, rmsd_values, 'b-', linewidth=2, label='RMSD')
            ax1.fill_between(time_points, 
                            np.array(rmsd_values) - np.array(rmsd_std),
                            np.array(rmsd_values) + np.array(rmsd_std),
                            alpha=0.3, color='blue')
            ax1.set_xlabel('Time (ns)')
            ax1.set_ylabel('RMSD (nm)')
            ax1.set_title('RMSD vs Time')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            ax2.hist(rmsd_values, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('RMSD (nm)')
            ax2.set_ylabel('Frequency')
            ax2.set_title('RMSD Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, 'rmsd_plot.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "rmsd_plot",
                "output_file": output_file,
                "statistics": {
                    "mean_rmsd": float(np.mean(rmsd_values)) if rmsd_values else 0,
                    "std_rmsd": float(np.std(rmsd_values)) if rmsd_values else 0,
                    "max_rmsd": float(np.max(rmsd_values)) if rmsd_values else 0,
                    "min_rmsd": float(np.min(rmsd_values)) if rmsd_values else 0
                },
                "message": "RMSD图创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "plot_failed"
            }
    
    def create_rmsf_plot(self, rmsf_data: Dict) -> Dict:
        """创建RMSF图表
        
        Args:
            rmsf_data: RMSF数据
            
        Returns:
            可视化结果
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_plot("RMSF Plot")
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            residue_ids = rmsf_data.get('residue_ids', [])
            residue_names = rmsf_data.get('residue_names', [])
            rmsf_values = rmsf_data.get('rmsf_values', [])
            key_residues = rmsf_data.get('key_residues', [])
            
            colors = ['blue' if i not in key_residues else 'red' 
                      for i in range(len(rmsf_values))]
            
            ax1.bar(residue_ids, rmsf_values, color=colors, alpha=0.7, edgecolor='black')
            ax1.set_xlabel('Residue Number')
            ax1.set_ylabel('RMSF (nm)')
            ax1.set_title('RMSF per Residue')
            ax1.grid(True, alpha=0.3, axis='y')
            
            for i, res_id in enumerate(residue_ids):
                if i < len(residue_names):
                    ax1.text(res_id, rmsf_values[i], residue_names[i],
                             ha='center', va='bottom', fontsize=8, rotation=90)
            
            ax2.barh(residue_ids, rmsf_values, color=colors, alpha=0.7, edgecolor='black')
            ax2.set_xlabel('RMSF (nm)')
            ax2.set_ylabel('Residue Number')
            ax2.set_title('RMSF per Residue (Horizontal)')
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.invert_yaxis()
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, 'rmsf_plot.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            flexible_residues = [(i, rmsf_values[i]) 
                             for i in range(len(rmsf_values)) 
                             if rmsf_values[i] > np.mean(rmsf_values) + np.std(rmsf_values)]
            flexible_residues.sort(key=lambda x: x[1], reverse=True)
            
            return {
                "success": True,
                "visualization_type": "rmsf_plot",
                "output_file": output_file,
                "flexible_residues": [
                    {"residue_id": r[0], "residue_name": residue_names[r[0]] if r[0] < len(residue_names) else 'UNK', 
                     "rmsf": float(r[1])}
                    for r in flexible_residues[:10]
                ],
                "statistics": {
                    "mean_rmsf": float(np.mean(rmsf_values)) if rmsf_values else 0,
                    "std_rmsf": float(np.std(rmsf_values)) if rmsf_values else 0,
                    "max_rmsf": float(np.max(rmsf_values)) if rmsf_values else 0,
                    "flexible_count": len(flexible_residues)
                },
                "message": "RMSF图创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "plot_failed"
            }
    
    def create_conformation_heatmap(self, pca_data: Dict, tsne_data: Dict) -> Dict:
        """创建构象分布热图
        
        Args:
            pca_data: PCA数据
            tsne_data: t-SNE数据
            
        Returns:
            可视化结果
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_plot("Conformation Heatmap")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            pca_coords = np.array(pca_data.get('coordinates', []))
            tsne_coords = np.array(tsne_data.get('coordinates', []))
            cluster_labels = pca_data.get('cluster_labels', [])
            
            n_clusters = len(set(cluster_labels))
            colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
            
            if len(pca_coords) > 0:
                scatter1 = ax1.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                                      c=[colors[label] for label in cluster_labels], 
                                      alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                ax1.set_xlabel('PC1')
                ax1.set_ylabel('PC2')
                ax1.set_title('PCA - Conformation Distribution')
                ax1.grid(True, alpha=0.3)
                
                if pca_data.get('explained_variance'):
                    ax1.text(0.02, 0.98, f'PC1: {pca_data["explained_variance"][0]:.2%}',
                             transform=ax1.transAxes, fontsize=10, verticalalignment='top')
                    ax1.text(0.02, 0.94, f'PC2: {pca_data["explained_variance"][1]:.2%}',
                             transform=ax1.transAxes, fontsize=10, verticalalignment='top')
            
            if len(tsne_coords) > 0:
                scatter2 = ax2.scatter(tsne_coords[:, 0], tsne_coords[:, 1], 
                                       c=[colors[label] for label in cluster_labels], 
                                       alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
                ax2.set_xlabel('t-SNE 1')
                ax2.set_ylabel('t-SNE 2')
                ax2.set_title('t-SNE - Conformation Distribution')
                ax2.grid(True, alpha=0.3)
            
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=colors[i], markersize=10, 
                                        label=f'Cluster {i+1}') 
                           for i in range(n_clusters)]
            ax2.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, 'conformation_heatmap.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "conformation_heatmap",
                "output_file": output_file,
                "n_clusters": n_clusters,
                "n_conformations": len(pca_coords),
                "message": "构象热图创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "heatmap_failed"
            }
    
    def create_interaction_network(self, interaction_data: Dict) -> Dict:
        """创建相互作用网络图
        
        Args:
            interaction_data: 相互作用数据
            
        Returns:
            可视化结果
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_plot("Interaction Network")
        
        try:
            fig, ax = plt.subplots(figsize=(12, 10))
            
            nodes = interaction_data.get('nodes', [])
            edges = interaction_data.get('edges', [])
            key_residues = interaction_data.get('key_residues', [])
            
            pos = np.array([[node['x'], node['y']] for node in nodes])
            node_colors = ['red' if node['id'] in key_residues else 'lightblue' 
                          for node in nodes]
            node_sizes = [300 if node['id'] in key_residues else 150 
                         for node in nodes]
            
            for i, node in enumerate(nodes):
                ax.scatter(pos[i, 0], pos[i, 1], 
                         color=node_colors[i], s=node_sizes[i], 
                         edgecolors='black', linewidth=1.5, zorder=2)
                
                ax.text(pos[i, 0], pos[i, 1], f"{node.get('name', '')}\n{node.get('id', '')}",
                         ha='center', va='center', fontsize=8, 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), zorder=3)
            
            for edge in edges:
                source = edge['source']
                target = edge['target']
                strength = edge.get('strength', 1.0)
                
                if source < len(pos) and target < len(pos):
                    ax.plot([pos[source, 0], pos[target, 0]],
                            [pos[source, 1], pos[target, 1]],
                            color='gray', alpha=min(0.8, strength * 0.5), 
                            linewidth=strength * 2, zorder=1)
            
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Residue Interaction Network')
            ax.grid(True, alpha=0.3)
            ax.axis('equal')
            
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='red', markersize=12, 
                            label='Key Residue', linestyle='None'),
                plt.Line2D([0], [0], marker='o', color='lightblue', markersize=10, 
                            label='Normal Residue', linestyle='None')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            output_file = os.path.join(self.output_dir, 'interaction_network.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "interaction_network",
                "output_file": output_file,
                "n_nodes": len(nodes),
                "n_edges": len(edges),
                "key_residues": key_residues,
                "message": "相互作用网络图创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "network_failed"
            }
    
    def create_comprehensive_dashboard(self, trajectory_data: Dict, 
                                   analysis_data: Dict,
                                   output_format: str = 'html') -> Dict:
        """创建综合仪表板
        
        Args:
            trajectory_data: 轨迹数据
            analysis_data: 分析数据
            output_format: 输出格式
            
        Returns:
            仪表板创建结果
        """
        try:
            if output_format == 'html':
                return self._create_html_dashboard(trajectory_data, analysis_data)
            else:
                return self._create_image_dashboard(trajectory_data, analysis_data)
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "dashboard_failed"
            }
    
    def _create_html_dashboard(self, trajectory_data: Dict, analysis_data: Dict) -> Dict:
        """创建HTML仪表板"""
        rmsd_stats = analysis_data.get('rmsd', {})
        rmsf_stats = analysis_data.get('rmsf', {})
        key_residues = analysis_data.get('key_residues', [])
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Trajectory Analysis Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background: #1e3a5f; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
        .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
        .card h3 {{ color: #1e3a5f; margin-top: 0; }}
        .stat {{ margin: 10px 0; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #333; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        .key-residue {{ background: #ffebee; border-left: 4px solid #ff6b6b; padding: 10px; margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🧬 Trajectory Analysis Dashboard</h1>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="dashboard">
        <div class="card">
            <h3>RMSD Statistics</h3>
            <div class="stat">
                <div class="stat-value">{rmsd_stats.get('mean', 'N/A'):.3f} nm</div>
                <div class="stat-label">Mean RMSD</div>
            </div>
            <div class="stat">
                <div class="stat-value">{rmsd_stats.get('std', 'N/A'):.3f} nm</div>
                <div class="stat-label">Std Deviation</div>
            </div>
        </div>
        
        <div class="card">
            <h3>RMSF Statistics</h3>
            <div class="stat">
                <div class="stat-value">{rmsf_stats.get('mean', 'N/A'):.3f} nm</div>
                <div class="stat-label">Mean RMSF</div>
            </div>
            <div class="stat">
                <div class="stat-value">{rmsf_stats.get('flexible_count', 'N/A')}</div>
                <div class="stat-label">Flexible Residues</div>
            </div>
        </div>
        
        <div class="card">
            <h3>Key Residues</h3>
            <div class="stat">
                <div class="stat-value">{len(key_residues)}</div>
                <div class="stat-label">Total Key Residues</div>
            </div>
        </div>
        
        <div class="card">
            <h3>Key Residue Details</h3>
            {"".join([f'<div class="key-residue"><strong>{res.get("residue_name", "UNK")}{res.get("residue_id", "")}</strong> - RMSF: {res.get("rmsf", 0):.3f} nm</div>' for res in key_residues[:10]])}
        </div>
    </div>
</body>
</html>
        """
        
        output_file = os.path.join(self.output_dir, 'trajectory_dashboard.html')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return {
            "success": True,
            "visualization_type": "html_dashboard",
            "output_file": output_file,
            "message": "HTML仪表板创建成功"
        }
    
    def _create_image_dashboard(self, trajectory_data: Dict, analysis_data: Dict) -> Dict:
        """创建图片仪表板"""
        if not MATPLOTLIB_AVAILABLE:
            return self._mock_plot("Dashboard")
        
        try:
            fig = plt.figure(figsize=(16, 10))
            
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            
            rmsd_data = analysis_data.get('rmsd', {})
            rmsf_data = analysis_data.get('rmsf', {})
            
            ax1 = fig.add_subplot(gs[0, 0])
            time_points = rmsd_data.get('time_points', [])
            rmsd_values = rmsd_data.get('rmsd_values', [])
            if time_points and rmsd_values:
                ax1.plot(time_points, rmsd_values, 'b-', linewidth=2)
                ax1.set_xlabel('Time (ns)')
                ax1.set_ylabel('RMSD (nm)')
                ax1.set_title('RMSD vs Time')
                ax1.grid(True, alpha=0.3)
            
            ax2 = fig.add_subplot(gs[0, 1])
            residue_ids = rmsf_data.get('residue_ids', [])
            rmsf_values = rmsf_data.get('rmsf_values', [])
            if residue_ids and rmsf_values:
                ax2.bar(residue_ids, rmsf_values, color='green', alpha=0.7)
                ax2.set_xlabel('Residue Number')
                ax2.set_ylabel('RMSF (nm)')
                ax2.set_title('RMSF per Residue')
                ax2.grid(True, alpha=0.3, axis='y')
            
            ax3 = fig.add_subplot(gs[0, 2])
            key_residues = analysis_data.get('key_residues', [])
            if key_residues:
                residue_names = [res.get('residue_name', 'UNK') for res in key_residues[:10]]
                confidences = [res.get('probability', 0) * 100 for res in key_residues[:10]]
                ax3.barh(range(len(residue_names)), confidences, color='red', alpha=0.7)
                ax3.set_yticks(range(len(residue_names)))
                ax3.set_yticklabels(residue_names)
                ax3.set_xlabel('Confidence (%)')
                ax3.set_title('Key Residue Confidence')
                ax3.grid(True, alpha=0.3, axis='x')
            
            ax4 = fig.add_subplot(gs[1, 0])
            if time_points and rmsd_values:
                ax4.hist(rmsd_values, bins=30, color='blue', alpha=0.7)
                ax4.set_xlabel('RMSD (nm)')
                ax4.set_ylabel('Frequency')
                ax4.set_title('RMSD Distribution')
                ax4.grid(True, alpha=0.3)
            
            ax5 = fig.add_subplot(gs[1, 1])
            pca_data = analysis_data.get('pca', {})
            pca_coords = np.array(pca_data.get('coordinates', []))
            cluster_labels = pca_data.get('cluster_labels', [])
            if len(pca_coords) > 0:
                colors = plt.cm.tab10(np.linspace(0, 1, len(set(cluster_labels))))
                ax5.scatter(pca_coords[:, 0], pca_coords[:, 1], 
                           c=[colors[label] for label in cluster_labels], 
                           alpha=0.6, s=50)
                ax5.set_xlabel('PC1')
                ax5.set_ylabel('PC2')
                ax5.set_title('PCA - Conformation Distribution')
                ax5.grid(True, alpha=0.3)
            
            ax6 = fig.add_subplot(gs[1, 2])
            stats_text = f"""
Trajectory Statistics:
------------------------
Total Frames: {trajectory_data.get('n_frames', 'N/A')}
Simulation Time: {trajectory_data.get('simulation_time', 'N/A')} ns
Mean RMSD: {rmsd_data.get('mean', 'N/A'):.3f} nm
Mean RMSF: {rmsf_data.get('mean', 'N/A'):.3f} nm
Key Residues: {len(key_residues)}
            """
            ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, 
                     fontsize=10, verticalalignment='center', family='monospace')
            ax6.axis('off')
            ax6.set_title('Summary')
            
            output_file = os.path.join(self.output_dir, 'trajectory_dashboard.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return {
                "success": True,
                "visualization_type": "image_dashboard",
                "output_file": output_file,
                "message": "图片仪表板创建成功"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "dashboard_failed"
            }
    
    def _mock_3d_plot(self, trajectory_data: Dict) -> Dict:
        """模拟3D图"""
        return {
            "success": True,
            "visualization_type": "3d_trajectory",
            "output_file": os.path.join(self.output_dir, '3d_trajectory.png'),
            "n_frames": len(trajectory_data.get('coordinates', [])),
            "message": "3D轨迹图创建成功（模拟）",
            "note": "Matplotlib未安装，使用模拟结果"
        }
    
    def _mock_animation(self, trajectory_data: Dict) -> Dict:
        """模拟动画"""
        return {
            "success": True,
            "visualization_type": "animation",
            "output_file": os.path.join(self.output_dir, 'trajectory_animation.gif'),
            "n_frames": len(trajectory_data.get('coordinates', [])),
            "format": "gif",
            "message": "轨迹动画创建成功（模拟）",
            "note": "Matplotlib未安装，使用模拟结果"
        }
    
    def _mock_plot(self, plot_type: str) -> Dict:
        """模拟图"""
        return {
            "success": True,
            "visualization_type": plot_type,
            "output_file": os.path.join(self.output_dir, f'{plot_type.lower().replace(" ", "_")}.png'),
            "message": f"{plot_type}创建成功（模拟）",
            "note": "Matplotlib未安装，使用模拟结果"
        }

trajectory_visualizer = TrajectoryVisualizer()
