import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster
import json

try:
    import MDAnalysis as mda
    MDA_AVAILABLE = True
except ImportError:
    MDA_AVAILABLE = False
    print("MDAnalysis未安装，将使用简化分析")

# 单独导入分析模块
try:
    from MDAnalysis.analysis import rms
    RMS_AVAILABLE = True
except ImportError:
    RMS_AVAILABLE = False
    print("MDAnalysis RMS分析模块未安装")

# RMSF需要手动计算，MDAnalysis 2.10.0版本没有rmsf模块

class TrajectoryAnalyzer:
    """轨迹分析器 - 支持PCA、t-SNE降维和智能抽帧"""
    
    def __init__(self):
        self.analysis_results = {}
        
    def load_trajectory(self, topology_file: str, trajectory_file: str) -> Dict:
        """加载轨迹文件
        
        Args:
            topology_file: 拓扑文件（.pdb, .psf, .prmtop, .top, .tpr）
            trajectory_file: 轨迹文件（.xtc, .trr, .dcd）
            
        Returns:
            加载结果
            
        注意:
            - 如果使用aggressive清理级别删除了.top文件，可以使用.tpr文件
            - .tpr文件包含完整的系统信息，可用于轨迹分析
        """
        if not MDA_AVAILABLE:
            return {
                "success": False,
                "error": "MDAnalysis库未安装",
                "suggestion": "请安装: pip install MDAnalysis"
            }
        
        try:
            universe = mda.Universe(topology_file, trajectory_file)
            
            # 存储universe对象以便后续使用
            self.universe = universe
            
            return {
                "success": True,
                "topology_file": topology_file,
                "trajectory_file": trajectory_file,
                "n_frames": len(universe.trajectory),
                "n_atoms": universe.atoms.n_atoms,
                "n_residues": universe.residues.n_residues,
                "dt": universe.trajectory.dt,
                "total_time": universe.trajectory.n_frames * universe.trajectory.dt
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "load_failed"
            }
    
    def extract_key_frames(self, topology_file: str, trajectory_file: str,
                         num_frames: int = 100, method: str = "rmsd") -> Dict:
        """智能抽帧 - 基于RMSD聚类
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            num_frames: 提取的关键帧数
            method: 抽帧方法（rmsd, clustering, uniform）
            
        Returns:
            抽帧结果
        """
        load_result = self.load_trajectory(topology_file, trajectory_file)
        
        if not load_result["success"]:
            return load_result
        
        universe = self.universe
        total_frames = load_result["n_frames"]
        
        try:
            if method == "rmsd":
                return self._extract_by_rmsd(universe, num_frames)
            elif method == "clustering":
                return self._extract_by_clustering(universe, num_frames)
            elif method == "uniform":
                return self._extract_by_uniform(universe, num_frames)
            else:
                return {
                    "success": False,
                    "error": f"未知方法: {method}",
                    "status": "invalid_method"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "extraction_failed"
            }
    
    def _extract_by_rmsd(self, universe, num_frames: int) -> Dict:
        """基于RMSD的抽帧"""
        protein = universe.select_atoms('protein')
        # 记录第一帧坐标作为参考
        universe.trajectory[0]
        ref_coords = protein.positions.copy()

        # 计算每帧相对于第一帧的RMSD
        rmsd_values = []
        for ts in universe.trajectory:
            rmsd_val = rms.rmsd(protein.positions, ref_coords)
            rmsd_values.append(float(rmsd_val))

        rmsd_values = np.array(rmsd_values)

        # 基于RMSD变化选择关键帧
        rmsd_diff = np.diff(rmsd_values)
        rmsd_diff_abs = np.abs(rmsd_diff)

        # 选择RMSD变化最大的帧（diff长度比frames少1，+1对齐到后一帧）
        top_indices = np.argsort(rmsd_diff_abs)[-num_frames:] + 1
        top_indices = np.sort(top_indices)

        # 提取关键帧的坐标
        key_frames = []
        for idx in top_indices:
            universe.trajectory[int(idx)]
            coords = protein.positions
            key_frames.append({
                "frame_index": int(idx),
                "time": float(universe.trajectory[int(idx)].time),
                "rmsd": float(rmsd_values[idx]),
                "rmsd_diff": float(rmsd_diff_abs[idx - 1]),
                "coordinates": coords.tolist()
            })
        
        return {
            "success": True,
            "method": "rmsd_based",
            "total_frames": len(universe.trajectory),
            "extracted_frames": len(key_frames),
            "key_frames": key_frames,
            "statistics": {
                "mean_rmsd": float(np.mean(rmsd_values)),
                "std_rmsd": float(np.std(rmsd_values)),
                "max_rmsd": float(np.max(rmsd_values)),
                "min_rmsd": float(np.min(rmsd_values))
            }
        }
    
    def _extract_by_clustering(self, universe, num_frames: int) -> Dict:
        """基于聚类的抽帧"""
        # 提取所有蛋白质原子坐标
        all_coords = []
        for ts in universe.trajectory:
            coords = universe.select_atoms('protein').positions
            all_coords.append(coords.flatten())
        
        all_coords = np.array(all_coords)
        
        # K-means聚类
        kmeans = KMeans(n_clusters=num_frames, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(all_coords)
        
        # 为每个聚类选择代表性帧
        key_frames = []
        for cluster_id in range(num_frames):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            center_idx = cluster_indices[len(cluster_indices) // 2]
            
            universe.trajectory[center_idx]
            coords = universe.select_atoms('protein').positions
            
            # 计算到聚类中心的距离
            distances = []
            for idx in cluster_indices:
                universe.trajectory[idx]
                cluster_coords = universe.select_atoms('protein').positions
                dist = np.linalg.norm(coords - cluster_coords)
                distances.append(dist)
            
            key_frames.append({
                "frame_index": int(center_idx),
                "time": float(universe.trajectory[center_idx].time),
                "cluster_id": int(cluster_id),
                "cluster_size": int(len(cluster_indices)),
                "mean_distance": float(np.mean(distances)),
                "coordinates": coords.tolist()
            })
        
        # 按聚类大小排序
        key_frames.sort(key=lambda x: x["cluster_size"], reverse=True)
        
        return {
            "success": True,
            "method": "clustering_based",
            "total_frames": len(universe.trajectory),
            "extracted_frames": len(key_frames),
            "key_frames": key_frames,
            "statistics": {
                "clusters": num_frames,
                "avg_cluster_size": float(np.mean([f["cluster_size"] for f in key_frames])),
                "avg_mean_distance": float(np.mean([f["mean_distance"] for f in key_frames]))
            }
        }
    
    def _extract_by_uniform(self, universe, num_frames: int) -> Dict:
        """均匀抽帧"""
        total_frames = len(universe.trajectory)
        step = max(1, total_frames // num_frames)
        
        key_frames = []
        for i in range(0, total_frames, step):
            if len(key_frames) >= num_frames:
                break
                
            universe.trajectory[i]
            coords = universe.select_atoms('protein').positions
            
            key_frames.append({
                "frame_index": int(i),
                "time": float(universe.trajectory[i].time),
                "coordinates": coords.tolist()
            })
        
        return {
            "success": True,
            "method": "uniform",
            "total_frames": total_frames,
            "extracted_frames": len(key_frames),
            "key_frames": key_frames,
            "statistics": {
                "sampling_step": step,
                "coverage": float(len(key_frames) / num_frames)
            }
        }
    
    def pca_analysis(self, topology_file: str, trajectory_file: str,
                    n_components: int = 2, atom_selection: str = "protein") -> Dict:
        """PCA降维分析
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            n_components: 主成分数（默认2）
            atom_selection: 原子选择（protein, backbone, ca）
            
        Returns:
            PCA分析结果
        """
        load_result = self.load_trajectory(topology_file, trajectory_file)
        
        if not load_result["success"]:
            return load_result
        
        universe = self.universe
        selection = universe.select_atoms(atom_selection)
        
        try:
            # 收集所有帧的坐标
            coords_data = []
            for ts in universe.trajectory:
                coords = selection.positions
                coords_data.append(coords.flatten())
            
            coords_data = np.array(coords_data)
            
            # PCA分析
            pca = PCA(n_components=n_components)
            coords_pca = pca.fit_transform(coords_data)
            
            # 计算每个主成分解释的方差
            explained_variance = pca.explained_variance_ratio_
            
            # 计算构象聚类
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_labels = kmeans.fit_predict(coords_pca)
            
            return {
                "success": True,
                "n_components": n_components,
                "explained_variance": explained_variance.tolist(),
                "total_variance": float(np.sum(explained_variance)),
                "principal_components": pca.components_.tolist(),
                "mean_structure": pca.mean_.reshape(-1, 3).tolist(),
                "transformed_coords": coords_pca.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "n_frames": len(coords_data),
                "atom_selection": atom_selection,
                "analysis_summary": {
                    "pc1_variance": float(explained_variance[0]),
                    "pc2_variance": float(explained_variance[1]) if n_components >= 2 else None,
                    "dominant_motion": "PC1" if explained_variance[0] > 0.5 else "multiple"
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "pca_failed"
            }
    
    def tsne_analysis(self, topology_file: str, trajectory_file: str,
                     n_components: int = 2, perplexity: float = 30.0,
                     n_iter: int = 1000, atom_selection: str = "protein") -> Dict:
        """t-SNE非线性降维可视化
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            n_components: 降维后的维度（默认2）
            perplexity: t-SNE困惑度（默认30）
            n_iter: 迭代次数（默认1000）
            atom_selection: 原子选择
            
        Returns:
            t-SNE分析结果
        """
        load_result = self.load_trajectory(topology_file, trajectory_file)
        
        if not load_result["success"]:
            return load_result
        
        universe = self.universe
        selection = universe.select_atoms(atom_selection)
        
        # 先用PCA降维以减少维度
        coords_data = []
        for ts in universe.trajectory:
            coords = selection.positions
            coords_data.append(coords.flatten())
        
        coords_data = np.array(coords_data)
        
        try:
            # PCA预降维
            pca = PCA(n_components=min(50, coords_data.shape[1]))
            coords_pca = pca.fit_transform(coords_data)
            
            # t-SNE降维
            tsne = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                n_iter=n_iter,
                random_state=42,
                learning_rate=200.0,
                init='pca'
            )
            coords_tsne = tsne.fit_transform(coords_pca)
            
            # 聚类
            kmeans = KMeans(n_clusters=5, random_state=42)
            cluster_labels = kmeans.fit_predict(coords_tsne)
            
            # 计算KL散度（t-SNE损失）
            kl_divergence = tsne.kl_divergence_ if hasattr(tsne, 'kl_divergence_') else None
            
            return {
                "success": True,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_iter": n_iter,
                "transformed_coords": coords_tsne.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "n_frames": len(coords_data),
                "atom_selection": atom_selection,
                "kl_divergence": float(kl_divergence) if kl_divergence else None,
                "cluster_centers": kmeans.cluster_centers_.tolist(),
                "analysis_summary": {
                    "pca_variance_explained": float(np.sum(pca.explained_variance_ratio_)),
                    "n_clusters": 5,
                    "convergence": tsne.n_iter_ if hasattr(tsne, 'n_iter_') else n_iter
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "tsne_failed"
            }
    
    def cluster_conformations(self, topology_file: str, trajectory_file: str,
                          n_clusters: int = 5, method: str = "kmeans") -> Dict:
        """构象聚类分析
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            n_clusters: 聚类数
            method: 聚类方法（kmeans, hierarchical）
            
        Returns:
            聚类分析结果
        """
        load_result = self.load_trajectory(topology_file, trajectory_file)
        
        if not load_result["success"]:
            return load_result
        
        universe = self.universe
        selection = universe.select_atoms('protein')
        
        # 收集坐标
        coords_data = []
        for ts in universe.trajectory:
            coords = selection.positions
            coords_data.append(coords.flatten())
        
        coords_data = np.array(coords_data)
        
        try:
            if method == "kmeans":
                return self._kmeans_clustering(coords_data, n_clusters, universe)
            elif method == "hierarchical":
                return self._hierarchical_clustering(coords_data, n_clusters, universe)
            else:
                return {
                    "success": False,
                    "error": f"未知方法: {method}",
                    "status": "invalid_method"
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "clustering_failed"
            }
    
    def _kmeans_clustering(self, coords_data: np.ndarray, n_clusters: int, universe) -> Dict:
        """K-means聚类"""
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        cluster_labels = kmeans.fit_predict(coords_data)
        
        # 分析每个聚类
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            
            # 找到代表性帧
            center_idx = cluster_indices[len(cluster_indices) // 2]
            universe.trajectory[center_idx]
            center_coords = universe.select_atoms('protein').positions
            
            # 计算聚类内距离
            cluster_coords = coords_data[cluster_indices]
            if len(cluster_coords) > 1:
                distances = pdist(cluster_coords)
                avg_intra_distance = float(np.mean(distances))
            else:
                avg_intra_distance = 0.0
            
            clusters.append({
                "cluster_id": int(cluster_id),
                "representative_frame": int(center_idx),
                "cluster_size": int(len(cluster_indices)),
                "center_coordinates": center_coords.tolist(),
                "avg_intra_distance": avg_intra_distance,
                "frame_indices": cluster_indices.tolist()
            })
        
        return {
            "success": True,
            "method": "kmeans",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "cluster_labels": cluster_labels.tolist(),
            "inertia": float(kmeans.inertia_),
            "analysis_summary": {
                "total_frames": len(coords_data),
                "avg_cluster_size": float(np.mean([c["cluster_size"] for c in clusters])),
                "largest_cluster": max(clusters, key=lambda x: x["cluster_size"]),
                "smallest_cluster": min(clusters, key=lambda x: x["cluster_size"])
            }
        }
    
    def _hierarchical_clustering(self, coords_data: np.ndarray, n_clusters: int, universe) -> Dict:
        """层次聚类"""
        # 计算距离矩阵
        distance_matrix = pdist(coords_data)
        condensed_distance_matrix = squareform(distance_matrix)
        
        # 层次聚类
        linkage_matrix = linkage(condensed_distance_matrix, method='ward')
        
        # 切割树
        cluster_labels = fcluster(linkage_matrix, t=n_clusters, criterion='maxclust')
        
        # 分析聚类
        clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id + 1)[0]
            
            if len(cluster_indices) > 0:
                center_idx = cluster_indices[len(cluster_indices) // 2]
                universe.trajectory[center_idx]
                center_coords = universe.select_atoms('protein').positions
                
                clusters.append({
                    "cluster_id": int(cluster_id),
                    "representative_frame": int(center_idx),
                    "cluster_size": int(len(cluster_indices)),
                    "center_coordinates": center_coords.tolist(),
                    "frame_indices": cluster_indices.tolist()
                })
        
        return {
            "success": True,
            "method": "hierarchical",
            "n_clusters": n_clusters,
            "clusters": clusters,
            "cluster_labels": cluster_labels.tolist(),
            "linkage_matrix": linkage_matrix.tolist(),
            "analysis_summary": {
                "total_frames": len(coords_data),
                "avg_cluster_size": float(np.mean([c["cluster_size"] for c in clusters])) if clusters else 0,
                "largest_cluster": max(clusters, key=lambda x: x["cluster_size"]) if clusters else None,
                "smallest_cluster": min(clusters, key=lambda x: x["cluster_size"]) if clusters else None
            }
        }
    
    def analyze_rmsd_rmsf(self, topology_file: str, trajectory_file: str) -> Dict:
        """分析RMSD和RMSF
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            
        Returns:
            RMSD和RMSF分析结果
        """
        load_result = self.load_trajectory(topology_file, trajectory_file)
        
        if not load_result["success"]:
            return load_result
        
        universe = self.universe
        protein = universe.select_atoms('protein')
        
        # 检查分析模块是否可用
        if not MDA_AVAILABLE:
            return {
                "success": False,
                "error": "MDAnalysis库未安装",
                "suggestion": "请安装: pip install MDAnalysis"
            }
        
        try:
            # RMSD分析
            if RMS_AVAILABLE:
                rmsd_analysis_obj = rms.RMSD(protein, protein).run()
                # results.rmsd 是 (n_frames, 3) 的二维数组，第3列才是RMSD值
                raw = rmsd_analysis_obj.results.rmsd if hasattr(rmsd_analysis_obj, 'results') else rmsd_analysis_obj.rmsd
                raw = np.array(raw)
                rmsd_values = raw[:, 2] if raw.ndim == 2 else raw
            else:
                return {
                    "success": False,
                    "error": "MDAnalysis RMS分析模块未安装",
                    "suggestion": "请安装完整版MDAnalysis"
                }
            
            # RMSF分析（优化版 - 一次性读取所有坐标）
            print(f"[RMSF] 开始计算RMSF，残基数: {len(protein.residues)}, 帧数: {len(universe.trajectory)}")
            
            # 一次性收集所有帧的蛋白质坐标
            all_coords = []
            for ts in universe.trajectory:
                coords = protein.positions.copy()
                all_coords.append(coords)
            
            all_coords = np.array(all_coords)
            n_frames, n_atoms, _ = all_coords.shape
            
            rmsf_values = []
            rmsf_info = []
            
            # 对每个残基计算RMSF
            for i, residue in enumerate(protein.residues):
                # 获取该残基在所有帧中的坐标
                residue_atom_indices = residue.atoms.ix
                residue_coords = all_coords[:, residue_atom_indices, :]
                
                # 计算RMSF：对每个原子算RMSF后取均值（标准做法）
                mean_coord = np.mean(residue_coords, axis=0)  # (n_atoms, 3)
                per_atom_rmsf = np.sqrt(np.mean(np.sum((residue_coords - mean_coord) ** 2, axis=2), axis=0))  # (n_atoms,)
                rmsf_val = float(np.mean(per_atom_rmsf))
                rmsf_values.append(rmsf_val)
                
                rmsf_info.append({
                    "residue_id": int(residue.resid),
                    "residue_name": residue.resname,
                    "rmsf": float(rmsf_val.item() if hasattr(rmsf_val, 'item') else rmsf_val)
                })
                
                # 进度提示
                if (i + 1) % 10 == 0:
                    print(f"[RMSF] 已处理 {i + 1}/{len(protein.residues)} 个残基")
            
            # 找到柔性最大的残基
            flexible_residues = []
            for info in rmsf_info:
                if info["rmsf"] > np.mean(rmsf_values) + np.std(rmsf_values):
                    flexible_residues.append(info)
            
            return {
                "success": True,
                "rmsd": {
                    "values": rmsd_values.tolist() if isinstance(rmsd_values, np.ndarray) else [float(x.item() if hasattr(x, 'item') else x) for x in rmsd_values],
                    "mean": float(np.mean(rmsd_values).item() if hasattr(np.mean(rmsd_values), 'item') else np.mean(rmsd_values)),
                    "std": float(np.std(rmsd_values).item() if hasattr(np.std(rmsd_values), 'item') else np.std(rmsd_values)),
                    "max": float(np.max(rmsd_values).item() if hasattr(np.max(rmsd_values), 'item') else np.max(rmsd_values)),
                    "min": float(np.min(rmsd_values).item() if hasattr(np.min(rmsd_values), 'item') else np.min(rmsd_values))
                },
                "rmsf": {
                    "values": rmsf_values.tolist() if isinstance(rmsf_values, np.ndarray) else [float(x.item() if hasattr(x, 'item') else x) for x in rmsf_values],
                    "mean": float(np.mean(rmsf_values).item() if hasattr(np.mean(rmsf_values), 'item') else np.mean(rmsf_values)),
                    "std": float(np.std(rmsf_values).item() if hasattr(np.std(rmsf_values), 'item') else np.std(rmsf_values)),
                    "max": float(np.max(rmsf_values).item() if hasattr(np.max(rmsf_values), 'item') else np.max(rmsf_values)),
                    "min": float(np.min(rmsf_values).item() if hasattr(np.min(rmsf_values), 'item') else np.min(rmsf_values))
                },
                "flexible_residues": flexible_residues[:10],
                "n_frames": len(rmsd_values),
                "n_residues": len(rmsf_info),
                "analysis_summary": {
                    "flexible_residue_count": len(flexible_residues),
                    "avg_rmsd": float(np.mean(rmsd_values).item() if hasattr(np.mean(rmsd_values), 'item') else np.mean(rmsd_values)),
                    "avg_rmsf": float(np.mean(rmsf_values).item() if hasattr(np.mean(rmsf_values), 'item') else np.mean(rmsf_values))
                }
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "analysis_failed"
            }

    def full_trajectory_analysis(self, topology_file: str, trajectory_file: str) -> Dict:
        """完整轨迹分析
        
        Args:
            topology_file: 拓扑文件
            trajectory_file: 轨迹文件
            
        Returns:
            完整分析结果
        """
        results = {}
        
        # 加载轨迹
        load_result = self.load_trajectory(topology_file, trajectory_file)
        if not load_result["success"]:
            return load_result
        results["load"] = load_result
        
        # RMSD/RMSF分析
        rmsd_result = self.analyze_rmsd_rmsf(topology_file, trajectory_file)
        results["rmsd_rmsf"] = rmsd_result
        
        # PCA分析
        pca_result = self.pca_analysis(topology_file, trajectory_file, n_components=2)
        results["pca"] = pca_result
        
        # t-SNE分析
        tsne_result = self.tsne_analysis(topology_file, trajectory_file, n_components=2)
        results["tsne"] = tsne_result
        
        # 聚类分析
        cluster_result = self.cluster_conformations(topology_file, trajectory_file, n_clusters=5)
        results["clustering"] = cluster_result
        
        # 智能抽帧
        key_frames_result = self.extract_key_frames(topology_file, trajectory_file, num_frames=100, method="rmsd")
        results["key_frames"] = key_frames_result
        
        results["success"] = True
        results["analysis_time"] = "complete"
        
        return results

trajectory_analyzer = TrajectoryAnalyzer()
