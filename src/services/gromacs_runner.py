import os
import subprocess
import json
import time
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np

class GromacsRunner:
    """GROMACS分子动力学模拟执行器
    
    支持完整的MD模拟工作流：
    - 系统准备
    - 能量最小化
    - NVT/NPT平衡
    - 生产运行
    - 轨迹分析
    - GPU加速
    """
    
    def __init__(self, gmx_path: str = None):
        """初始化GROMACS运行器
        
        Args:
            gmx_path: GROMACS可执行文件路径，默认为None（自动检测GPU版本）
        """
        # 优先使用GPU版本和数据文件
        if gmx_path is None:
            self.gmx_path = "/usr/local/gromacs-gpu/bin/gmx"
            # 设置环境变量使用GPU版本的数据文件
            os.environ['GMXLIB'] = '/usr/local/gromacs-gpu/share/gromacs/top'
            
            # 如果GPU版本不存在，则自动检测
            try:
                result = subprocess.run(
                    [self.gmx_path, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if "GPU support" not in result.stdout or "CUDA" not in result.stdout:
                    raise FileNotFoundError
            except (FileNotFoundError, subprocess.TimeoutExpired):
                self.gmx_path = self._detect_gpu_gromacs()
        else:
            self.gmx_path = gmx_path
            
        self.simulation_jobs = {}
        self.gpu_enabled = self._check_gpu_support()
        
        print(f"[GROMACS] 使用版本: {self.gmx_path}")
        print(f"[GROMACS] GPU支持: {'启用' if self.gpu_enabled else '禁用'}")
        print(f"[GROMACS] 数据文件路径: {os.environ.get('GMXLIB', 'default')}")
    
    def _detect_gpu_gromacs(self) -> str:
        """自动检测GPU版本的GROMACS"""
        gpu_paths = [
            "/usr/local/gromacs-gpu/bin/gmx",
            "/usr/local/gromacs/bin/gmx",
            "gmx"
        ]
        
        for path in gpu_paths:
            try:
                result = subprocess.run(
                    [path, "version"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if "GPU support" in result.stdout and "CUDA" in result.stdout:
                    print(f"[GROMACS] 找到GPU版本: {path}")
                    return path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        print("[GROMACS] 未找到GPU版本，使用默认gmx")
        return "gmx"
    
    def _check_gpu_support(self) -> bool:
        """检查GPU支持"""
        try:
            result = subprocess.run(
                [self.gmx_path, "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            return "GPU support" in result.stdout and "CUDA" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        
    def check_gromacs_installation(self) -> Dict:
        """检查GROMACS是否正确安装"""
        try:
            result = subprocess.run(
                [self.gmx_path, "version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            version_info = result.stdout.strip()
            
            return {
                "installed": True,
                "version": version_info,
                "gmx_path": self.gmx_path,
                "gpu_enabled": self.gpu_enabled,
                "status": "ready"
            }
        except FileNotFoundError:
            return {
                "installed": False,
                "error": "GROMACS未找到",
                "suggestion": "请安装GROMACS或提供正确的gmx路径",
                "status": "not_installed"
            }
        except Exception as e:
            return {
                "installed": False,
                "error": str(e),
                "status": "error"
            }
    
    def verify_gpu_usage(self) -> Dict:
        """验证GPU是否正在使用
        
        Returns:
            GPU使用状态信息
        """
        if not self.gpu_enabled:
            return {
                "gpu_enabled": False,
                "message": "GPU支持未启用",
                "suggestion": "请安装GPU版本的GROMACS"
            }
        
        return {
                "gpu_enabled": True,
                "gmx_path": self.gmx_path,
                "message": "GPU支持已启用，MD模拟将使用GPU加速",
                "note": "请使用nvidia-smi监控GPU使用情况"
            }
    
    def _cleanup_old_files(self, work_dir: str, base_name: str):
        """清理工作目录中的旧文件，但保留轨迹和分析文件"""
        import glob
        
        # 需要保留的文件扩展名
        keep_extensions = ['.xtc', '.trr', '.tpr', '.edr', '.log', '.gro']
        
        patterns = [
            f"{base_name}_processed.gro",
            f"{base_name}_topol.top",
            f"{base_name}_boxed.gro",
            f"{base_name}_solvated.gro",
            f"{base_name}_ionized.gro",
            f"{base_name}_em.*",
            f"{base_name}_nvt.*",
            f"{base_name}_npt.*",
            f"{base_name}_md.*"
        ]
        
        for pattern in patterns:
            files = glob.glob(os.path.join(work_dir, pattern))
            for file in files:
                # 检查文件扩展名是否需要保留
                _, ext = os.path.splitext(file)
                if ext in keep_extensions:
                    print(f"[CLEANUP] 保留轨迹/分析文件: {os.path.basename(file)}")
                    continue
                
                try:
                    os.remove(file)
                    print(f"[CLEANUP] 已删除旧文件: {os.path.basename(file)}")
                except Exception as e:
                    print(f"[CLEANUP] 删除文件失败: {file} - {e}")
    
    def _copy_forcefield_files(self, work_dir: str, forcefield: str):
        """复制力场文件到工作目录"""
        import shutil
        gmxlib = os.environ.get('GMXLIB', '/usr/share/gromacs/top')
        
        # 复制力场目录
        forcefield_dir = os.path.join(gmxlib, forcefield + '.ff')
        if os.path.exists(forcefield_dir):
            dest_dir = os.path.join(work_dir, forcefield + '.ff')
            shutil.copytree(forcefield_dir, dest_dir, dirs_exist_ok=True)
            print(f"[PREPARE] 复制力场文件: {forcefield_dir} -> {dest_dir}")
        
        # 复制溶剂文件
        solvent_files = [
            'tip3p.gro',
            'tip3p.itp',
            'tip4p.gro',
            'tip4p.itp',
            'tip5p.gro',
            'spc.itp',
            'spce.itp',
            'spc216.gro'
        ]
        
        for solvent_file in solvent_files:
            src_file = os.path.join(gmxlib, solvent_file)
            if os.path.exists(src_file):
                dest_file = os.path.join(work_dir, solvent_file)
                shutil.copy2(src_file, dest_file)
                print(f"[PREPARE] 复制溶剂文件: {solvent_file}")
    
    def prepare_system(self, pdb_file: str, forcefield: str = "amber99sb-ildn",
                     water_model: str = "tip4p", box_type: str = "dodecahedron",
                     distance: float = 1.0, work_dir: str = None) -> Dict:
        """准备模拟系统
        
        Args:
            pdb_file: 输入PDB文件路径
            forcefield: 力场名称，默认'amber99sb-ildn'
            water_model: 水模型，默认'tip4p' (GPU版本支持)
            box_type: 模拟盒类型，默认'dodecahedron'
            distance: 盒子边界距离(nm)，默认1.0
            work_dir: 工作目录，默认为pdb文件所在目录
            
        Returns:
            准备结果字典
        """
        if work_dir is None:
            work_dir = os.path.dirname(pdb_file)
        
        os.makedirs(work_dir, exist_ok=True)
        
        # 清理旧文件，避免力场冲突
        base_name = os.path.splitext(os.path.basename(pdb_file))[0]
        self._cleanup_old_files(work_dir, base_name)
        
        # 复制力场文件到工作目录
        self._copy_forcefield_files(work_dir, forcefield)
        
        try:
            # 步骤1: 生成拓扑文件
            cmd = [
                self.gmx_path, "pdb2gmx",
                "-f", pdb_file,
                "-o", f"{base_name}_processed.gro",
                "-p", f"{base_name}_topol.top",
                "-water", water_model,
                "-ff", forcefield
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "pdb2gmx",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            # 步骤2: 定义模拟盒
            cmd = [
                self.gmx_path, "editconf",
                "-f", f"{base_name}_processed.gro",
                "-o", f"{base_name}_boxed.gro",
                "-c", "-d", str(distance),
                "-bt", box_type
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "editconf",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            # 步骤3: 溶剂化
            cmd = [
                self.gmx_path, "solvate",
                "-cp", f"{base_name}_boxed.gro",
                "-cs", water_model,
                "-o", f"{base_name}_solvated.gro",
                "-p", f"{base_name}_topol.top"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "solvate",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            return {
                "success": True,
                "status": "prepared",
                "forcefield": forcefield,
                "water_model": water_model,
                "box_type": box_type,
                "output_files": {
                    "gro": f"{work_dir}/{base_name}_solvated.gro",
                    "top": f"{work_dir}/{base_name}_topol.top"
                },
                "message": "系统准备完成"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "step": "system_preparation",
                "error": "操作超时",
                "status": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "step": "system_preparation",
                "error": str(e),
                "status": "error"
            }
    
    def add_ions(self, gro_file: str, top_file: str, ion_type: str = "NaCl",
                 concentration: float = 0.15, work_dir: str = None) -> Dict:
        """添加离子以中和电荷
        
        Args:
            gro_file: 输入GRO文件
            top_file: 拓扑文件
            ion_type: 离子类型，默认'NaCl'
            concentration: 离子浓度(M)，默认0.15
            work_dir: 工作目录
            
        Returns:
            加离子结果
        """
        if work_dir is None:
            work_dir = os.path.dirname(gro_file)
        
        base_name = os.path.splitext(os.path.basename(gro_file))[0]
        
        try:
            # 创建简单的.mdp文件用于genion
            mdp_content = """title               = Ion minimization
integrator          = steep
dt                  = 0.002
nsteps              = 10000
nstxout             = 500
nstvout             = 500
nstenergy           = 500
nstlog              = 500
continuation         = no
constraint_algorithm = lincs
constraints         = h-bonds
cutoff-scheme       = Verlet
ns_type             = grid
nstlist             = 10
rlist               = 1.0
coulombtype         = PME
rcoulomb            = 1.0
rvdw                = 1.0
pbc                 = xyz
"""
            mdp_file = os.path.join(work_dir, "ions.mdp")
            with open(mdp_file, 'w') as f:
                f.write(mdp_content)
            
            # 步骤1: 生成tpr文件
            cmd = [
                self.gmx_path, "grompp",
                "-f", mdp_file,
                "-p", top_file,
                "-c", gro_file,
                "-o", f"{base_name}_ions.tpr",
                "-maxwarn", "1"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "grompp_ions",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            # 步骤2: 添加离子
            cmd = [
                self.gmx_path, "genion",
                "-s", f"{base_name}_ions.tpr",
                "-o", f"{base_name}_solv_ions.gro",
                "-p", top_file,
                "-pname", "NA",
                "-nname", "CL",
                "-neutral",
                "-conc", str(concentration)
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                input="13\n",
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "genion",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            return {
                "success": True,
                "status": "ions_added",
                "ion_type": ion_type,
                "concentration": concentration,
                "output_files": {
                    "gro": f"{work_dir}/{base_name}_solv_ions.gro",
                    "top": top_file
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "step": "add_ions",
                "error": str(e),
                "status": "error"
            }
    
    def energy_minimize(self, gro_file: str, top_file: str,
                      nsteps: int = 10000, emtol: float = 1000.0,
                      work_dir: str = None) -> Dict:
        """能量最小化
        
        Args:
            gro_file: 输入GRO文件
            top_file: 拓扑文件
            nsteps: 最大步数，默认50000
            emtol: 能量收敛标准(kJ/mol/nm)，默认1000
            work_dir: 工作目录
            
        Returns:
            最小化结果
        """
        if work_dir is None:
            work_dir = os.path.dirname(gro_file)
        
        base_name = os.path.splitext(os.path.basename(gro_file))[0]
        
        # 创建mdp文件
        mdp_content = f"""; Energy minimization
integrator  = steep
emtol       = {emtol}
nsteps      = {nsteps}
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rlist        = 1.0
coulombtype  = PME
rcoulomb     = 1.0
rvdw         = 1.0
pbc          = xyz
constraints  = h-bonds
"""
        
        mdp_file = os.path.join(work_dir, "em.mdp")
        with open(mdp_file, 'w') as f:
            f.write(mdp_content)
        
        try:
            # 生成tpr文件
            cmd = [
                self.gmx_path, "grompp",
                "-f", mdp_file,
                "-p", top_file,
                "-c", gro_file,
                "-o", f"{base_name}_em.tpr",
                "-maxwarn", "1"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "grompp_em",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            # 运行能量最小化（使用GPU加速）
            print(f"[GROMACS] 运行能量最小化...")
            print(f"[GROMACS] 使用路径: {self.gmx_path}")
            print(f"[GROMACS] GPU支持: {self.gpu_enabled}")
            cmd = [
                self.gmx_path, "mdrun",
                "-deffnm", f"{base_name}_em",
                "-s", f"{base_name}_em.tpr",
                "-nb", "gpu",
                "-v"
            ]
            print(f"[GROMACS] 执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=1800
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "mdrun_em",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            # 提取最终能量
            energy_file = os.path.join(work_dir, f"{base_name}_em.log")
            final_energy = self._extract_final_energy(energy_file)
            
            return {
                "success": True,
                "status": "energy_minimized",
                "final_energy": final_energy,
                "nsteps": nsteps,
                "emtol": emtol,
                "output_files": {
                    "gro": f"{work_dir}/{base_name}_em.gro",
                    "trr": f"{work_dir}/{base_name}_em.trr",
                    "log": energy_file
                }
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "step": "energy_minimization",
                "error": "能量最小化超时",
                "status": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "step": "energy_minimization",
                "error": str(e),
                "status": "error"
            }
    
    def equilibrate_system(self, gro_file: str, top_file: str,
                         temperature: float = 310.0, pressure: float = 1.0,
                         nvt_steps: int = 5000, npt_steps: int = 10000,
                         work_dir: str = None) -> Dict:
        """NVT和NPT平衡
        
        Args:
            gro_file: 输入GRO文件
            top_file: 拓扑文件
            temperature: 目标温度(K)，默认310K
            pressure: 目标压力(bar)，默认1.0 bar
            nvt_steps: NVT步数，默认50000
            npt_steps: NPT步数，默认100000
            work_dir: 工作目录
            
        Returns:
            平衡结果
        """
        if work_dir is None:
            work_dir = os.path.dirname(gro_file)
        
        base_name = os.path.splitext(os.path.basename(gro_file))[0]
        
        # NVT平衡mdp文件
        nvt_mdp_content = f"""; NVT equilibration
integrator  = md
dt          = 0.002
nsteps      = {nvt_steps}
nstxout     = 5000
nstvout     = 5000
nstfout     = 5000
nstenergy   = 500
nstlog      = 500
continuation = no
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rlist        = 1.0
coulombtype  = PME
rcoulomb     = 1.0
rvdw         = 1.0
pbc          = xyz
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1 0.1
ref_t        = {temperature} {temperature}
constraints  = h-bonds
"""
        
        # NPT平衡mdp文件
        npt_mdp_content = f"""; NPT equilibration
integrator  = md
dt          = 0.002
nsteps      = {npt_steps}
nstxout     = 5000
nstvout     = 5000
nstfout     = 5000
nstenergy   = 500
nstlog      = 500
continuation = yes
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rlist        = 1.0
coulombtype  = PME
rcoulomb     = 1.0
rvdw         = 1.0
pbc          = xyz
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1 0.1
ref_t        = {temperature} {temperature}
pcoupl       = Parrinello-Rahman
pcoupltype   = isotropic
tau_p        = 2.0
ref_p        = {pressure}
compressibility = 4.5e-5
constraints  = h-bonds
"""
        
        import time
        import traceback
        start_time = time.time()
        
        try:
            # NVT平衡
            print(f"[EQUILIB] 开始NVT平衡...")
            print(f"[EQUILIB] 预计NVT步数: {nvt_steps:,}步")
            print(f"[EQUILIB] 预计时间: {nvt_steps * 0.002:.1f} ps ({nvt_steps * 0.002 / 1000:.2f} ns)")
            
            nvt_mdp = os.path.join(work_dir, "nvt.mdp")
            with open(nvt_mdp, 'w') as f:
                f.write(nvt_mdp_content)
            
            grompp_start = time.time()
            cmd = [
                self.gmx_path, "grompp",
                "-f", nvt_mdp,
                "-p", top_file,
                "-c", gro_file,
                "-o", f"{base_name}_nvt.tpr",
                "-maxwarn", "1"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            grompp_time = time.time() - grompp_start
            print(f"[EQUILIB] ✅ NVT TPR生成完成，耗时: {grompp_time:.1f}秒")
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "grompp_nvt",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            print(f"[EQUILIB] 运行NVT MD模拟...")
            mdrun_start = time.time()
            cmd = [
                self.gmx_path, "mdrun",
                "-deffnm", f"{base_name}_nvt",
                "-s", f"{base_name}_nvt.tpr",
                "-nb", "gpu",
                "-v"
            ]
            
            print(f"[EQUILIB] 执行命令: {' '.join(cmd)}")
            
            # 实时输出GROMACS进度
            process = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # 实时读取并输出进度（减少日志输出，每1000步输出一次）
            step_count = 0
            for line in process.stdout:
                line_stripped = line.strip()
                
                # 显示包含错误信息的行
                if 'error' in line.lower() or 'fatal' in line.lower() or 'failed' in line.lower():
                    print(f"[GROMACS-NVT] ⚠️  {line_stripped}")
                
                if 'step' in line.lower() and 'will finish' not in line.lower():
                    step_count += 1
                    if step_count % 10 == 0:  # 每10次输出一次（约1000步）
                        print(f"[GROMACS-NVT] {line_stripped}")
            
            # 等待进程完成
            process.wait()
            
            mdrun_time = time.time() - mdrun_start
            print(f"[EQUILIB] ✅ NVT模拟完成，耗时: {mdrun_time:.1f}秒")
            print(f"[EQUILIB] NVT总耗时: {grompp_time + mdrun_time:.1f}秒")
            
            if process.returncode != 0:
                print(f"[EQUILIB] ⚠️  NVT模拟返回码: {process.returncode}")
                return {
                    "success": False,
                    "step": "mdrun_nvt",
                    "error": f"NVT模拟失败，返回码: {process.returncode}",
                    "status": "failed"
                }
            
            # NPT平衡
            print(f"[EQUILIB] 开始NPT平衡...")
            print(f"[EQUILIB] 预计NPT步数: {npt_steps:,}步")
            print(f"[EQUILIB] 预计时间: {npt_steps * 0.002:.1f} ps ({npt_steps * 0.002 / 1000:.2f} ns)")
            
            npt_mdp = os.path.join(work_dir, "npt.mdp")
            with open(npt_mdp, 'w') as f:
                f.write(npt_mdp_content)
            
            grompp_start = time.time()
            cmd = [
                self.gmx_path, "grompp",
                "-f", npt_mdp,
                "-p", top_file,
                "-c", f"{base_name}_nvt.gro",
                "-t", f"{base_name}_nvt.cpt",
                "-o", f"{base_name}_npt.tpr",
                "-maxwarn", "1"
            ]
            
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            grompp_time = time.time() - grompp_start
            print(f"[EQUILIB] ✅ NPT TPR生成完成，耗时: {grompp_time:.1f}秒")
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "step": "grompp_npt",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            print(f"[EQUILIB] 运行NPT MD模拟...")
            mdrun_start = time.time()
            cmd = [
                self.gmx_path, "mdrun",
                "-deffnm", f"{base_name}_npt",
                "-s", f"{base_name}_npt.tpr",
                "-nb", "gpu",
                "-v"
            ]
            
            # 实时输出GROMACS进度
            process = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # 实时读取并输出进度（减少日志输出，每1000步输出一次）
            step_count = 0
            for line in process.stdout:
                if 'step' in line.lower() and 'will finish' not in line.lower():
                    step_count += 1
                    if step_count % 10 == 0:  # 每10次输出一次（约1000步）
                        print(f"[GROMACS-NPT] {line.strip()}")
            
            # 等待进程完成
            process.wait()
            
            mdrun_time = time.time() - mdrun_start
            print(f"[EQUILIB] ✅ NPT模拟完成，耗时: {mdrun_time:.1f}秒")
            print(f"[EQUILIB] NPT总耗时: {grompp_time + mdrun_time:.1f}秒")
            print(f"[EQUILIB] 平衡系统总耗时: {grompp_time + mdrun_time + grompp_time + mdrun_time:.1f}秒")
            
            if process.returncode != 0:
                return {
                    "success": False,
                    "step": "mdrun_npt",
                    "error": f"NPT模拟失败，返回码: {process.returncode}",
                    "status": "failed"
                }
            
            return {
                "success": True,
                "status": "equilibrated",
                "temperature": temperature,
                "pressure": pressure,
                "nvt_steps": nvt_steps,
                "npt_steps": npt_steps,
                "output_files": {
                    "gro": f"{work_dir}/{base_name}_npt.gro",
                    "xtc": f"{work_dir}/{base_name}_npt.xtc",
                    "cpt": f"{work_dir}/{base_name}_npt.cpt"
                }
            }
            
        except subprocess.TimeoutExpired:
            print(f"[EQUILIB] ❌ 平衡过程超时")
            print(f"[EQUILIB] Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "step": "equilibration",
                "error": "平衡过程超时",
                "status": "timeout"
            }
        except Exception as e:
            print(f"[EQUILIB] ❌ 平衡过程发生异常: {str(e)}")
            print(f"[EQUILIB] Traceback:\n{traceback.format_exc()}")
            return {
                "success": False,
                "step": "equilibration",
                "error": str(e),
                "status": "error"
            }
    
    def run_production_md(self, gro_file: str, top_file: str,
                         time_ns: float = 0.01, temperature: float = 310.0,
                         pressure: float = 1.0, dt: float = 0.002,
                         work_dir: str = None) -> Dict:
        """生产运行 - 主要的MD模拟
        
        Args:
            gro_file: 输入GRO文件
            top_file: 拓扑文件
            time_ns: 模拟时间(ns)，默认100ns
            temperature: 温度(K)，默认310K
            pressure: 压力(bar)，默认1.0 bar
            dt: 时间步长(ps)，默认0.002 ps
            work_dir: 工作目录
            
        Returns:
            生产运行结果
        """
        if work_dir is None:
            work_dir = os.path.dirname(gro_file)
        
        base_name = os.path.splitext(os.path.basename(gro_file))[0]
        
        nsteps = int(time_ns * 1000 / dt)
        
        # 生产运行mdp文件
        mdp_content = f"""; Production MD
integrator  = md
dt          = {dt}
nsteps      = {nsteps}
nstxout     = 500
nstvout     = 500
nstfout     = 500
nstenergy   = 500
nstlog      = 500
nstxout-compressed = 500
compressed-x-grps  = System
continuation = yes
cutoff-scheme = Verlet
ns_type      = grid
nstlist      = 10
rlist        = 1.0
coulombtype  = PME
rcoulomb     = 1.0
rvdw         = 1.0
pbc          = xyz
tcoupl       = V-rescale
tc-grps      = Protein Non-Protein
tau_t        = 0.1 0.1
ref_t        = {temperature} {temperature}
pcoupl       = Parrinello-Rahman
pcoupltype   = isotropic
tau_p        = 2.0
ref_p        = {pressure}
compressibility = 4.5e-5
constraints  = h-bonds
"""
        
        mdp_file = os.path.join(work_dir, "md.mdp")
        with open(mdp_file, 'w') as f:
            f.write(mdp_content)
        
        try:
            # 生成tpr文件
            print(f"[GROMACS] 生成MD TPR文件...")
            print(f"[GROMACS] TPR输入文件: {gro_file}")
            print(f"[GROMACS] TPR拓扑文件: {top_file}")
            print(f"[GROMACS] TPR参数文件: {mdp_file}")
            cmd = [
                self.gmx_path, "grompp",
                "-f", mdp_file,
                "-p", top_file,
                "-c", gro_file,
                "-o", f"{base_name}_md.tpr",
                "-maxwarn", "1"
            ]
            
            print(f"[GROMACS] 执行grompp命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode != 0:
                print(f"[GROMACS] ❌ TPR生成失败！")
                print(f"[GROMACS] 错误信息: {result.stderr}")
                return {
                    "success": False,
                    "step": "grompp_md",
                    "error": result.stderr,
                    "status": "failed"
                }
            
            print(f"[GROMACS] ✅ TPR文件生成成功: {base_name}_md.tpr")
            
            # 运行MD（使用GPU加速）
            import time
            mdrun_start = time.time()
            print(f"[GROMACS] 运行MD模拟...")
            print(f"[GROMACS] 使用路径: {self.gmx_path}")
            print(f"[GROMACS] GPU支持: {self.gpu_enabled}")
            print(f"[GROMACS] 预计步数: {nsteps:,}步")
            print(f"[GROMACS] 预计时间: {time_ns:.1f} ns")
            print(f"[GROMACS] 时间步长: {dt} ps")
            estimated_time_min = (nsteps * 0.001) / 60  # 预估时间（分钟）
            print(f"[GROMACS] ⏱️  预计耗时: {estimated_time_min:.1f} 分钟")
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
            ntomp = min(cpu_count - 1, 8)  # 使用最多8个线程，留1个核心给系统
            
            cmd = [
                self.gmx_path, "mdrun",
                "-deffnm", f"{base_name}_md",
                "-s", f"{base_name}_md.tpr",
                "-nb", "gpu",
                "-pme", "gpu",  # 添加PME GPU加速，提高GPU使用率
                "-gpu_id", "0",  # 明确指定使用GPU 0
                "-ntmpi", "1",
                "-ntomp", str(ntomp),
                "-v"
            ]
            
            print(f"[GROMACS] GPU加速参数: -nb gpu -pme gpu")
            print(f"[GROMACS] CPU线程数: {ntomp}")
            print(f"[GROMACS] 执行命令: {' '.join(cmd)}")

            
            # 实时输出GROMACS进度
            process = subprocess.Popen(
                cmd,
                cwd=work_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                universal_newlines=True
            )
            
            # 实时读取并输出进度
            print(f"[GROMACS-MD] 开始读取GROMACS输出...")
            step_count = 0
            last_progress_time = time.time()
            start_time = time.time()
            
            for line in process.stdout:
                line_stripped = line.strip()
                
                # 只显示包含完成信息的行
                if any(keyword in line.lower() for keyword in ['finished', 'done', 'completed']):
                    print(f"[GROMACS-MD] {line_stripped}")
                
                # 尝试解析步数进度
                if 'step' in line.lower() and 'will finish' not in line.lower():
                    step_count += 1
                    if step_count % 50 == 0:  # 每50次输出一次（减少输出）
                        import re
                        step_match = re.search(r'step\s+(\d+)', line, re.IGNORECASE)
                        if step_match:
                            current_step = int(step_match.group(1))
                            progress = (current_step / nsteps) * 100
                            elapsed_total = time.time() - start_time
                            
                            if elapsed_total > 0:
                                steps_per_second = current_step / elapsed_total
                                remaining_steps = nsteps - current_step
                                remaining_time_sec = remaining_steps / steps_per_second
                                remaining_time_min = remaining_time_sec / 60
                                
                                print(f"[GROMACS-MD] ⏱️  进度: {progress:.1f}% | 步数: {current_step:,}/{nsteps:,} | 已用时间: {elapsed_total:.1f}秒 | 剩余时间: {remaining_time_min:.1f}分钟")
            
            print(f"[GROMACS-MD] 输出读取完成")
            
            # 等待进程完成
            process.wait()
            
            mdrun_time = time.time() - mdrun_start
            print(f"[GROMACS] ✅ MD模拟完成，耗时: {mdrun_time:.1f}秒")
            
            if process.returncode != 0:
                return {
                    "success": False,
                    "step": "mdrun_md",
                    "error": f"MD模拟失败，返回码: {process.returncode}",
                    "status": "failed"
                }
            
            # 计算实际模拟时间
            actual_frames = self._count_frames(f"{work_dir}/{base_name}_md.xtc")
            actual_time = actual_frames * dt / 1000
            
            # 自动清理临时文件
            cleanup_result = self.auto_cleanup_after_production(work_dir, f"{base_name}_md")
            
            return {
                "success": True,
                "status": "completed",
                "simulation_time_ns": actual_time,
                "target_time_ns": time_ns,
                "nsteps": nsteps,
                "dt": dt,
                "temperature": temperature,
                "pressure": pressure,
                "frames": actual_frames,
                "output_files": {
                    "xtc": f"{work_dir}/{base_name}_md.xtc",
                    "gro": f"{work_dir}/{base_name}_md.gro",
                    "trr": f"{work_dir}/{base_name}_md.trr",
                    "edr": f"{work_dir}/{base_name}_md.edr",
                    "log": f"{work_dir}/{base_name}_md.log"
                },
                "cleanup": cleanup_result
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "step": "production_md",
                "error": "生产运行超时",
                "status": "timeout"
            }
        except Exception as e:
            return {
                "success": False,
                "step": "production_md",
                "error": str(e),
                "status": "error"
            }
    
    def analyze_trajectory(self, trajectory_file: str, top_file: str,
                       work_dir: str = None) -> Dict:
        """分析轨迹文件
        
        Args:
            trajectory_file: 轨迹文件(xtc/trr)
            top_file: 拓扑文件
            work_dir: 工作目录
            
        Returns:
            分析结果
        """
        if work_dir is None:
            work_dir = os.path.dirname(trajectory_file)
        
        base_name = os.path.splitext(os.path.basename(trajectory_file))[0]
        
        try:
            # 提取RMSD
            cmd = [
                self.gmx_path, "rms",
                "-s", top_file,
                "-f", trajectory_file,
                "-o", f"{base_name}_rmsd.xvg"
            ]
            
            subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                input="Protein\nProtein\n",
                timeout=600
            )
            
            # 提取RMSF
            cmd = [
                self.gmx_path, "rmsf",
                "-s", top_file,
                "-f", trajectory_file,
                "-o", f"{base_name}_rmsf.xvg"
            ]
            
            subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                input="Protein\n",
                timeout=600
            )
            
            # 提取回转半径
            cmd = [
                self.gmx_path, "gyrate",
                "-s", top_file,
                "-f", trajectory_file,
                "-o", f"{base_name}_gyrate.xvg"
            ]
            
            subprocess.run(
                cmd,
                cwd=work_dir,
                capture_output=True,
                text=True,
                input="Protein\n",
                timeout=600
            )
            
            # 解析XVG文件
            rmsd_data = self._parse_xvg(f"{work_dir}/{base_name}_rmsd.xvg")
            rmsf_data = self._parse_xvg(f"{work_dir}/{base_name}_rmsf.xvg")
            gyrate_data = self._parse_xvg(f"{work_dir}/{base_name}_gyrate.xvg")
            
            return {
                "success": True,
                "status": "analyzed",
                "trajectory_file": trajectory_file,
                "analysis": {
                    "rmsd": {
                        "mean": float(np.mean(rmsd_data)) if rmsd_data else 0,
                        "std": float(np.std(rmsd_data)) if rmsd_data else 0,
                        "max": float(np.max(rmsd_data)) if rmsd_data else 0,
                        "min": float(np.min(rmsd_data)) if rmsd_data else 0
                    },
                    "rmsf": {
                        "mean": float(np.mean(rmsf_data)) if rmsf_data else 0,
                        "std": float(np.std(rmsf_data)) if rmsf_data else 0,
                        "max_residue": int(np.argmax(rmsf_data)) + 1 if rmsf_data else 0
                    },
                    "gyrate": {
                        "mean": float(np.mean(gyrate_data)) if gyrate_data else 0,
                        "std": float(np.std(gyrate_data)) if gyrate_data else 0
                    }
                },
                "output_files": {
                    "rmsd": f"{work_dir}/{base_name}_rmsd.xvg",
                    "rmsf": f"{work_dir}/{base_name}_rmsf.xvg",
                    "gyrate": f"{work_dir}/{base_name}_gyrate.xvg"
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "step": "trajectory_analysis",
                "error": str(e),
                "status": "error"
            }
    
    def run_full_workflow(self, pdb_file: str, time_ns: float = 100.0,
                        temperature: float = 310.0, pressure: float = 1.0,
                        work_dir: str = None) -> Dict:
        """运行完整的MD工作流
        
        Args:
            pdb_file: 输入PDB文件
            time_ns: 模拟时间(ns)
            temperature: 温度(K)
            pressure: 压力(bar)
            work_dir: 工作目录
            
        Returns:
            完整工作流结果
        """
        print("=" * 60)
        print("[WORKFLOW] ========== 开始MD工作流 ==========")
        print(f"[WORKFLOW] PDB文件: {pdb_file}")
        print(f"[WORKFLOW] 模拟时间: {time_ns} ns")
        print(f"[WORKFLOW] 温度: {temperature} K")
        print(f"[WORKFLOW] 压力: {pressure} bar")
        print(f"[WORKFLOW] GPU支持: {self.gpu_enabled}")
        print(f"[WORKFLOW] GROMACS路径: {self.gmx_path}")
        print("=" * 60)
        
        job_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # 如果work_dir为None，设置为PDB文件所在目录
        if work_dir is None:
            work_dir = os.path.dirname(pdb_file)
            print(f"[WORKFLOW] 调试信息: work_dir设置为PDB文件所在目录: {work_dir}")
        
        workflow_start_time = time.time()
        workflow_results = {
            "job_id": job_id,
            "steps": [],
            "start_time": datetime.utcnow().isoformat(),
            "pdb_file": pdb_file,
            "parameters": {
                "time_ns": time_ns,
                "temperature": temperature,
                "pressure": pressure,
                "forcefield": "amber99sb-ildn"
            }
        }
        
        # 步骤1: 系统准备
        print("[WORKFLOW] 步骤 1/6: 系统准备...")
        result = self.prepare_system(pdb_file, work_dir=work_dir)
        workflow_results["steps"].append({"name": "system_preparation", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ❌ 系统准备失败: {result.get('error', 'Unknown error')}")
            return {**workflow_results, "status": "failed", "failed_at": "system_preparation"}
        
        print(f"[WORKFLOW] ✅ 系统准备完成")
        gro_file = result["output_files"]["gro"]
        top_file = result["output_files"]["top"]
        
        # 步骤2: 加离子
        print("[WORKFLOW] 步骤 2/6: 加离子...")
        result = self.add_ions(gro_file, top_file, work_dir=work_dir)
        workflow_results["steps"].append({"name": "add_ions", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ❌ 加离子失败: {result.get('error', 'Unknown error')}")
            return {**workflow_results, "status": "failed", "failed_at": "add_ions"}
        
        print(f"[WORKFLOW] ✅ 加离子完成")
        gro_file = result["output_files"]["gro"]
        
        # 步骤3: 能量最小化
        print("[WORKFLOW] 步骤 3/6: 能量最小化...")
        result = self.energy_minimize(gro_file, top_file, work_dir=work_dir)
        workflow_results["steps"].append({"name": "energy_minimization", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ❌ 能量最小化失败: {result.get('error', 'Unknown error')}")
            return {**workflow_results, "status": "failed", "failed_at": "energy_minimization"}
        
        print(f"[WORKFLOW] ✅ 能量最小化完成")
        gro_file = result["output_files"]["gro"]
        
        # 步骤4: 平衡
        print("[WORKFLOW] 步骤 4/6: 平衡系统...")
        result = self.equilibrate_system(gro_file, top_file, temperature, pressure, work_dir=work_dir)
        workflow_results["steps"].append({"name": "equilibration", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ❌ 平衡失败: {result.get('error', 'Unknown error')}")
            return {**workflow_results, "status": "failed", "failed_at": "equilibration"}
        
        print(f"[WORKFLOW] ✅ 平衡完成")
        gro_file = result["output_files"]["gro"]
        
        # 步骤5: 生产运行
        print("[WORKFLOW] 步骤 5/6: 生产MD模拟...")
        result = self.run_production_md(gro_file, top_file, time_ns, temperature, pressure, work_dir=work_dir)
        workflow_results["steps"].append({"name": "production_md", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ❌ 生产MD模拟失败: {result.get('error', 'Unknown error')}")
            return {**workflow_results, "status": "failed", "failed_at": "production_md"}
        
        print(f"[WORKFLOW] ✅ 生产MD模拟完成")
        
        # 步骤6: 轨迹分析
        print("[WORKFLOW] 步骤 6/6: 轨迹分析...")
        xtc_file = result["output_files"]["xtc"]
        result = self.analyze_trajectory(xtc_file, top_file, work_dir=work_dir)
        workflow_results["steps"].append({"name": "trajectory_analysis", "result": result})
        
        if not result["success"]:
            print(f"[WORKFLOW] ⚠️  轨迹分析失败: {result.get('error', 'Unknown error')}")
        else:
            print(f"[WORKFLOW] ✅ 轨迹分析完成")
        
        # 步骤7: 清理临时文件
        print("[WORKFLOW] 步骤 7/7: 清理临时文件...")
        print(f"[WORKFLOW] 调试信息: pdb_file={pdb_file}, work_dir={work_dir}")
        if pdb_file is None:
            print("[WORKFLOW] ⚠️  警告: pdb_file为None，跳过清理")
            cleanup_result = {"success": True, "deleted_count": 0, "freed_space_mb": 0}
        else:
            base_name = os.path.splitext(os.path.basename(pdb_file))[0]
            print(f"[WORKFLOW] 调试信息: base_name={base_name}")
            cleanup_result = self.cleanup_workflow(work_dir, base_name, cleanup_level="moderate")
        workflow_results["steps"].append({"name": "cleanup", "result": cleanup_result})
        
        # 检查清理是否成功
        if not cleanup_result.get("success", True):
            print(f"[WORKFLOW] ⚠️  清理临时文件失败: {cleanup_result.get('error', 'Unknown error')}")
        else:
            print(f"[WORKFLOW] ✅ 清理完成，删除了 {cleanup_result.get('deleted_count', 0)} 个文件，释放 {cleanup_result.get('freed_space_mb', 0)} MB 空间")
        
        workflow_results["end_time"] = datetime.utcnow().isoformat()
        workflow_results["status"] = "completed"
        workflow_results["success"] = True
        workflow_results["cleanup_summary"] = {
            "deleted_count": cleanup_result.get("deleted_count", 0),
            "freed_space_mb": cleanup_result.get("freed_space_mb", 0)
        }
        
        # 计算总耗时
        total_time = time.time() - workflow_start_time
        workflow_results["total_time"] = round(total_time, 2)
        
        print("=" * 60)
        print("[WORKFLOW] ========== 工作流完成 ==========")
        print(f"[WORKFLOW] 总耗时: {total_time:.2f}秒")
        print(f"[WORKFLOW] 状态: {workflow_results['status']}")
        print("=" * 60)
        
        self.simulation_jobs[job_id] = workflow_results
        
        return workflow_results
    
    def _extract_final_energy(self, log_file: str) -> Optional[float]:
        """从log文件中提取最终能量"""
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            
            for line in reversed(lines):
                if "Potential Energy" in line or "Energy" in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        try:
                            return float(part)
                        except ValueError:
                            continue
        except:
            pass
        
        return None
    
    def _count_frames(self, xtc_file: str) -> int:
        """计算轨迹文件中的帧数"""
        try:
            cmd = [
                self.gmx_path, "dump",
                "-s", xtc_file
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            return result.stdout.count("frame")
        except:
            return 0
    
    def _parse_xvg(self, xvg_file: str) -> List[float]:
        """解析XVG文件"""
        try:
            with open(xvg_file, 'r') as f:
                lines = f.readlines()
            
            data = []
            for line in lines:
                if line.startswith('#') or line.startswith('@'):
                    continue
                try:
                    values = [float(x) for x in line.split()]
                    if len(values) >= 2:
                        data.append(values[1])
                except ValueError:
                    continue
            
            return data
        except:
            return []
    
    def cleanup_temp_files(self, work_dir: str, base_name: str, 
                          keep_files: List[str] = None) -> Dict:
        """清理临时文件
        
        Args:
            work_dir: 工作目录
            base_name: 文件基础名称
            keep_files: 需要保留的文件模式列表（如['*.xtc', '*.tpr', '*.edr']）
            
        Returns:
            清理结果
        """
        if keep_files is None:
            keep_files = ['*.xtc', '*.trr', '*.tpr', '*.edr', '*.log', '*.gro']
        
        import glob
        import re
        
        try:
            # 需要保留的文件模式转换为正则表达式
            keep_patterns = []
            for pattern in keep_files:
                # 将通配符转换为正则表达式
                regex = pattern.replace('.', r'\.').replace('*', r'.*')
                keep_patterns.append(re.compile(regex))
            
            # 删除的临时文件类型（不包含轨迹文件）
            temp_extensions = ['.cpt', '#', '~']
            
            deleted_files = []
            kept_files = []
            
            # 扫描工作目录中的所有文件
            for file_path in glob.glob(os.path.join(work_dir, f'{base_name}*')):
                filename = os.path.basename(file_path)
                
                # 检查是否需要保留（保留模式优先级最高）
                should_keep = False
                for pattern in keep_patterns:
                    if pattern.match(filename):
                        should_keep = True
                        break
                
                # 检查是否是临时文件
                is_temp = any(filename.endswith(ext) for ext in temp_extensions)
                
                if should_keep:
                    # 匹配保留模式的文件，直接保留
                    kept_files.append(filename)
                    print(f"[CLEANUP] 保留轨迹/分析文件: {filename}")
                elif is_temp:
                    # 临时文件，删除
                    try:
                        os.remove(file_path)
                        deleted_files.append(filename)
                    except Exception as e:
                        continue
            
            # 计算释放的空间
            freed_space = sum(
                os.path.getsize(os.path.join(work_dir, f))
                for f in deleted_files
                if os.path.exists(os.path.join(work_dir, f))
            )
            
            return {
                "success": True,
                "deleted_files": deleted_files,
                "kept_files": kept_files,
                "deleted_count": len(deleted_files),
                "freed_space_bytes": freed_space,
                "freed_space_mb": round(freed_space / (1024 * 1024), 2),
                "work_dir": work_dir
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def cleanup_workflow(self, work_dir: str, base_name: str, 
                        cleanup_level: str = "moderate") -> Dict:
        """清理整个工作流的临时文件
        
        Args:
            work_dir: 工作目录
            base_name: 文件基础名称
            cleanup_level: 清理级别
                - 'minimal': 只删除.cpt文件
                - 'moderate': 删除.trr和.cpt文件（推荐）
                - 'aggressive': 删除所有临时文件，只保留核心结果文件
                
        Returns:
            清理结果
        """
        print(f"[CLEANUP] 调试信息: work_dir={work_dir}, base_name={base_name}, cleanup_level={cleanup_level}")
        
        if work_dir is None:
            print("[CLEANUP] ⚠️  警告: work_dir为None")
            return {"success": False, "error": "work_dir is None"}
        
        if base_name is None:
            print("[CLEANUP] ⚠️  警告: base_name为None")
            return {"success": False, "error": "base_name is None"}
        if cleanup_level == "minimal":
            keep_files = ['*']
            temp_only = ['.cpt']
        elif cleanup_level == "moderate":
            keep_files = ['*.xtc', '*.tpr', '*.edr', '*.log', '*.top', '*.gro']
            temp_only = ['.trr', '.cpt']
        else:  # aggressive
            keep_files = ['*.xtc', '*.tpr', '*.edr', '*.log']
            temp_only = ['.trr', '.cpt', '.gro', '.top', '#', '~']
        
        import glob
        import re
        
        try:
            deleted_files = []
            kept_files = []
            deleted_files_sizes = []  # 存储已删除文件的大小
            
            # 扫描工作目录中的所有文件
            for file_path in glob.glob(os.path.join(work_dir, f'{base_name}*')):
                filename = os.path.basename(file_path)
                
                # 检查是否是临时文件
                is_temp = any(filename.endswith(ext) for ext in temp_only)
                
                # 检查是否在保留列表中
                should_keep = False
                for pattern in keep_files:
                    regex = pattern.replace('.', r'\.').replace('*', r'.*')
                    if re.match(regex, filename):
                        should_keep = True
                        break
                
                if is_temp or not should_keep:
                    try:
                        file_size = os.path.getsize(file_path)  # 在删除前获取文件大小
                        os.remove(file_path)
                        deleted_files.append(filename)
                        deleted_files_sizes.append(file_size)
                        print(f"[CLEANUP] 已删除: {filename}")
                    except Exception as e:
                        print(f"[CLEANUP] 删除失败 {filename}: {str(e)}")
                        continue
                else:
                    kept_files.append(filename)
            
            # 额外清理：删除所有备份文件（以#或~开头，或以.1#等结尾）
            backup_patterns = [
                f'#{base_name}*',
                f'~{base_name}*',
                f'*.{base_name}.*',
            ]
            
            for pattern in backup_patterns:
                for file_path in glob.glob(os.path.join(work_dir, pattern)):
                    try:
                        filename = os.path.basename(file_path)
                        file_size = os.path.getsize(file_path)
                        os.remove(file_path)
                        deleted_files.append(filename)
                        deleted_files_sizes.append(file_size)
                        print(f"[CLEANUP] 已删除备份文件: {filename}")
                    except Exception as e:
                        print(f"[CLEANUP] 删除备份文件失败 {filename}: {str(e)}")
                        continue
            
            # 删除所有以.1#, .2#, .3#, .4#等结尾的文件
            for i in range(1, 10):
                for ext in ['.1#', '.2#', '.3#', '.4#', '.5#', '.6#', '.7#', '.8#', '.9#']:
                    for file_path in glob.glob(os.path.join(work_dir, f'*{ext}')):
                        try:
                            filename = os.path.basename(file_path)
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_files.append(filename)
                            deleted_files_sizes.append(file_size)
                            print(f"[CLEANUP] 已删除备份文件: {filename}")
                        except Exception as e:
                            print(f"[CLEANUP] 删除备份文件失败 {filename}: {str(e)}")
                            continue
            
            # 计算释放的空间（使用已存储的文件大小）
            freed_space = sum(deleted_files_sizes)
            
            return {
                "success": True,
                "cleanup_level": cleanup_level,
                "deleted_files": deleted_files,
                "kept_files": kept_files,
                "deleted_count": len(deleted_files),
                "freed_space_bytes": freed_space,
                "freed_space_mb": round(freed_space / (1024 * 1024), 2),
                "work_dir": work_dir
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "status": "error"
            }
    
    def auto_cleanup_after_production(self, work_dir: str, base_name: str) -> Dict:
        """生产运行后自动清理
        
        这个方法会在生产运行完成后自动调用，清理不需要的文件
        
        Args:
            work_dir: 工作目录
            base_name: 文件基础名称
            
        Returns:
            清理结果
        """
        return self.cleanup_workflow(work_dir, base_name, cleanup_level="moderate")

gromacs_runner = GromacsRunner()
