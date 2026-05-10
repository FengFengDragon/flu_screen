import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, Optional
from src.services.structure_predictor import structure_predictor


class FormatConverter:
    """文件格式转换服务 - 支持多种输入格式转换为GROMACS兼容格式"""
    
    SUPPORTED_INPUT_FORMATS = ['.pdb', '.gro', '.xyz', '.mol2', '.sdf', '.fasta']
    TARGET_FORMAT = '.gro'
    
    def __init__(self, gromacs_path='gmx'):
        self.gromacs_path = gromacs_path
        self.temp_dir = tempfile.mkdtemp()
    
    def convert_to_gro(self, input_file: str, output_file: Optional[str] = None) -> Tuple[bool, str]:
        """
        将支持的各种格式转换为GRO格式
        
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径（可选，默认为同目录下）
        
        Returns:
            (success, message): 是否成功及结果消息
        """
        input_path = Path(input_file)
        
        if not input_path.exists():
            return False, f"输入文件不存在: {input_file}"
        
        input_ext = input_path.suffix.lower()
        
        if input_ext == self.TARGET_FORMAT:
            return True, "文件已经是GRO格式，无需转换"
        
        if input_ext not in self.SUPPORTED_INPUT_FORMATS:
            return False, f"不支持的文件格式: {input_ext}。支持的格式: {self.SUPPORTED_INPUT_FORMATS}"
        
        if output_file is None:
            output_file = input_path.with_suffix(self.TARGET_FORMAT)
        
        try:
            if input_ext == '.pdb':
                result = self._convert_pdb_to_gro(input_file, output_file)
            elif input_ext == '.xyz':
                result = self._convert_xyz_to_gro(input_file, output_file)
            elif input_ext in ['.mol2', '.sdf']:
                result = self._convert_molfile_to_gro(input_file, output_file)
            elif input_ext == '.fasta':
                result = self._convert_fasta_to_gro(input_file, output_file)
            else:
                result = (False, f"暂不支持从 {input_ext} 转换")
            
            return result
        
        except Exception as e:
            return False, f"转换失败: {str(e)}"
    
    def _convert_pdb_to_gro(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """使用GROMACS将PDB转换为GRO"""
        try:
            cmd = [
                self.gromacs_path, 'editconf',
                '-f', input_file,
                '-o', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return True, f"成功将PDB转换为GRO: {output_file}"
        
        except subprocess.CalledProcessError as e:
            return False, f"GROMACS转换失败: {e.stderr}"
        except FileNotFoundError:
            return False, "未找到GROMACS (gmx)，请确保已安装并添加到PATH"
    
    def _convert_xyz_to_gro(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """使用GROMACS将XYZ转换为GRO"""
        try:
            cmd = [
                self.gromacs_path, 'editconf',
                '-f', input_file,
                '-o', output_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            return True, f"成功将XYZ转换为GRO: {output_file}"
        
        except subprocess.CalledProcessError as e:
            return False, f"GROMACS转换失败: {e.stderr}"
    
    def _convert_molfile_to_gro(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """
        将MOL2/SDF转换为GRO
        先使用OpenBabel转换为PDB，再用GROMACS转换为GRO
        """
        try:
            intermediate_pdb = os.path.join(self.temp_dir, 'intermediate.pdb')
            
            babel_cmd = ['obabel', '-i', Path(input_file).suffix, input_file, 
                        '-o', 'pdb', '-O', intermediate_pdb]
            
            try:
                subprocess.run(babel_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError:
                return False, "未找到OpenBabel，请安装: pip install openbabel"
            
            return self._convert_pdb_to_gro(intermediate_pdb, output_file)
        
        except Exception as e:
            return False, f"MOL文件转换失败: {str(e)}"
    
    def validate_file(self, file_path: str) -> dict:
        """
        验证文件格式和内容
        
        Returns:
            dict: 包含验证结果
        """
        path = Path(file_path)
        
        result = {
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "extension": path.suffix.lower(),
            "is_supported": path.suffix.lower() in self.SUPPORTED_INPUT_FORMATS,
            "can_convert": True
        }
        
        if not result["is_supported"]:
            result["message"] = f"不支持的文件格式"
            result["can_convert"] = False
        elif path.suffix.lower() == self.TARGET_FORMAT:
            result["message"] = "GROMACS原生格式，无需转换"
            result["can_convert"] = False
        else:
            result["message"] = f"支持该格式，可转换为{self.TARGET_FORMAT}"
        
        return result
    
    def get_supported_formats(self) -> list:
        """返回支持的输入格式列表"""
        return self.SUPPORTED_INPUT_FORMATS
    
    def _convert_fasta_to_gro(self, input_file: str, output_file: str) -> Tuple[bool, str]:
        """
        将FASTA序列转换为GRO格式
        
        Args:
            input_file: FASTA文件路径
            output_file: 输出GRO文件路径
        
        Returns:
            (success, message): 是否成功及结果消息
        """
        try:
            success, message, pdb_file = structure_predictor.fasta_to_structure(input_file, method='pdb_search')
            
            if not success or not pdb_file:
                return False, f"FASTA到结构转换失败: {message}"
            
            intermediate_pdb = pdb_file
            
            if output_file is None:
                output_file = Path(input_file).with_suffix(self.TARGET_FORMAT)
            
            return self._convert_pdb_to_gro(intermediate_pdb, str(output_file))
        
        except Exception as e:
            return False, f"FASTA转换过程出错: {str(e)}"
    
    def get_fasta_info(self, fasta_file: str) -> dict:
        """
        获取FASTA文件信息
        
        Args:
            fasta_file: FASTA文件路径
        
        Returns:
            dict: 序列信息
        """
        return structure_predictor.get_sequence_info(fasta_file)
    
    def get_prediction_methods(self) -> list:
        """返回可用的结构预测方法"""
        return structure_predictor.get_prediction_methods()
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


format_converter = FormatConverter()
