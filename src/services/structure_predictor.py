import os
import tempfile
from pathlib import Path
from typing import Tuple, Optional, Dict
import subprocess


class StructurePredictor:
    """从FASTA序列预测3D结构的服务"""
    
    def __init__(self, alphafold_path=None):
        self.alphafold_path = alphafold_path
        self.temp_dir = tempfile.mkdtemp()
    
    def fasta_to_structure(self, fasta_file: str, method: str = 'alphafold') -> Tuple[bool, str, Optional[str]]:
        """
        将FASTA序列转换为3D结构文件
        
        Args:
            fasta_file: FASTA序列文件路径
            method: 预测方法 ('alphafold', 'pdb_search', 'simple')
        
        Returns:
            (success, message, structure_file): 是否成功、消息、结构文件路径
        """
        fasta_path = Path(fasta_file)
        
        if not fasta_path.exists():
            return False, f"FASTA文件不存在: {fasta_file}", None
        
        try:
            if method == 'alphafold':
                return self._predict_with_alphafold(fasta_file)
            elif method == 'pdb_search':
                return self._search_pdb_database(fasta_file)
            elif method == 'simple':
                return self._simple_prediction(fasta_file)
            else:
                return False, f"不支持的预测方法: {method}", None
        
        except Exception as e:
            return False, f"结构预测失败: {str(e)}", None
    
    def _predict_with_alphafold(self, fasta_file: str) -> Tuple[bool, str, Optional[str]]:
        """
        使用AlphaFold预测3D结构
        
        Args:
            fasta_file: FASTA文件路径
        
        Returns:
            (success, message, pdb_file): 是否成功、消息、PDB文件路径
        
        Note:
            需要安装AlphaFold
            推荐使用ColabFold版本（更容易部署）
        """
        try:
            fasta_path = Path(fasta_file)
            output_dir = os.path.join(self.temp_dir, 'alphafold_output')
            os.makedirs(output_dir, exist_ok=True)
            
            output_pdb = os.path.join(output_dir, fasta_path.stem + '_predicted.pdb')
            
            if self.alphafold_path:
                cmd = [
                    self.alphafold_path,
                    '--fasta_paths', fasta_file,
                    '--output_dir', output_dir
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
                
                if result.returncode == 0:
                    if os.path.exists(output_pdb):
                        return True, f"AlphaFold预测成功: {output_pdb}", output_pdb
                    else:
                        return False, "AlphaFold运行完成但未找到输出文件", None
                else:
                    return False, f"AlphaFold预测失败: {result.stderr}", None
            
            else:
                return False, "未配置AlphaFold路径。推荐使用PDB数据库搜索或在线AlphaFold预测", None
        
        except subprocess.TimeoutExpired:
            return False, "AlphaFold预测超时（超过1小时）", None
        except FileNotFoundError:
            return False, "未找到AlphaFold，请安装或使用PDB搜索功能", None
        except Exception as e:
            return False, f"AlphaFold预测出错: {str(e)}", None
    
    def _search_pdb_database(self, fasta_file: str) -> Tuple[bool, str, Optional[str]]:
        """
        在PDB数据库中搜索相似序列
        
        Args:
            fasta_file: FASTA文件路径
        
        Returns:
            (success, message, pdb_file): 是否成功、消息、PDB文件路径
        """
        try:
            from Bio import SeqIO
            from Bio.Blast import NCBIWWW
            from Bio import ExPASy
            
            fasta_path = Path(fasta_file)
            sequences = list(SeqIO.parse(fasta_file, 'fasta'))
            
            if not sequences:
                return False, "FASTA文件中没有有效序列", None
            
            sequence = sequences[0]
            seq_str = str(sequence.seq)
            
            if len(seq_str) < 10:
                return False, "序列太短，无法进行有意义搜索", None
            
            output_dir = os.path.join(self.temp_dir, 'pdb_search')
            os.makedirs(output_dir, exist_ok=True)
            
            downloaded_file = os.path.join(output_dir, fasta_path.stem + '_from_pdb.pdb')
            
            try:
                blast_result = NCBIWWW.qblast(
                    program='blastp',
                    database='pdb',
                    sequence=seq_str,
                    hitlist_size=5,
                    format_type='XML'
                )
                
                if blast_result.alignments and len(blast_result.alignments) > 0:
                    best_hit = blast_result.alignments[0]
                    pdb_id = best_hit.hit_id.split('|')[0][:4]
                    
                    try:
                        pdb_handle = ExPASy.get_sifts(pdb_id)
                        with open(downloaded_file, 'w') as f:
                            f.write(pdb_handle.read())
                        
                        return True, f"从PDB数据库找到相似结构: {pdb_id}", downloaded_file
                    
                    except Exception as e:
                        return False, f"PDB下载失败: {str(e)}", None
                else:
                    return False, "在PDB数据库中未找到相似序列", None
            
            except ImportError:
                return False, "需要安装Biopython: pip install biopython", None
            except Exception as e:
                return False, f"PDB搜索失败: {str(e)}", None
        
        except Exception as e:
            return False, f"搜索过程出错: {str(e)}", None
    
    def _simple_prediction(self, fasta_file: str) -> Tuple[bool, str, Optional[str]]:
        """
        简单的序列到结构预测（使用简化的折叠算法）
        
        Args:
            fasta_file: FASTA文件路径
        
        Returns:
            (success, message, pdb_file): 是否成功、消息、PDB文件路径
        
        Note:
            这是一个简化的实现，仅用于演示
            实际应用建议使用AlphaFold或从PDB下载
        """
        try:
            from Bio import SeqIO
            from Bio.PDB import PDBIO, Structure, Model, Chain
            from Bio.PDB.Polypeptide import PPBuilder
            
            fasta_path = Path(fasta_file)
            sequences = list(SeqIO.parse(fasta_file, 'fasta'))
            
            if not sequences:
                return False, "FASTA文件中没有有效序列", None
            
            sequence = sequences[0]
            seq_str = str(sequence.seq)
            
            if len(seq_str) < 10:
                return False, "序列太短，无法构建结构", None
            
            output_dir = os.path.join(self.temp_dir, 'simple_prediction')
            os.makedirs(output_dir, exist_ok=True)
            
            output_pdb = os.path.join(output_dir, fasta_path.stem + '_simple.pdb')
            
            ppb = PPBuilder()
            pp = ppb.build_peptides(sequence.seq)
            
            structure = Structure.Structure(id=sequence.id)
            model = Model.Model(id=0)
            chain = Chain.Chain(id='A')
            
            for i, polypeptide in enumerate(pp):
                for j, residue in enumerate(polypeptide):
                    chain.add(residue)
            
            model.add(chain)
            structure.add(model)
            
            from Bio.PDB import PDBIO
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_pdb)
            
            if os.path.exists(output_pdb):
                return True, f"简单预测完成（质量较低，建议使用AlphaFold）: {output_pdb}", output_pdb
            else:
                return False, "简单预测失败，未生成PDB文件", None
        
        except ImportError:
            return False, "需要安装Biopython: pip install biopython", None
        except Exception as e:
            return False, f"简单预测失败: {str(e)}", None
    
    def get_sequence_info(self, fasta_file: str) -> Dict:
        """
        分析FASTA序列信息
        
        Args:
            fasta_file: FASTA文件路径
        
        Returns:
            dict: 序列信息
        """
        try:
            from Bio import SeqIO
            
            sequences = list(SeqIO.parse(fasta_file, 'fasta'))
            
            if not sequences:
                return {"error": "无效的FASTA文件"}
            
            sequence = sequences[0]
            seq_str = str(sequence.seq)
            
            amino_acids = {
                'A': 0, 'R': 0, 'N': 0, 'D': 0, 'C': 0, 'Q': 0,
                'E': 0, 'G': 0, 'H': 0, 'I': 0, 'L': 0, 'K': 0,
                'M': 0, 'F': 0, 'P': 0, 'S': 0, 'T': 0, 'W': 0,
                'Y': 0, 'V': 0
            }
            
            for aa in seq_str:
                if aa in amino_acids:
                    amino_acids[aa] += 1
            
            return {
                "sequence_id": sequence.id,
                "description": sequence.description,
                "length": len(seq_str),
                "molecular_weight": self._calculate_mw(seq_str),
                "composition": amino_acids,
                "is_valid_protein": all(aa in amino_acids for aa in seq_str),
                "prediction_recommended": len(seq_str) > 50
            }
        
        except ImportError:
            return {"error": "需要安装Biopython: pip install biopython"}
        except Exception as e:
            return {"error": f"序列分析失败: {str(e)}"}
    
    def _calculate_mw(self, sequence: str) -> float:
        """计算蛋白质分子量"""
        aa_weights = {
            'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.16,
            'Q': 146.15, 'E': 147.13, 'G': 75.07, 'H': 155.16,
            'I': 131.17, 'L': 131.17, 'K': 146.19, 'M': 149.21,
            'F': 165.19, 'P': 115.13, 'S': 105.09, 'T': 119.12,
            'W': 204.23, 'Y': 181.19, 'V': 117.15
        }
        
        mw = 0.0
        for aa in sequence:
            if aa in aa_weights:
                mw += aa_weights[aa]
        
        return round(mw, 2)
    
    def get_prediction_methods(self) -> list:
        """返回可用的预测方法"""
        return [
            {
                "id": "alphafold",
                "name": "AlphaFold",
                "description": "DeepMind的AlphaFold预测，准确性最高",
                "requirements": "需要安装AlphaFold或使用在线服务",
                "time_estimate": "10-60分钟",
                "quality": "高"
            },
            {
                "id": "pdb_search",
                "name": "PDB数据库搜索",
                "description": "在PDB数据库中搜索相似序列的结构",
                "requirements": "需要安装Biopython",
                "time_estimate": "1-5分钟",
                "quality": "中高"
            },
            {
                "id": "simple",
                "name": "简单预测",
                "description": "使用简化的折叠算法预测（仅用于演示）",
                "requirements": "需要安装Biopython",
                "time_estimate": "秒级",
                "quality": "低"
            }
        ]
    
    def cleanup(self):
        """清理临时文件"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass


structure_predictor = StructurePredictor()
