import os
import requests
import json
from datetime import datetime

class TargetService:
    """靶点识别服务 - 流感病毒靶点数据库和PDB结构获取"""
    
    INFLUENZA_TARGETS = {
        "NP": {
            "name": "Nucleoprotein",
            "full_name": "Influenza A Nucleoprotein",
            "pdb_id": "4IDJ",
            "uniprot": "P03466",
            "description": "核蛋白，流感病毒高度保守靶点，不易产生耐药性",
            "binding_site": "核心口袋区域",
            "references": [
                {"year": 2019, "title": "Structure of influenza A NP", "journal": "PNAS"},
                {"year": 2021, "title": "NP inhibitors as broad-spectrum antivirals", "journal": "Nature Communications"}
            ]
        },
        "NA": {
            "name": "Neuraminidase",
            "full_name": "Influenza A Neuraminidase",
            "pdb_id": "4WE8",
            "uniprot": "P03452",
            "description": "神经氨酸酶，经典抗流感药物靶点（如奥司他韦）",
            "binding_site": "活性中心口袋",
            "references": [
                {"year": 2020, "title": "NA inhibitors structure-activity relationship", "journal": "J Med Chem"}
            ]
        },
        "HA": {
            "name": "Hemagglutinin",
            "full_name": "Influenza A Hemagglutinin",
            "pdb_id": "4WE8",
            "uniprot": "P03452",
            "description": "血凝素，病毒入侵细胞的关键蛋白",
            "binding_site": "受体结合位点",
            "references": []
        },
        "M2": {
            "name": "Matrix protein 2",
            "full_name": "Influenza A M2 ion channel",
            "pdb_id": "5DSE",
            "uniprot": "P06821",
            "description": "M2离子通道，金刚烷胺类药物靶点",
            "binding_site": "离子通道孔",
            "references": []
        },
        "PB2": {
            "name": "Polymerase PB2",
            "full_name": "RNA-dependent RNA polymerase PB2 subunit",
            "pdb_id": "5E7T",
            "uniprot": "P02731",
            "description": "RNA聚合酶亚基，宿主特异性靶点",
            "binding_site": "帽结合结构域",
            "references": [
                {"year": 2023, "title": "PB2 inhibitors block viral transcription", "journal": "Science Advances"}
            ]
        }
    }
    
    def __init__(self, pdb_dir=None):
        self.pdb_dir = pdb_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'pdb')
        os.makedirs(self.pdb_dir, exist_ok=True)
    
    def get_target_info(self, target_code):
        """获取靶点信息"""
        return self.INFLUENZA_TARGETS.get(target_code.upper())
    
    def list_targets(self):
        """列出所有可用靶点"""
        return [
            {
                "code": code,
                "name": info["name"],
                "description": info["description"],
                "pdb_id": info["pdb_id"]
            }
            for code, info in self.INFLUENZA_TARGETS.items()
        ]
    
    def download_pdb_structure(self, target_code, output_dir=None):
        """从RCSB PDB下载靶点3D结构"""
        target = self.get_target_info(target_code)
        if not target:
            return {"success": False, "error": "未知的靶点代码"}
        
        pdb_id = target["pdb_id"]
        save_dir = output_dir or self.pdb_dir
        os.makedirs(save_dir, exist_ok=True)
        
        pdb_file = os.path.join(save_dir, f"{target_code}_{pdb_id}.pdb")
        
        if os.path.exists(pdb_file):
            return {"success": True, "file": pdb_file, "cached": True}
        
        try:
            url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            with open(pdb_file, 'w') as f:
                f.write(response.text)
            
            return {
                "success": True,
                "file": pdb_file,
                "pdb_id": pdb_id,
                "target": target["name"],
                "description": target["description"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def search_literature_targets(self, disease="流感病毒"):
        """使用LLM检索特定疾病的潜在靶点"""
        from src.services.llm_client import llm_client
        
        prompt = f"""请检索关于{disease}的潜在治疗靶点信息。

请搜索以下信息：
1. 病毒生命周期中的关键蛋白
2. 已报道的药物靶点
3. 2023-2025年最新的前沿靶点研究
4. 靶点的保守性和耐药性考虑

请以JSON格式返回结果，格式如下：
{{
    "targets": [
        {{
            "name": "靶点名称",
            "uniprot": "UniProt ID",
            "function": "功能描述",
            "drugability": "成药性评估",
            "references": ["参考文献"]
        }}
    ],
    "recommendations": "推荐靶点及理由"
}}

只返回JSON，不要其他内容。"""
        
        try:
            result = llm_client.chat(prompt)
            import re
            json_match = re.search(r'\{[\s\S]*\}', result)
            if json_match:
                return json.loads(json_match.group())
        except Exception as e:
            print(f"文献检索失败: {e}")
        
        return {"targets": [], "recommendations": "请手动选择靶点"}
    
    def analyze_binding_sites(self, pdb_file):
        """分析PDB文件中的结合位点"""
        if not os.path.exists(pdb_file):
            return {"success": False, "error": "PDB文件不存在"}
        
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()
        
        residues = []
        for line in pdb_content.split('\n'):
            if line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    residue = line[17:20].strip()
                    chain = line[21]
                    residue_num = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip()
                    
                    if element == 'CA' and residue == 'HIS':
                        residues.append({
                            "residue": residue,
                            "chain": chain,
                            "position": residue_num,
                            "coords": (x, y, z)
                        })
                except:
                    continue
        
        return {
            "success": True,
            "binding_sites": residues[:10],
            "pdb_file": pdb_file
        }

target_service = TargetService()
