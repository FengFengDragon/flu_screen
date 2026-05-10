# AI辅助生物医药分子机制探索平台

基于 Flask 的流感病毒蛋白分子对接与虚拟筛选平台，集成 GROMACS 分子动力学模拟、机器学习结合位点预测、AutoDock Vina 分子对接、3D 分子可视化等功能。

## 功能模块

| 模块 | 路由 | 功能 |
|------|------|------|
| 首页 | `/` | 平台总览与导航 |
| 虚拟筛选 | `/virtual-screening` | 小分子预处理、Lipinski/ADMET/PAINS 过滤、Vina 分级对接、3D 可视化 |
| 机器学习预测 | `/ml-binding` | 随机森林/SVM/梯度提升预测结合位点 |
| 分子动力学模拟 | `/md-simulation` | GROMACS MD 模拟（系统准备、能量最小化、NVT/NPT 平衡、生产运行） |
| 轨迹分析 | `/trajectory-analysis` | RMSD/RMSF 分析、PCA/t-SNE 降维、构象聚类 |
| 轨迹可视化 | `/trajectory-visualization` | 3D 轨迹图、RMSD/RMSF 图表、构象热图、综合仪表板 |
| 工作流管理 | `/workflow` | 完整实验流程追踪 |
| 深度学习 | `/deep-learning` | GCN/GAT/Transformer 模型（需启用） |

## 环境要求

- Python 3.10+
- RDKit（用于分子处理和描述符计算）
- AutoDock Vina（用于分子对接，可选）
- GROMACS 2020+（用于 MD 模拟，可选）

## 安装步骤

### 1. 克隆项目

```bash
git clone https://github.com/FengFengDragon/flu_screen.git
cd flu_screen
```

### 2. 创建虚拟环境

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

**RDKit 安装**（如果 pip 安装失败）：

```bash
# 方法1：conda 安装（推荐）
conda install -c conda-forge rdkit

# 方法2：pip 安装
pip install rdkit
```

**AutoDock Vina 安装**（可选，用于分子对接）：

```bash
# Linux
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86
chmod +x vina_1.2.5_linux_x86
sudo mv vina_1.2.5_linux_x86 /usr/local/bin/vina

# Windows
# 从 https://github.com/ccsb-scripps/AutoDock-Vina/releases 下载 vina.exe
# 放到项目目录或添加到 PATH
```

### 4. 配置环境变量

复制并编辑 `.env` 文件：

```env
# 数据库配置（默认使用 SQLite，无需额外配置）
# 如需 MySQL，设置 DB_TYPE=mysql 并填写以下信息
DB_TYPE=sqlite
MYSQL_HOST=localhost
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=your_password
MYSQL_DATABASE=flu_screen

# Flask 配置
FLASK_APP=src/app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key

# 外部蛋白文件目录（可选，指向你存放 PDB 文件的目录）
EXTERNAL_PDB_DIR=/path/to/your/pdb/files

# 深度学习模块（设为1启用）
ENABLE_DEEP_LEARNING=0
```

### 5. 初始化数据库

```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### 6. 准备蛋白文件（可选）

将 PDB 文件放入以下目录：

```
data/pdb/          # 本地 PDB 文件
data/pdb_cache/    # 自动生成的缓存（AlphaFold CIF 转 PDB）
```

或在 `.env` 中配置 `EXTERNAL_PDB_DIR` 指向你的蛋白文件目录。支持 `.pdb` 文件和包含 CIF 模型的 `.zip` 文件（AlphaFold 预测结果）。

### 7. 训练 ML 模型（首次运行自动训练）

首次使用结合位点预测时，系统会自动训练模型并保存到 `data/models/`。也可以手动触发：

```python
from src.services.ml_algorithms import train_model_on_synthetic_data
train_model_on_synthetic_data('random_forest', 'data/models')
```

### 8. 启动项目

```bash
python run.py
```

浏览器访问 **http://localhost:5000**

## 项目结构

```
flu_screen/
├── run.py                          # 启动入口
├── requirements.txt                # Python 依赖
├── .env                            # 环境变量配置（不纳入版本控制）
├── .gitignore
│
├── src/
│   ├── app.py                      # Flask 应用工厂，Blueprint 注册
│   ├── config.py                   # 配置管理
│   ├── __init__.py                 # db 实例
│   │
│   ├── models/
│   │   └── experiment.py           # 实验数据模型
│   │
│   ├── routes/                     # API 路由（Blueprint）
│   │   ├── virtual_screening.py    # 虚拟筛选 + 3D 可视化 API
│   │   ├── ml_binding.py           # 机器学习结合位点预测 API
│   │   ├── molecular_dynamics.py   # GROMACS MD 模拟 API
│   │   ├── trajectory_analysis.py  # 轨迹分析 API
│   │   ├── trajectory_visualization.py  # 轨迹可视化 API
│   │   ├── deep_learning.py        # 深度学习 API
│   │   └── workflow.py             # 工作流管理 API
│   │
│   └── services/                   # 业务逻辑层
│       ├── vina_docking.py         # AutoDock Vina 分子对接
│       ├── ml_binding_predictor.py # ML 结合位点预测
│       ├── ml_algorithms.py        # ML 算法实现（RF/SVM/GB）
│       ├── residue_feature_extractor.py  # 残基特征提取
│       ├── ligand_preprocessor.py  # 配体预处理
│       ├── gromacs_runner.py       # GROMACS 运行器
│       ├── trajectory_analyzer.py  # 轨迹分析
│       ├── deep_learning_models.py # 深度学习模型
│       └── ...                     # 其他服务
│
├── templates/                      # 前端页面
│   ├── base.html                   # 基础模板
│   ├── index.html                  # 首页导航
│   ├── virtual_screening.html      # 虚拟筛选 + 3D 可视化页面
│   ├── ml_binding.html             # 机器学习预测页面
│   ├── md_simulation.html          # MD 模拟页面
│   ├── trajectory_analysis.html    # 轨迹分析页面
│   ├── trajectory_visualization.html  # 轨迹可视化页面
│   ├── deep_learning.html          # 深度学习页面
│   └── workflow.html               # 工作流页面
│
├── static/                         # 静态资源
│   ├── css/style.css
│   └── js/app.js
│
├── data/                           # 数据目录（不纳入版本控制）
│   ├── flu_screen.db               # SQLite 数据库
│   ├── pdb/                        # 本地 PDB 文件
│   ├── pdb_cache/                  # PDB 缓存（CIF→PDB 转换结果）
│   └── models/                     # 训练好的 ML 模型（.pkl）
│
└── migrations/                     # 数据库迁移文件
```

## 核心功能说明

### 虚拟筛选与分子对接

1. **小分子预处理**：支持 SMILES 输入，自动生成 3D 构象
2. **药物过滤**：Lipinski 五规则、ADMET 预测、PAINS 过滤
3. **分子对接**：AutoDock Vina 对接，支持自动结合位点检测
4. **3D 可视化**：基于 3Dmol.js 的蛋白质-配体交互可视化

### 机器学习结合位点预测

- 支持随机森林、SVM、梯度提升等算法
- 基于残基结构特征的预测（溶剂可及性、深度、邻居密度等）
- 自动训练与模型保存
- 可视化展示预测的结合位点

### 注意事项

- 程序在 **WSL (Windows Subsystem for Linux)** 中运行时，Windows 路径会自动转换为 `/mnt/x/...` 格式
- 首次使用分子对接前请确保 AutoDock Vina 已安装
- GROMACS MD 模拟功能需要单独安装 GROMACS
- 深度学习模块需设置 `ENABLE_DEEP_LEARNING=1` 并安装 PyTorch

## 技术栈

- **后端**：Flask 3.0、SQLAlchemy、Flask-Migrate
- **前端**：HTML5、CSS3、JavaScript、3Dmol.js
- **分子对接**：AutoDock Vina、RDKit
- **机器学习**：scikit-learn（随机森林、SVM、梯度提升）
- **MD 模拟**：GROMACS
- **数据库**：SQLite（默认）/ MySQL（可选）

## 许可证

本项目仅用于学术研究目的。
