# 分子动力学模拟与AI分析平台

基于毕设要求的分子动力学模拟和AI分析平台，集成GROMACS MD模拟、轨迹分析、深度学习模型、多模态数据融合和轨迹可视化增强功能。

## 项目概述

本平台为研究蛋白质结构和功能提供了完整的分析工具链，从分子动力学模拟到深度学习预测，支持多种分析方法和可视化展示。

## 功能特性

### 1. GROMACS分子动力学模拟
- 系统准备（蛋白质结构处理、溶剂化、离子添加）
- 能量最小化（最陡下降法）
- NVT和NPT平衡
- 生产运行
- 完整的MD工作流程管理

### 2. 轨迹分析与PCA/t-SNE降维
- RMSD（均方根偏差）分析
- RMSF（均方根涨落）分析
- PCA（主成分分析）降维
- t-SNE（t分布随机邻域嵌入）降维
- K-means聚类
- 层次聚类
- 智能帧提取（基于聚类或能量）
- 构象分布分析

### 3. 深度学习模型
- GCN（图卷积网络）- 关键残基识别
- GAT（图注意力网络）- 关键残基识别
- Transformer模型 - 蛋白质功能预测
- 模型训练与评估
- 预测与解释

### 4. 多模态数据融合
- PDB数据库管理
- 实验数据管理（IC50、Ki、活性类型）
- 特征提取（PDB特征、实验数据特征）
- 早期融合（特征拼接）
- 晚期融合（加权平均）
- 混合融合（结合早期和晚期）
- 多头注意力融合模型
- 融合预测（单次和批量）

### 5. 轨迹可视化增强
- 3D轨迹图
- 轨迹动画（GIF/MP4/AVI）
- RMSD图表（曲线和分布）
- RMSF图表（残基柔性分析）
- 构象分布热图（PCA和t-SNE）
- 相互作用网络图
- 综合仪表板（HTML和图片）
- 批量可视化

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  Web界面层                          │
│  (Flask Templates + JavaScript + CSS)              │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  API路由层                         │
│  (Flask Blueprints)                               │
├────────────────────────────────────────────────────┤
│ • molecular_dynamics    • trajectory_analysis      │
│ • deep_learning        • multimodal_fusion        │
│ • trajectory_visualization                       │
└────────────────────┬────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────┐
│                  服务层                           │
│  (Business Logic Services)                        │
├────────────────────────────────────────────────────┤
│ • gromacs_runner         • trajectory_analyzer   │
│ • deep_learning_models   • multimodal_fusion     │
│ • trajectory_visualizer                         │
└────────────────────────────────────────────────────┘
```

## 文件结构

### 核心应用文件
```
flu-screen/
├── src/
│   ├── app.py                           # Flask应用主文件
│   ├── config.py                        # 配置管理
│   └── models.py                        # 数据库模型
```

### 服务层
```
src/services/
├── gromacs_runner.py                    # GROMACS MD模拟服务
│   ├── prepare_system()                 # 系统准备
│   ├── minimize_energy()                # 能量最小化
│   ├── equilibrate_nvt()                # NVT平衡
│   ├── equilibrate_npt()                # NPT平衡
│   ├── run_production()                 # 生产运行
│   └── run_full_md_workflow()          # 完整MD工作流
│
├── trajectory_analyzer.py                # 轨迹分析服务
│   ├── analyze_trajectory()             # 轨迹分析
│   ├── calculate_rmsd()                # RMSD计算
│   ├── calculate_rmsf()                # RMSF计算
│   ├── perform_pca()                   # PCA降维
│   ├── perform_tsne()                  # t-SNE降维
│   ├── cluster_conformations()          # 构象聚类
│   └── extract_key_frames()            # 关键帧提取
│
├── deep_learning_models.py              # 深度学习模型服务
│   ├── GCNModel                       # 图卷积网络
│   ├── GATModel                       # 图注意力网络
│   ├── TransformerModel                # Transformer模型
│   ├── train_gcn_model()               # 训练GCN模型
│   ├── train_gat_model()               # 训练GAT模型
│   ├── train_transformer_model()        # 训练Transformer模型
│   ├── predict_key_residues()          # 预测关键残基
│   └── predict_function()             # 预测蛋白质功能
│
├── multimodal_fusion.py               # 多模态融合服务
│   ├── MultimodalFusionModel         # 融合神经网络
│   ├── add_pdb_entry()               # 添加PDB条目
│   ├── add_experimental_entry()       # 添加实验数据
│   ├── extract_pdb_features()        # 提取PDB特征
│   ├── extract_exp_features()        # 提取实验特征
│   ├── fuse_features_early()         # 早期融合
│   ├── fuse_features_late()          # 晚期融合
│   ├── fuse_features_hybrid()        # 混合融合
│   ├── train_fusion_model()          # 训练融合模型
│   └── predict_fusion()             # 融合预测
│
└── trajectory_visualizer.py           # 轨迹可视化服务
    ├── create_3d_trajectory_plot()   # 3D轨迹图
    ├── create_trajectory_animation()  # 轨迹动画
    ├── create_rmsd_plot()           # RMSD图
    ├── create_rmsf_plot()           # RMSF图
    ├── create_conformation_heatmap() # 构象热图
    ├── create_interaction_network()  # 相互作用网络
    ├── create_comprehensive_dashboard() # 综合仪表板
    └── batch_visualize()           # 批量可视化
```

### 路由层
```
src/routes/
├── molecular_dynamics.py             # MD模拟API路由
│   ├── GET  /api/md/status          # 系统状态
│   ├── POST /api/md/prepare         # 准备系统
│   ├── POST /api/md/minimize        # 能量最小化
│   ├── POST /api/md/equilibrate-nvt # NVT平衡
│   ├── POST /api/md/equilibrate-npt # NPT平衡
│   ├── POST /api/md/production     # 生产运行
│   └── POST /api/md/full-workflow  # 完整工作流
│
├── trajectory_analysis.py            # 轨迹分析API路由
│   ├── GET  /api/trajectory/status          # 系统状态
│   ├── POST /api/trajectory/analyze         # 轨迹分析
│   ├── POST /api/trajectory/rmsd           # RMSD计算
│   ├── POST /api/trajectory/rmsf           # RMSF计算
│   ├── POST /api/trajectory/pca            # PCA降维
│   ├── POST /api/trajectory/tsne           # t-SNE降维
│   ├── POST /api/trajectory/cluster        # 构象聚类
│   └── POST /api/trajectory/extract-frames # 关键帧提取
│
├── deep_learning.py                 # 深度学习API路由
│   ├── GET  /api/deep-learning/status       # 系统状态
│   ├── POST /api/deep-learning/train-gcn    # 训练GCN
│   ├── POST /api/deep-learning/train-gat    # 训练GAT
│   ├── POST /api/deep-learning/train-transformer # 训练Transformer
│   ├── POST /api/deep-learning/predict-key-residues # 预测关键残基
│   └── POST /api/deep-learning/predict-function # 预测功能
│
├── multimodal_fusion.py            # 多模态融合API路由
│   ├── GET  /api/multimodal/status              # 系统状态
│   ├── GET  /api/multimodal/pdb-database       # PDB数据库
│   ├── POST /api/multimodal/pdb-database       # 添加PDB条目
│   ├── DEL  /api/multimodal/pdb-database/<id>  # 删除PDB条目
│   ├── GET  /api/multimodal/experimental-data   # 实验数据
│   ├── POST /api/multimodal/experimental-data   # 添加实验数据
│   ├── DEL  /api/multimodal/experimental-data/<id> # 删除实验数据
│   ├── POST /api/multimodal/extract-features  # 特征提取
│   ├── POST /api/multimodal/fuse-features     # 特征融合
│   ├── POST /api/multimodal/train-fusion      # 训练融合模型
│   ├── POST /api/multimodal/predict           # 融合预测
│   ├── POST /api/multimodal/batch-predict     # 批量预测
│   ├── GET  /api/multimodal/statistics       # 统计信息
│   └── POST /api/multimodal/export-data      # 导出数据
│
└── trajectory_visualization.py       # 轨迹可视化API路由
    ├── GET  /api/trajectory-visualization/status            # 系统状态
    ├── POST /api/trajectory-visualization/create-3d-plot   # 3D轨迹图
    ├── POST /api/trajectory-visualization/create-animation  # 轨迹动画
    ├── POST /api/trajectory-visualization/create-rmsd-plot # RMSD图
    ├── POST /api/trajectory-visualization/create-rmsf-plot # RMSF图
    ├── POST /api/trajectory-visualization/create-conformation-heatmap # 构象热图
    ├── POST /api/trajectory-visualization/create-interaction-network # 相互作用网络
    ├── POST /api/trajectory-visualization/create-dashboard # 综合仪表板
    ├── GET  /api/trajectory-visualization/download/<filename> # 下载文件
    ├── GET  /api/trajectory-visualization/list-visualizations # 文件列表
    ├── DEL  /api/trajectory-visualization/clear-visualizations # 清除文件
    └── POST /api/trajectory-visualization/batch-visualize # 批量可视化
```

### Web界面
```
templates/
├── base.html                       # 基础模板
├── index.html                      # 首页
├── md_simulation.html               # MD模拟界面
│   ├── 系统准备
│   ├── 能量最小化
│   ├── NVT/NPT平衡
│   ├── 生产运行
│   └── 完整工作流
│
├── trajectory_analysis.html         # 轨迹分析界面
│   ├── 轨迹上传
│   ├── RMSD分析
│   ├── RMSF分析
│   ├── PCA降维
│   ├── t-SNE降维
│   ├── 构象聚类
│   └── 关键帧提取
│
├── deep_learning.html               # 深度学习界面
│   ├── GCN训练
│   ├── GAT训练
│   ├── Transformer训练
│   ├── 关键残基预测
│   └── 功能预测
│
├── multimodal_fusion.html          # 多模态融合界面
│   ├── 概览（系统状态、统计信息）
│   ├── PDB数据库管理
│   ├── 实验数据管理
│   ├── 特征融合（早期/晚期/混合）
│   ├── 模型训练
│   └── 预测分析
│
└── trajectory_visualization.html    # 轨迹可视化界面
    ├── 概览（系统状态、文件列表）
    ├── 3D轨迹
    ├── 轨迹动画
    ├── RMSD图
    ├── RMSF图
    ├── 构象热图
    ├── 相互作用网络
    └── 综合仪表板
```

### 其他文件
```
flu-screen/
├── migrations/                     # 数据库迁移文件
├── static/                        # 静态资源（CSS、JS、图片）
├── data/                          # 数据目录
│   ├── gromacs/                   # GROMACS工作目录
│   ├── trajectories/               # 轨迹文件
│   ├── models/                    # 训练好的模型
│   ├── pdb_database/              # PDB数据库
│   ├── experimental_data/         # 实验数据
│   └── visualizations/            # 可视化输出
├── requirements.txt                # Python依赖
├── run.py                        # 启动脚本
└── README.md                     # 项目文档
```

## 安装依赖

### 基础依赖
```bash
pip install -r requirements.txt
```

### 主要依赖包
- Flask 3.0.0
- Flask-SQLAlchemy 3.1.1
- Flask-Migrate 4.0.5
- numpy
- pandas

### 可选依赖（增强功能）

#### GROMACS MD模拟
需要安装GROMACS（推荐版本2020+）
```bash
# Windows
# 下载安装GROMACS并添加到PATH

# Linux/Mac
conda install -c conda-forge gromacs
```

#### 轨迹分析
```bash
pip install MDAnalysis
```

#### 深度学习
```bash
pip install torch torchvision
# 或者
pip install tensorflow
```

#### 多模态融合
```bash
pip install scikit-learn
```

#### 可视化
```bash
pip install matplotlib
```

## 运行项目

### 1. 环境准备
```bash
cd e:\pycharmcode\flu-screen
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 初始化数据库
```bash
flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

### 3. 启动应用
```bash
python run.py
```

### 4. 访问应用

| 功能 | 访问地址 |
|------|----------|
| 首页 | http://localhost:5000/ |
| MD模拟 | http://localhost:5000/md-simulation |
| 轨迹分析 | http://localhost:5000/trajectory-analysis |
| 深度学习 | http://localhost:5000/deep-learning |
| 多模态融合 | http://localhost:5000/multimodal-fusion |
| 轨迹可视化 | http://localhost:5000/trajectory-visualization |
| 化合物管理 | http://localhost:5000/compounds |
| 实验管理 | http://localhost:5000/experiments |
| 数据分析 | http://localhost:5000/analysis |
| 筛选工作流 | http://localhost:5000/workflow |

## API端点汇总

### MD模拟API (`/api/md/*`)
| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/status` | 获取MD系统状态 |
| POST | `/prepare` | 准备MD系统 |
| POST | `/minimize` | 能量最小化 |
| POST | `/equilibrate-nvt` | NVT平衡 |
| POST | `/equilibrate-npt` | NPT平衡 |
| POST | `/production` | 生产运行 |
| POST | `/full-workflow` | 完整MD工作流 |

### 轨迹分析API (`/api/trajectory/*`)
| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/status` | 获取分析系统状态 |
| POST | `/analyze` | 完整轨迹分析 |
| POST | `/rmsd` | RMSD计算 |
| POST | `/rmsf` | RMSF计算 |
| POST | `/pca` | PCA降维 |
| POST | `/tsne` | t-SNE降维 |
| POST | `/cluster` | 构象聚类 |
| POST | `/extract-frames` | 关键帧提取 |

### 深度学习API (`/api/deep-learning/*`)
| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/status` | 获取深度学习系统状态 |
| POST | `/train-gcn` | 训练GCN模型 |
| POST | `/train-gat` | 训练GAT模型 |
| POST | `/train-transformer` | 训练Transformer模型 |
| POST | `/predict-key-residues` | 预测关键残基 |
| POST | `/predict-function` | 预测蛋白质功能 |

### 多模态融合API (`/api/multimodal/*`)
| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/status` | 获取融合系统状态 |
| GET/POST | `/pdb-database` | PDB数据库CRUD |
| GET/POST | `/experimental-data` | 实验数据CRUD |
| POST | `/extract-features` | 特征提取 |
| POST | `/fuse-features` | 特征融合 |
| POST | `/train-fusion` | 训练融合模型 |
| POST | `/predict` | 融合预测 |
| POST | `/batch-predict` | 批量预测 |
| GET | `/statistics` | 统计信息 |
| POST | `/export-data` | 导出数据 |

### 轨迹可视化API (`/api/trajectory-visualization/*`)
| 方法 | 端点 | 功能 |
|------|------|------|
| GET | `/status` | 获取可视化系统状态 |
| POST | `/create-3d-plot` | 创建3D轨迹图 |
| POST | `/create-animation` | 创建轨迹动画 |
| POST | `/create-rmsd-plot` | 创建RMSD图 |
| POST | `/create-rmsf-plot` | 创建RMSF图 |
| POST | `/create-conformation-heatmap` | 创建构象热图 |
| POST | `/create-interaction-network` | 创建相互作用网络 |
| POST | `/create-dashboard` | 创建综合仪表板 |
| GET | `/download/<filename>` | 下载可视化文件 |
| GET | `/list-visualizations` | 列出所有文件 |
| DELETE | `/clear-visualizations` | 清除所有文件 |
| POST | `/batch-visualize` | 批量可视化 |

## 使用示例

### MD模拟示例
```python
import requests

# 准备系统
response = requests.post('http://localhost:5000/api/md/prepare', json={
    'protein_file': 'protein.pdb',
    'force_field': 'amber99sb-ildn',
    'solvent': 'tip3p',
    'box_type': 'dodecahedron',
    'box_distance': 1.0
})

# 运行完整工作流
response = requests.post('http://localhost:5000/api/md/full-workflow', json={
    'protein_file': 'protein.pdb',
    'minimization_steps': 50000,
    'equilibration_time': 100,
    'production_time': 10000,
    'temperature': 310
})
```

### 轨迹分析示例
```python
# 轨迹分析
response = requests.post('http://localhost:5000/api/trajectory/analyze', json={
    'trajectory_file': 'traj.xtc',
    'topology_file': 'topol.tpr',
    'reference_structure': 'protein.pdb'
})

# PCA降维
response = requests.post('http://localhost:5000/api/trajectory/pca', json={
    'trajectory_file': 'traj.xtc',
    'topology_file': 'topol.tpr',
    'n_components': 2
})
```

### 深度学习示例
```python
# 训练GCN模型
response = requests.post('http://localhost:5000/api/deep-learning/train-gcn', json={
    'protein_structure': 'protein.pdb',
    'epochs': 100,
    'learning_rate': 0.001,
    'train_val_split': 0.2
})

# 预测关键残基
response = requests.post('http://localhost:5000/api/deep-learning/predict-key-residues', json={
    'protein_structure': 'protein.pdb',
    'model_type': 'gcn'
})
```

### 多模态融合示例
```python
# 特征融合
response = requests.post('http://localhost:5000/api/multimodal/fuse-features', json={
    'pdb_id': '4AVX',
    'exp_id': 'EXP001',
    'fusion_type': 'early'
})

# 训练融合模型
response = requests.post('http://localhost:5000/api/multimodal/train-fusion', json={
    'training_pairs': [['4AVX', 'EXP001'], ['4GMS', 'EXP002']],
    'labels': [0, 1, 0, 1],
    'epochs': 100,
    'learning_rate': 0.001
})
```

## 配置说明

### 环境变量
在项目根目录创建`.env`文件：
```env
# Flask配置
FLASK_APP=src.app
FLASK_ENV=development
SECRET_KEY=your-secret-key

# 数据库配置
DATABASE_URL=sqlite:///flu_screen.db

# GROMACS配置
GMX_EXECUTABLE=gmx
GMX_WORK_DIR=data/gromacs

# 深度学习配置
DEVICE=cpu
MODEL_DIR=data/models

# 可视化配置
VISUALIZATION_DIR=data/visualizations
```

## 技术栈

- **后端框架**: Flask
- **数据库**: SQLAlchemy + SQLite/PostgreSQL
- **MD模拟**: GROMACS
- **轨迹分析**: MDAnalysis
- **深度学习**: PyTorch/TensorFlow
- **数据科学**: NumPy, Pandas, Scikit-learn
- **可视化**: Matplotlib
- **前端**: HTML5, CSS3, JavaScript, Canvas API

## 开发说明

- Python版本：3.10+
- 推荐使用虚拟环境
- 数据库迁移使用Flask-Migrate
- API遵循RESTful设计
- 支持跨域请求（CORS）

## 常见问题

### Q: GROMACS未找到怎么办？
A: 确保GROMACS已安装并添加到系统PATH，或在配置文件中指定GMX_EXECUTABLE路径。

### Q: 深度学习功能不可用？
A: 需要安装PyTorch或TensorFlow。如果未安装，系统会使用模拟数据。

### Q: 可视化功能受限？
A: 需要安装Matplotlib。如果未安装，系统会返回模拟结果。

### Q: 如何更换数据库？
A: 修改`.env`文件中的DATABASE_URL，支持SQLite、MySQL、PostgreSQL等。

## 许可证

本项目仅用于学术研究目的。

## 联系方式

如有问题或建议，请联系项目维护者。
