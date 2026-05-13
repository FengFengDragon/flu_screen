from flask import Blueprint, request, jsonify, Response, current_app
import os
import re
import json
import time
import logging
import tempfile
from rdkit import Chem
from src.services.ligand_preprocessor import ligand_preprocessor
from src.services.enhanced_docking import enhanced_docking

logger = logging.getLogger(__name__)

bp = Blueprint('virtual_screening', __name__, url_prefix='/api/virtual-screening')


def _parse_sdf_to_smiles(content_bytes: bytes) -> list:
    """从 SDF 文件的二进制内容中提取所有分子的 SMILES"""
    tmp = tempfile.NamedTemporaryFile(suffix='.sdf', delete=False)
    try:
        tmp.write(content_bytes)
        tmp.flush()
        tmp.close()
        supplier = Chem.SDMolSupplier(tmp.name, removeHs=True)
        smiles_list = []
        for mol in supplier:
            if mol is not None:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_list.append(smi)
        return smiles_list
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


@bp.route('/parse', methods=['POST'])
def parse_smiles():
    """解析并验证 SMILES 列表"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400
    result = ligand_preprocessor.parse_smiles_batch(smiles_list)
    return jsonify({"success": True, **result})


@bp.route('/descriptors', methods=['POST'])
def get_descriptors():
    """计算单个分子描述符"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    if not smiles:
        return jsonify({"success": False, "error": "未提供 SMILES"}), 400
    result = ligand_preprocessor.calculate_descriptors(smiles)
    if result is None:
        return jsonify({"success": False, "error": "无效的 SMILES"}), 400
    return jsonify({"success": True, "descriptors": result})


@bp.route('/filter/lipinski', methods=['POST'])
def filter_lipinski():
    """Lipinski 五规则过滤"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400
    results = [ligand_preprocessor.lipinski_filter(s) for s in smiles_list]
    passed = [r for r in results if r.get('pass')]
    return jsonify({"success": True, "total": len(smiles_list),
                    "passed": len(passed), "results": results})


@bp.route('/filter/admet', methods=['POST'])
def filter_admet():
    """ADMET 性质过滤"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400
    results = [ligand_preprocessor.admet_filter(s) for s in smiles_list]
    passed = [r for r in results if r.get('pass')]
    return jsonify({"success": True, "total": len(smiles_list),
                    "passed": len(passed), "results": results})


@bp.route('/filter/pains', methods=['POST'])
def filter_pains():
    """PAINS 干扰化合物过滤"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400
    results = [ligand_preprocessor.pains_filter(s) for s in smiles_list]
    passed = [r for r in results if r.get('pass')]
    return jsonify({"success": True, "total": len(smiles_list),
                    "passed": len(passed), "results": results})


@bp.route('/preprocess', methods=['POST'])
def preprocess():
    """完整预处理流程：解析 → Lipinski → ADMET → PAINS → 描述符"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    filters = data.get('filters', None)
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400
    result = ligand_preprocessor.full_preprocess(smiles_list, filters)
    return jsonify(result)


@bp.route('/conformer', methods=['POST'])
def generate_conformer():
    """生成单个分子的 3D 构象"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    if not smiles:
        return jsonify({"success": False, "error": "未提供 SMILES"}), 400
    result = ligand_preprocessor.generate_3d_conformer(smiles)
    return jsonify(result)


@bp.route('/to-pdb', methods=['POST'])
def to_pdb():
    """将 SMILES 转为 PDB 格式（供对接使用）"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    if not smiles:
        return jsonify({"success": False, "error": "未提供 SMILES"}), 400
    result = ligand_preprocessor.mol_to_pdb_block(smiles)
    return jsonify(result)


@bp.route('/screen', methods=['POST'])
def screen():
    """
    完整虚拟筛选流程：预处理 → 分级对接
    输入: smiles_list, target_pdb(可选), binding_site_coords(可选), center(可选), box_size(可选),
          filters(可选), tier1_count, tier2_count, tier3_count
    """
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400

    filters = data.get('filters', None)
    target_pdb = data.get('target_pdb', None)
    binding_site_coords = data.get('binding_site_coords', None)
    center = data.get('center', None)
    box_size = data.get('box_size', None)
    tier1_count = data.get('tier1_count', 500)
    tier2_count = data.get('tier2_count', 100)
    tier3_count = data.get('tier3_count', 20)
    adaptive = data.get('adaptive', False)

    preprocess_result = ligand_preprocessor.full_preprocess(smiles_list, filters)
    if not preprocess_result['success']:
        return jsonify(preprocess_result)

    passed_smiles = [m['smiles'] for m in preprocess_result['passed_molecules']]
    if not passed_smiles:
        return jsonify({
            "success": False,
            "error": "所有分子均未通过预处理过滤",
            "preprocess_stats": preprocess_result['stats']
        })

    docking_result = enhanced_docking.tiered_docking(
        passed_smiles,
        target_pdb=target_pdb,
        binding_site_coords=binding_site_coords,
        center=center,
        box_size=box_size,
        tier1_count=tier1_count,
        tier2_count=tier2_count,
        tier3_count=tier3_count,
        adaptive=adaptive,
    )

    return jsonify({
        "success": True,
        "preprocess_stats": preprocess_result['stats'],
        "docking": docking_result,
        "top_candidates": docking_result.get('tier3', {}).get('compounds', [])[:10],
    })


@bp.route('/screen-stream', methods=['POST'])
def screen_stream():
    """SSE 流式虚拟筛选，带进度返回"""
    data = request.get_json() or {}
    smiles_list = data.get('smiles_list', [])
    if not smiles_list:
        return jsonify({"success": False, "error": "未提供 SMILES 列表"}), 400

    filters = data.get('filters', None)
    target_pdb = data.get('target_pdb', None)
    binding_site_coords = data.get('binding_site_coords', None)
    center = data.get('center', None)
    box_size = data.get('box_size', None)
    tier1_count = data.get('tier1_count', 500)
    tier2_count = data.get('tier2_count', 100)
    tier3_count = data.get('tier3_count', 20)
    adaptive = data.get('adaptive', False)

    def generate():
        total_molecules = len(smiles_list)

        yield _sse_event('progress', {
            'step': 'start', 'pct': 0,
            'msg': f'开始处理 {total_molecules} 个小分子...'
        })

        yield _sse_event('progress', {
            'step': 'preprocess', 'pct': 10,
            'msg': f'正在预处理 {total_molecules} 个分子（Lipinski/ADMET/PAINS过滤）...'
        })
        time.sleep(0.1)

        preprocess_result = ligand_preprocessor.full_preprocess(smiles_list, filters)
        if not preprocess_result['success']:
            yield _sse_event('error', {'msg': preprocess_result.get('error', '预处理失败')})
            return

        stats = preprocess_result['stats']
        yield _sse_event('progress', {
            'step': 'preprocess_done', 'pct': 30,
            'msg': f'预处理完成: {stats["passed"]}/{stats["total"]} 通过过滤'
        })

        passed_smiles = [m['smiles'] for m in preprocess_result['passed_molecules']]
        if not passed_smiles:
            yield _sse_event('error', {'msg': '所有分子均未通过预处理过滤', 'preprocess_stats': stats})
            return

        docking_method = "Vina" if target_pdb else "RDKit"
        yield _sse_event('progress', {
            'step': 'tier1', 'pct': 33,
            'msg': f'使用 {docking_method} 对接，开始 Tier1 粗筛（{len(passed_smiles)} 个分子）...'
        })
        time.sleep(0.1)

        docking_result = enhanced_docking.tiered_docking(
            passed_smiles,
            target_pdb=target_pdb,
            binding_site_coords=binding_site_coords,
            center=center,
            box_size=box_size,
            tier1_count=tier1_count,
            tier2_count=tier2_count,
            tier3_count=tier3_count,
            adaptive=adaptive,
        )

        t1 = len(docking_result.get('tier1', {}).get('compounds', []))
        yield _sse_event('progress', {
            'step': 'tier1_done', 'pct': 55,
            'msg': f'Tier1 完成: {t1} 个分子通过粗筛'
        })

        t2 = len(docking_result.get('tier2', {}).get('compounds', []))
        yield _sse_event('progress', {
            'step': 'tier2_done', 'pct': 70,
            'msg': f'Tier2 完成: {t2} 个分子通过标准对接'
        })

        t3 = len(docking_result.get('tier3', {}).get('compounds', []))
        yield _sse_event('progress', {
            'step': 'tier3_done', 'pct': 85,
            'msg': f'Tier3 完成: {t3} 个候选分子通过精细对接'
        })

        yield _sse_event('progress', {
            'step': 'finalizing', 'pct': 95,
            'msg': '正在整理结果...'
        })

        result = {
            "success": True,
            "preprocess_stats": stats,
            "docking": docking_result,
            "top_candidates": docking_result.get('tier3', {}).get('compounds', [])[:10],
        }
        yield _sse_event('result', result)

    return Response(generate(), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


def _sse_event(event_type, data):
    """构造 SSE 事件格式"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@bp.route('/binding-energy', methods=['POST'])
def binding_energy():
    """计算单个分子的结合自由能"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    target = data.get('target', 'NP')
    if not smiles:
        return jsonify({"success": False, "error": "未提供 SMILES"}), 400
    result = enhanced_docking.calculate_binding_free_energy(smiles, target)
    if result.get('success'):
        result['influenza_potential'] = enhanced_docking.evaluate_influenza_potential(
            result['mmgbsa_energy'])
    return jsonify(result)


@bp.route('/binding-mode', methods=['POST'])
def binding_mode():
    """分析分子结合模式"""
    data = request.get_json() or {}
    smiles = data.get('smiles', '')
    if not smiles:
        return jsonify({"success": False, "error": "未提供 SMILES"}), 400
    return jsonify(enhanced_docking.analyze_binding_mode(smiles))


def _cif_to_pdb(cif_content: str) -> str:
    """Convert mmCIF content to PDB format (ATOM/HETATM records only)."""
    lines = []
    atom_site_cols: list = []
    in_atom_site = False

    for line in cif_content.split('\n'):
        stripped = line.strip()

        if stripped == 'loop_':
            in_atom_site = False
            atom_site_cols = []
            continue

        if stripped.startswith('_atom_site.'):
            col_name = stripped.split('.', 1)[1].strip()
            atom_site_cols.append(col_name)
            in_atom_site = True
            continue

        if stripped.startswith('_') and in_atom_site and not stripped.startswith('_atom_site.'):
            in_atom_site = False
            continue

        if stripped.startswith('#'):
            in_atom_site = False
            continue

        if not in_atom_site or not stripped:
            continue

        parts = stripped.split()
        if len(parts) < len(atom_site_cols):
            # 字段数不足，跳过这行（可能是多行值的一部分）
            continue

        row = dict(zip(atom_site_cols, parts))
        record = row.get('group_PDB', '')
        if record not in ('ATOM', 'HETATM'):
            continue

        try:
            serial = int(row.get('id', '0'))
        except ValueError:
            serial = 0

        atom_name = row.get('auth_atom_id', row.get('label_atom_id', ''))
        alt_loc = row.get('label_alt_id', '')
        if alt_loc in ('.', '?'):
            alt_loc = ''
        res_name = row.get('auth_comp_id', row.get('label_comp_id', ''))
        chain = row.get('auth_asym_id', row.get('label_asym_id', ''))

        try:
            res_num = int(row.get('auth_seq_id', row.get('label_seq_id', '0')))
        except ValueError:
            res_num = 0

        try:
            x = float(row.get('Cartn_x', '0'))
            y = float(row.get('Cartn_y', '0'))
            z = float(row.get('Cartn_z', '0'))
        except ValueError:
            continue

        try:
            occupancy = float(row.get('occupancy', '1.0'))
        except ValueError:
            occupancy = 1.0

        try:
            b_factor = float(row.get('B_iso_or_equiv', '0.0'))
        except ValueError:
            b_factor = 0.0

        element = row.get('type_symbol', atom_name[0] if atom_name else 'C')

        pdb_line = (
            f"{record:<6s}{serial:5d} {atom_name:<4s}{alt_loc:1s}"
            f"{res_name:>3s} {chain:1s}{res_num:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
            f"{occupancy:6.2f}{b_factor:6.2f}          "
            f"{element:>2s}  "
        )
        lines.append(pdb_line)

    if lines:
        lines.append('END')
    return '\n'.join(lines)


def _resolve_path(raw_path):
    import re
    from pathlib import Path
    p = Path(raw_path)
    if p.is_dir():
        return p
    m = re.match(r'^([A-Za-z]):[/\\](.+)$', str(p).replace('\\', '/'))
    if m:
        drive = m.group(1).lower()
        rest = m.group(2)
        wsl = Path(f'/mnt/{drive}/{rest}')
        if wsl.is_dir():
            return wsl
    return p


@bp.route('/pdb-files', methods=['GET'])
def list_pdb_files():
    """列出所有可用的受体蛋白文件（PDB、ZIP中的CIF模型）"""
    import zipfile
    from pathlib import Path

    pdb_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'pdb'

    raw = current_app.config.get('EXTERNAL_PDB_DIR', '').strip()
    if not raw:
        raw = os.environ.get('EXTERNAL_PDB_DIR', '').strip()
    if not raw:
        raw = r'E:\graduationproject\分子对接+blast比对程序\蛋白存放'

    ext_path = _resolve_path(raw)
    if not ext_path.is_dir():
        ext_path = _resolve_path(r'E:\graduationproject\分子对接+blast比对程序\蛋白存放')
    if not ext_path.is_dir():
        ext_path = None

    print(f"[PDB] external dir: {ext_path}, valid={ext_path.is_dir() if ext_path else False}")

    cache_dir = Path(__file__).resolve().parent.parent.parent / 'data' / 'pdb_cache'
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    files = []
    
    if pdb_dir.is_dir():
        for f in sorted(pdb_dir.iterdir()):
            if f.suffix.lower() == '.pdb' and f.is_file():
                files.append({
                    'name': f.name,
                    'path': str(f.resolve()),
                    'size_kb': round(f.stat().st_size / 1024, 1),
                    'source': 'local',
                })
    
    if ext_path and ext_path.is_dir():
        for f in sorted(ext_path.iterdir()):
            if f.suffix.lower() == '.pdb' and f.is_file():
                files.append({
                    'name': f.name,
                    'path': str(f.resolve()),
                    'size_kb': round(f.stat().st_size / 1024, 1),
                    'source': 'external',
                })
            elif f.suffix.lower() == '.zip' and f.is_file():
                zip_name = f.stem
                try:
                    with zipfile.ZipFile(str(f), 'r') as zf:
                        model_cifs = [
                            n for n in zf.namelist()
                            if ('model_0' in n and n.endswith('.cif'))
                        ]
                        if not model_cifs:
                            model_cifs = [
                                n for n in zf.namelist()
                                if n.endswith('.cif') and 'model_' in n
                            ]
                        if not model_cifs:
                            model_cifs = [
                                n for n in zf.namelist() if n.endswith('.cif')
                            ]
                        if model_cifs:
                            cif_name = model_cifs[0]
                            cached_pdb = cache_dir / f"{zip_name}.pdb"
                            if not cached_pdb.exists():
                                cif_data = zf.read(cif_name).decode('utf-8', errors='ignore')
                                pdb_content = _cif_to_pdb(cif_data)
                                if pdb_content.strip():
                                    cached_pdb.write_text(pdb_content, encoding='utf-8')
                            if cached_pdb.exists():
                                files.append({
                                    'name': f'{zip_name}.pdb (from {f.name})',
                                    'path': str(cached_pdb.resolve()),
                                    'size_kb': round(cached_pdb.stat().st_size / 1024, 1),
                                    'source': 'alphafold',
                                })
                except Exception:
                    pass
    
    print(f"[PDB] Returning {len(files)} files")
    return jsonify({"success": True, "files": files})


@bp.route('/vina-dock', methods=['POST'])
def vina_dock_single():
    """单分子 Vina 对接（用于测试或详细分析）"""
    data = request.get_json() or {}
    pdb_file = data.get('pdb_file', '')
    smiles = data.get('smiles', '')
    center = data.get('center', None)
    box_size = data.get('box_size', None)
    binding_site_coords = data.get('binding_site_coords', None)
    exhaustiveness = data.get('exhaustiveness', 8)
    n_poses = data.get('n_poses', 10)

    if not pdb_file or not smiles:
        return jsonify({"success": False, "error": "需要提供 pdb_file 和 smiles"}), 400

    safe_path = _resolve_safe_pdb_path(pdb_file)
    if not safe_path:
        return jsonify({"success": False, "error": "PDB文件不存在或路径不合法"}), 404

    try:
        from src.services.vina_docking import vina_docking_service
        if not vina_docking_service.is_available():
            return jsonify({"success": False, "error": "vina 未安装，请先 pip install vina"}), 500

        result = vina_docking_service.dock_single(
            pdb_path=safe_path,
            smiles=smiles,
            center=center,
            box_size=box_size,
            binding_site_coords=binding_site_coords,
            exhaustiveness=exhaustiveness,
            n_poses=n_poses,
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route('/vina-status', methods=['GET'])
def vina_status():
    """检查 Vina 是否可用"""
    try:
        from src.services.vina_docking import vina_docking_service
        available = vina_docking_service.is_available()
    except ImportError:
        available = False
    return jsonify({"available": available})


@bp.route('/upload-parse', methods=['POST'])
def upload_and_parse():
    """上传文件并解析SMILES/蛋白质序列（支持多文件和文件夹）"""
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        uploaded_files = request.files.getlist('file')
    if not uploaded_files:
        single = request.files.get('file')
        if single:
            uploaded_files = [single]
    if not uploaded_files:
        return jsonify({"success": False, "error": "未上传文件"}), 400

    allowed_extensions = {'.txt', '.smi', '.csv', '.sdf'}

    all_proteins = []
    all_smiles = []
    all_errors = []
    seen_smiles = set()
    total_files = 0

    for file in uploaded_files:
        if not file or file.filename == '':
            continue
        file_ext = os.path.splitext(file.filename or '')[1].lower()
        if file_ext not in allowed_extensions:
            continue
        total_files += 1

        try:
            content_bytes = file.read()

            if file_ext == '.sdf':
                sdf_smiles = _parse_sdf_to_smiles(content_bytes)
                for smi in sdf_smiles:
                    if smi not in seen_smiles:
                        seen_smiles.add(smi)
                        all_smiles.append(smi)
                continue

            content = content_bytes.decode('utf-8')
            lines = content.split('\n')
            current_protein = None
            current_seq = []

            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                if line.startswith('>'):
                    if current_protein and current_seq:
                        all_proteins.append({
                            'id': current_protein,
                            'sequence': ''.join(current_seq),
                            'length': len(''.join(current_seq))
                        })
                    current_protein = line[1:] if len(line) > 1 else f'protein_{len(all_proteins) + 1}'
                    current_seq = []
                elif _is_smiles(line):
                    mol = Chem.MolFromSmiles(line)
                    if mol:
                        smi = Chem.MolToSmiles(mol)
                        if smi not in seen_smiles:
                            seen_smiles.add(smi)
                            all_smiles.append(smi)
                    else:
                        all_errors.append({'line': line_num, 'content': line, 'error': '无效的SMILES格式'})
                elif _is_protein_sequence(line):
                    current_seq.append(line)
                elif re.match(r'^[A-Za-z]+$', line):
                    current_seq.append(line)

            if current_protein and current_seq:
                all_proteins.append({
                    'id': current_protein,
                    'sequence': ''.join(current_seq),
                    'length': len(''.join(current_seq))
                })
        except Exception as e:
            all_errors.append({'file': file.filename, 'error': str(e)})

    if not all_smiles and not all_proteins:
        return jsonify({
            "success": False,
            "error": f"在 {total_files} 个文件中未找到有效分子或蛋白质序列"
        }), 400

    return jsonify({
        "success": True,
        "filename": f"{total_files} 个文件" if total_files > 1 else (uploaded_files[0].filename or ''),
        "proteins": all_proteins,
        "smiles_list": all_smiles,
        "stats": {
            "total_proteins": len(all_proteins),
            "total_smiles": len(all_smiles),
            "total_errors": len(all_errors),
            "total_files": total_files
        },
        "errors": all_errors
    })


@bp.route('/upload-screen', methods=['POST'])
def upload_and_screen():
    """上传文件并直接进行虚拟筛选（支持多文件和文件夹）"""
    uploaded_files = request.files.getlist('files')
    if not uploaded_files:
        uploaded_files = request.files.getlist('file')
    if not uploaded_files:
        single = request.files.get('file')
        if single:
            uploaded_files = [single]
    if not uploaded_files:
        return jsonify({"success": False, "error": "未上传文件"}), 400
    
    tier1_count = request.form.get('tier1_count', 500, type=int)
    tier2_count = request.form.get('tier2_count', 100, type=int)
    tier3_count = request.form.get('tier3_count', 20, type=int)
    target_pdb = request.form.get('target_pdb', None)
    adaptive = request.form.get('adaptive', 'false').lower() in ('true', '1', 'yes')
    allowed_extensions = {'.txt', '.smi', '.csv', '.sdf'}
    
    try:
        smiles_list = []
        seen_smiles = set()

        for file in uploaded_files:
            if not file or file.filename == '':
                continue
            file_ext = os.path.splitext(file.filename or '')[1].lower()
            if file_ext not in allowed_extensions:
                continue

            content_bytes = file.read()

            if file_ext == '.sdf':
                for smi in _parse_sdf_to_smiles(content_bytes):
                    if smi not in seen_smiles:
                        seen_smiles.add(smi)
                        smiles_list.append(smi)
            else:
                content = content_bytes.decode('utf-8')
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('>') and _is_smiles(line) and not _is_protein_sequence(line):
                        mol = Chem.MolFromSmiles(line)
                        if mol:
                            smi = Chem.MolToSmiles(mol)
                            if smi not in seen_smiles:
                                seen_smiles.add(smi)
                                smiles_list.append(smi)
        
        if not smiles_list:
            return jsonify({
                "success": False,
                "error": "文件中未找到有效的SMILES"
            }), 400
        
        preprocess_result = ligand_preprocessor.full_preprocess(smiles_list)
        
        if not preprocess_result['success']:
            return jsonify(preprocess_result)
        
        passed_smiles = [m['smiles'] for m in preprocess_result['passed_molecules']]
        
        if not passed_smiles:
            return jsonify({
                "success": False,
                "error": "所有分子均未通过预处理过滤",
                "preprocess_stats": preprocess_result['stats']
            })
        
        docking_result = enhanced_docking.tiered_docking(
            passed_smiles,
            target_pdb=target_pdb,
            tier1_count=tier1_count,
            tier2_count=tier2_count,
            tier3_count=tier3_count,
            adaptive=adaptive,
        )
        
        return jsonify({
            "success": True,
            "filename": file.filename,
            "preprocess_stats": preprocess_result['stats'],
            "docking": docking_result,
            "top_candidates": docking_result.get('tier3', {}).get('compounds', [])[:10],
        })
        
    except Exception as e:
        return jsonify({"success": False, "error": f"处理失败: {str(e)}"}), 500


def _resolve_safe_pdb_path(pdb_path: str) -> str | None:
    if not pdb_path:
        return None

    base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    allowed_dirs = [
        os.path.normpath(os.path.join(base, 'data', 'pdb')),
        os.path.normpath(os.path.join(base, 'data', 'pdb_cache')),
    ]

    ext_cfg = current_app.config.get('EXTERNAL_PDB_DIR', '').strip()
    if not ext_cfg:
        ext_cfg = os.environ.get('EXTERNAL_PDB_DIR', '').strip()
    if ext_cfg:
        resolved_ext = _resolve_path(ext_cfg)
        if resolved_ext.is_dir():
            allowed_dirs.append(str(resolved_ext))
    fallback = _resolve_path(r'E:\graduationproject\分子对接+blast比对程序\蛋白存放')
    if fallback.is_dir():
        allowed_dirs.append(str(fallback))

    resolved_pdb = _resolve_path(pdb_path)
    is_file = resolved_pdb.is_file()
    resolved_str = str(resolved_pdb.resolve()) if is_file else None
    print(f"[SAFE] input={pdb_path}")
    print(f"[SAFE] resolved={resolved_str}, is_file={is_file}")
    print(f"[SAFE] allowed={allowed_dirs}")
    if resolved_str:
        for allowed in allowed_dirs:
            if resolved_str.startswith(allowed + os.sep) or resolved_str == allowed:
                print(f"[SAFE] MATCHED allowed={allowed}")
                return resolved_str
    print(f"[SAFE] NO MATCH - returning None")
    return None


def _is_protein_sequence(text: str) -> bool:
    """判断文本是否是蛋白质序列（纯氨基酸字母，且不含 SMILES 特殊字符）"""
    if not text or len(text) < 3:
        return False

    # 含有 SMILES 特殊字符的一定不是蛋白质序列
    smiles_chars = set('()[]=#@+\\/-0123456789')
    if any(c in smiles_chars for c in text):
        return False

    protein_amino_acids = set('ACDEFGHIKLMNPQRSTVWYBZXUOacdefghiklmnpqrstvwyBZXUO')
    return all(c in protein_amino_acids for c in text)


def _is_smiles(text: str) -> bool:
    """判断文本是否是 SMILES 格式"""
    if not text or len(text) < 2:
        return False

    # 含有 SMILES 特殊字符则优先尝试解析
    smiles_chars = set('()[]=#@+\\/-')
    has_smiles_char = any(c in smiles_chars for c in text) or any(c.isdigit() for c in text)

    if has_smiles_char:
        return Chem.MolFromSmiles(text) is not None

    # 纯字母串：先排除蛋白质序列，再尝试 SMILES 解析
    if _is_protein_sequence(text):
        return False

    return Chem.MolFromSmiles(text) is not None


@bp.route('/pdb-content', methods=['POST'])
def get_pdb_content():
    data = request.get_json() or {}
    pdb_path = data.get('pdb_path', '')
    safe_path = _resolve_safe_pdb_path(pdb_path)
    if not safe_path:
        return jsonify({"success": False, "error": "PDB文件不存在或路径不合法"}), 404
    try:
        with open(safe_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        return jsonify({"success": True, "pdb_content": content, "path": safe_path})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route('/binding-sites', methods=['POST'])
def predict_binding_sites():
    data = request.get_json() or {}
    pdb_path = data.get('pdb_path', '')
    safe_path = _resolve_safe_pdb_path(pdb_path)
    if not safe_path:
        return jsonify({"success": False, "error": "PDB文件不存在或路径不合法"}), 404
    try:
        from src.services.ml_binding_predictor import ml_binding_predictor
        result = ml_binding_predictor.predict_binding_sites(safe_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@bp.route('/dock-3d', methods=['POST'])
def dock_3d():
    data = request.get_json() or {}
    pdb_path = data.get('pdb_path', '')
    smiles = data.get('smiles', '')
    center = data.get('center', None)
    box_size = data.get('box_size', None)
    binding_site_coords = data.get('binding_site_coords', None)
    if not pdb_path or not smiles:
        return jsonify({"success": False, "error": "需要 pdb_path 和 smiles"}), 400
    safe_path = _resolve_safe_pdb_path(pdb_path)
    if not safe_path:
        return jsonify({"success": False, "error": "PDB文件不存在或路径不合法"}), 404
    try:
        from src.services.vina_docking import vina_docking_service
        if not vina_docking_service.is_available():
            return jsonify({"success": False, "error": "Vina 未安装"}), 500
        if center is None or box_size is None:
            if binding_site_coords is None:
                try:
                    from src.services.ml_binding_predictor import ml_binding_predictor
                    bs_result = ml_binding_predictor.predict_binding_sites(safe_path)
                    if bs_result.get('success'):
                        residues = bs_result.get('binding_residues', [])
                        coords = []
                        for r in residues:
                            c = r.get('coordinates') or r.get('ca_coords')
                            if c and len(c) >= 3:
                                coords.append(c)
                        if coords:
                            binding_site_coords = coords
                except Exception:
                    pass
        result = vina_docking_service.dock_single(
            pdb_path=safe_path,
            smiles=smiles,
            center=center,
            box_size=box_size,
            binding_site_coords=binding_site_coords,
            exhaustiveness=8,
            n_poses=5,
        )
        if result.get('success'):
            work_dir = result.get('work_dir', '')
            pose_path = os.path.join(work_dir, 'ligand_out.pdbqt') if work_dir else ''
            ligand_pdbqt = ''
            if pose_path and os.path.exists(pose_path):
                with open(pose_path, 'r', encoding='utf-8', errors='ignore') as f:
                    ligand_pdbqt = f.read()
            result['ligand_pdbqt'] = ligand_pdbqt
            receptor_pdbqt_path = os.path.join(work_dir, 'receptor.pdbqt') if work_dir else ''
            receptor_pdbqt = ''
            if receptor_pdbqt_path and os.path.exists(receptor_pdbqt_path):
                with open(receptor_pdbqt_path, 'r', encoding='utf-8', errors='ignore') as f:
                    receptor_pdbqt = f.read()
            result['receptor_pdbqt'] = receptor_pdbqt
            # 读完文件后清理临时目录
            if work_dir and os.path.isdir(work_dir):
                import shutil as _shutil
                _shutil.rmtree(work_dir, ignore_errors=True)
            result.pop('work_dir', None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500
