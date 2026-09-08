"""Archive small diagnostic summaries; leave raw visual/sensor evidence untracked."""
import csv,html,json,pathlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
raw=ROOT/'tmp/fp12';out=ROOT/'results/hm3dv1_fp12_20260908';out.mkdir(exist_ok=False)
rows=json.loads((raw/'analysis_v2/summary.json').read_text())
labels={
309:('semantic_fp','镜子被识别为电视；语义实例 434 为 mirror。'),
1850:('semantic_fp','窗户被识别为电视；语义实例 318 为 window glass。'),
1340:('semantic_fp','红色沙发被识别为床；语义实例 709 为 couch。'),
1:('task_goal_excluded','检测到 wall tv 231，但任务 tv_monitor 目标只有 monitor 353、354。'),
638:('valid_instance_navigation_failure','检测到 chair 274，且 274 在任务目标列表内；不能归因于场景中另一把未收录椅子。'),
625:('task_goal_excluded','检测到 toilet 191，但任务目标只有 toilet 41；本例 stop_called=0。'),
1558:('valid_instance_navigation_failure','检测画面包含任务认可的 couch 9；距目标视点 0.329 m。另一个探针落在 pillow 58，说明局部像素标签不可替代候选关联。'),
1517:('category_boundary','检测为 chair 的高脚坐具被场景标为 stool 176、177，均不在 chair 任务目标中。单列类别边界。'),
5:('valid_instance_navigation_failure','检测到任务认可的 toilet 454，终点距离仍为 2.717 m；历史导航目标点缺失，具体投影/停止机制待定。'),
1796:('valid_instance_navigation_failure','第 467 视频帧检测到任务认可的 toilet 329；首次 navigate 在第 468 步，500 步超时，未调用目标停止。'),
1894:('semantic_fp','厨房垃圾桶被识别为马桶；语义实例 180 为 trashcan。'),
1600:('semantic_fp','床头灯附近区域被识别为椅子；检测框内探针对应 lamp 316。'),
}
names={'semantic_fp':'明确语义误检','task_goal_excluded':'真实同类实例未列为任务目标','category_boundary':'类别边界','valid_instance_navigation_failure':'检测实例有效，导航失败机制待定'}
for r in rows:
    r['audit_class'],r['finding']=labels[r['num']]
    r['classification_scope']='visually audited detector-associated instance, not complete historical nav_goal causality'
    r['av_rescue_status']='not_tested'
(out/'cases.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
with (out/'cases.csv').open('w',encoding='utf-8-sig',newline='') as f:
    fields=['num','scene','episode_id','category','selection_group','audit_class','finding','video_frame','replay_step','steps','stop_called','historical_distance','replayed_distance','alignment_correlation','historical_nav_goal_available','av_rescue_status']
    w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
prov=dict(replay_manifest=json.loads((raw/'visual/manifest.json').read_text()),replay_exit=json.loads((raw/'visual/exit.json').read_text()),
    analysis_commit='3e45d52',analysis_directory='analysis_v2',tests='python -m unittest discover -s tests -v: 15 passed',
    failed_attempts=[dict(directory='analysis_v1',status='failed',reason='NumPy bool_ not JSON serializable; fixed with bool conversion; original output retained')],
    evidence_root='/home/zyq/AV-Nav/runs/fp12_full_20260908',case_sources=[])
for r in rows:
    e=json.loads((raw/'visual'/f"{r['num']:04d}"/'evidence.json').read_text())
    prov['case_sources'].append(dict(num=r['num'],source_hashes=e['source_hashes'],video_sha256=e['video_sha256'],
        dataset_sha256=e['selection']['dataset_sha256'],log_sha256=e['selection']['log_sha256']))
(out/'provenance.json').write_text(json.dumps(prov,indent=2),encoding='utf-8')
table='\n'.join(f"| {r['num']} | {r['scene']} / {r['episode_id']} | {r['category']} | {names[r['audit_class']]} | {r['finding']} |" for r in rows)
report=f'''# HM3Dv1：12 个固定诊断案例的回放与实例证据

审计日期：2026-09-08。已完成 12/12 个案例的历史动作回放，共 2093 步，覆盖 7 个场景、5 类目标。该样本用于诊断流程和机制覆盖，不是随机抽样，不能外推到全部 427 个 episode。

## 已确认的候选实例结果

- 5 例明确语义误检：镜子→电视、窗户→电视、沙发→床、垃圾桶→马桶、床头灯区域→椅子。
- 2 例真实同类实例没有进入任务目标列表：wall tv 231、toilet 191。场景语义标注存在；尚未确定任务生成器排除它们的原因，不能直接称为漏标或擅自重算官方成功率。
- 1 例 stool/chair 类别边界，单列，不混入明确误检或漏标计数。
- 4 例检测到了任务认可的实例却导航失败。具体历史目标点、投影或停止机制仍待定。

上述分组是本次观察到的检测候选实例证据分类，不代表已经恢复了全部历史政策内部状态或证明唯一失败原因。AV 挽救数量尚未测试。

| 总序号 | 场景 / episode | 任务类别 | 复核结果 | 证据说明 |
|---|---|---|---|---|
{table}

## 选样协议及复核变化

回放前在提交 58b7a4d 固定 configs/diagnostics/hm3dv1_fp12.json，按语义误检、任务目标收录、定位/停止、证据不足四类线索各选 3 例。分组是筛查假设；回放后没有为凑类别数量替换案例。优先使用已有 36 例关键帧检查中的证据，兼顾跨场景和目标差异；受现有证据覆盖限制，12 例覆盖 7 个场景、5 类目标，没有 plant，且同一场景出现多例。不能把该样本称为六类别代表性样本。

重要反例：svBbv1Pavdk/33 因场景存在未收录 chair 132 而入选目标收录组，但实际检测到的是已收录的 chair 274。必须将场景级风险和 episode 实例级证据分开。

## 回放核对

从原分组日志读取逐步动作，按场景文件的 episode 枚举索引恢复初始位置、旋转和任务目标。明确避免原始 JSON 中重复 episode_id 的错配。

使用服务器现有 Habitat-Sim 0.2.4，640×480、HFOV 79°、相机高度 0.88 m、前进 0.25 m、转向 30°、allow_sliding=False，GPU 3。只执行原动作，不调用检测模型、不运行 AV、不改变轨迹。

12 例回放后的目标视点测地距离，全部在原视频两位小数记录的舍入误差内，最大差异 0.004981 m。每例检查起始、导航开始、途中和末段的抽取画面；关键证据帧还单独校准了复合视频中的 RGB 裁剪大小，并比较相邻状态。视频 f 帧对应回放 f+1 状态；停止/碰撞造成不变画面时相邻偏移可能并列。选定证据帧灰度相关性为 0.893–0.992，检测框和掩膜覆盖会降低相关性。该相关性仅是复核信号，原视频和回放画面同时保留。

逐案例保存原视频帧、RGB 回放、depth/semantic 数组、全部动作状态、相机位姿、语义对象 AABB、任务目标 ID 与视点、数据和场景资源哈希。传感器证据保存在服务器 runs，代码及小型摘要通过 origin 同步。

## 候选位置的含义与限制

根据画面选择检测框关联物体内的像素，将其深度通过相机内外参投到世界坐标，交叉核对语义实例 ID、AABB 和任务目标列表。结果见 cases.json 的 probes。所有探针世界点均落入对应语义 AABB；除沙发旁 pillow 边界探针外，5×5 邻域均为同一实例。它们是可复查的物体可见表面坐标，不是整物体中心，也不是历史 nav_goal。

原运行没有保存 nav_goal、候选点云及其跨帧关联状态，因此不能据此精确断言四例有效实例失败分别来自哪一种投影/停止机制。没有重建出这些状态时，报告必须保留待定。所有抽查对象中心都通过了初始楼层掩膜的高度条件；本样本没有证明楼层过滤是这些失败的原因。

## 对下一步实验的用途

5 例语义误检可作为冻结方法的诊断纠错候选，检查触发、拒绝、恢复探索。只有完整导航在相同步数预算内成功，才能算挽救。2 例任务未收录实例适合检验是否错误拒绝真实目标，不能拿“识别正确”替代官方任务成功。1 例类别边界用于单独报告。4 例有效实例失败应优先补录政策内部候选状态，避免把定位问题误当语义问题。

本批来自验证场景，不用于阈值调参。校准仍在训练场景进行。没有 AV 配对重跑，也没有改变任何原始成功率。

## 文件与执行记录

- cases.csv / cases.json：12 例表格、实例、深度、世界坐标、任务收录状态和回放核对结果。
- provenance.json：回放提交、分析提交、资源哈希及失败记录。
- 本地证据浏览页：D:/zyq/AV_nav/tmp/fp12/diagnostic12.html。
- 服务器原始证据：/home/zyq/AV-Nav/runs/fp12_full_20260908。
- 本地测试 15 项通过，服务器单例 smoke 和 12 例动作回放完成。
- analysis_v1 因 numpy.bool_ 无法 JSON 序列化失败；保留原目录，修复后 analysis_v2 全部完成。该故障未影响已完成的回放与传感器数据。
'''
(out/'report.md').write_text(report,encoding='utf-8')
cards=[]
for r in rows:
    n=f"{r['num']:04d}";e=json.loads((raw/'visual'/n/'evidence.json').read_text())
    body='<article><h2>'+html.escape(f"#{r['num']} {r['scene']} / {r['episode_id']} — {names[r['audit_class']]}")+'</h2><p>'+html.escape(r['finding'])+'</p>'
    body+='<p>'+html.escape(f"视频帧 {r['video_frame']} / 回放状态 {r['replay_step']}；终点距离：原记录 {r['historical_distance']:.2f} m，回放 {r['replayed_distance']:.6f} m")+'</p>'
    body+='<img loading="lazy" src="analysis_v2/'+n+'.jpg"><details><summary>查看像素探针与三维表面坐标</summary><pre>'+html.escape(json.dumps(r['probes'],ensure_ascii=False,indent=2))+'</pre></details>'
    body+='<details><summary>原视频抽取帧（保留检测框与地图）</summary>'
    for image in sorted((raw/'visual'/n).glob('original_*.jpg')):
        body+='<p>'+image.stem+'</p><img loading="lazy" src="visual/'+n+'/'+image.name+'">'
    body+='</details></article>';cards.append(body)
page='''<!doctype html><meta charset="utf-8"><title>12 个诊断案例证据</title><style>body{font:16px system-ui;max-width:1280px;margin:24px auto;padding:16px;background:#f4f6f8;color:#163044}article,header{background:white;padding:22px;margin:22px 0;border:1px solid #ccd6dd;border-radius:10px}img{width:100%}pre{overflow:auto;font-size:13px}summary{cursor:pointer;padding:10px;font-weight:bold}header{background:#fff6db}</style><h1>VLFM HM3Dv1：12 个固定诊断案例</h1><header>12/12 动作回放完成。5 例语义误检，2 例真实同类实例未列为任务目标，1 例类别边界，4 例有效检测实例的导航失败机制待定。这是定向诊断样本，不能外推总体比例。所有三维探针均为回放 RGB-D 表面点，历史 nav_goal 未保存。AV 挽救结果尚未测试。</header>'''+''.join(cards)
(raw/'diagnostic12.html').write_text(page,encoding='utf-8')
print('Archived 12 cases and created local evidence gallery.')
