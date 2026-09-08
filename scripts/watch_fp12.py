"""Monitor a fixed suite and write immutable progress/final diagnostic summaries.

Runs outside the experiment checkout. No code updates, process termination, tuning,
or Git mutations. Can keep recording after an SSH/client disconnect.
"""
import argparse,datetime,json,pathlib,subprocess,sys,time
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--suite-id',required=True);p.add_argument('--output',type=pathlib.Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
suite=a.root/'runs'/a.suite_id
previous=None;index=0
while True:
    code=subprocess.run([sys.executable,str(a.root/'scripts/summarize_fp12_suite.py'),'--root',str(a.root),'--suite-id',a.suite_id],capture_output=True,text=True)
    stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
    if code.returncode:
        with (a.output/'errors.jsonl').open('a') as f:f.write(json.dumps(dict(time=stamp,returncode=code.returncode,stderr=code.stderr))+'\n')
    else:
        data=json.loads(code.stdout);fingerprint=json.dumps(data,sort_keys=True)
        if fingerprint!=previous:
            data['snapshot_time']=stamp;index+=1
            (a.output/f'progress_{index:03d}.json').write_text(json.dumps(data,indent=2))
            previous=fingerprint
            print(stamp,'completed',data['complete'],'failed',data['failed'],flush=True)
        if (suite/'exit.json').exists():
            (a.output/'final.json').write_text(json.dumps(data,indent=2))
            lines=['# FP12 诊断对照最终执行记录','',f"完成 {data['complete']}/36 项，失败 {data['failed']} 项。",'',
                '固定参数、验证集定向诊断；不是总体有效性结果。所有原始案例历史上均失败，不能由此估计总体成功案例损失。','',
                '| 案例 | 策略 | 成功 | 步数 | 触发 | 换视点 | 拒绝事件 | 目标距离 |','|---|---|---:|---:|---:|---:|---:|---:|']
            for r in data['rows']:
                if r['status']=='complete':lines.append(f"| {r['num']} | {r['strategy']} | {r['success']} | {r['steps']} | {r['triggers']} | {r['views']} | {r['rejections']} | {r['distance']:.3f} |")
                else:lines.append(f"| {r['num']} | {r['strategy']} | 运行失败 | | | | | |")
            lines+=['','拒绝事件可重复，不能当作独立候选数或已纠正误检数。候选是否对应已审计物体，需要图像/坐标匹配。','',
                '## 与本轮基线配对','']
            for strategy in ['passive','active']:
                pairs=[r for r in data['pairs'] if r['strategy']==strategy]
                lines.append(f"- {strategy}：已配对 {len(pairs)}；挽救 {sum(r['recovered'] for r in pairs)}；损失 {sum(r['regressed'] for r in pairs)}。")
            (a.output/'final.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
            break
    time.sleep(30)
