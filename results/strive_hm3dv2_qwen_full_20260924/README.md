# STRIVE + Qwen3.5-9B：HM3Dv2 val 全量评估重启记录

## 检查结论

2026-09-24 检查发现，2026-09-17 启动的 suite 没有完成。旧任务完整完成了 episode 0–9 和 20–29；分片 10–19 只写出 episode 10–14 后失败。因此旧任务只有 20 个完整分片内 episode，另有 5 个失败分片中的部分 episode，不能计算或报告 HM3Dv2 全量效果。

失败发生在 Qwen 返回的结构化文本中包含非法控制字符。OpenAI SDK 在解析 JSON 时抛出 Pydantic `ValidationError`。旧适配器只会对 `LengthFinishReasonError` 使用精简最终字段重试，未处理这类无效 JSON，随后全量监督进程按 fail-fast 规则停止。这不是 CUDA OOM；当时两个成功分片均以退出码 0 完成。

## 修复与重新启动

- AV-Nav 修复提交：`fe62c0a3d30fe796a0d58990af4e0582064c1294`。
- STRIVE 固定提交：`1872d73b7db297705d251df73bf5f08ffed0d749`。
- `ValidationError`、`JSONDecodeError` 和长度超限现在都会触发一次只包含最终字段的精简重试。
- 单个 10-episode 分片失败时最多自动创建 3 个独立 attempt；三次均失败才停止 suite。
- 30 项本地单元测试全部通过。
- 新 suite 于 2026-09-24 启动，从 episode 0 重新运行全部 1000 个 episode。旧 suite 的部分指标不会混入新结果。

新 suite：

```text
strive_qwen_hm3dv2_val1000_fe62c0a_20260924
```

服务器固定工作树与结果目录：

```text
/home/zyq/AV-Nav-worktrees/strive-full-fe62c0a
/home/zyq/AV-Nav-worktrees/strive-full-fe62c0a/runs/strive_qwen_hm3dv2_val1000_fe62c0a_20260924
```

运行参数为 100 个分片、每片 10 个 episode、2 个并行 worker、每个分片最多 3 次 attempt。Qwen3.5-9B 使用 GPU0，两个 STRIVE worker 使用 GPU1。

完整公开配置见 [manifest.json](manifest.json)，预检见 [preflight.json](preflight.json)，数据文件哈希见 [dataset_manifest.json](dataset_manifest.json)。记录不含密钥、提示词、原始图像或视频。

## 完成标准

只有新 suite 的 100 个分片全部成功、汇总结果精确覆盖 episode 0–999 且无重复或缺失时，才报告最终 SR、SPL 和 distance-to-goal。启动检查和不足 1000 个 episode 的阶段性数值都不作为效果结论。
