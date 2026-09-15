# Iteration 062：AgenticPay BuyerScore-first 自动研究流水线

## 唯一优化目标

只修改 buyer、固定 native seller，并以 AgenticPay 官方 `mean BuyerScore` 为晋级主指标。
agreement、valid deal、timeout、belief calibration 和 action trace 只做约束或解释，不能替代
BuyerScore 宣称改进。

## 自动阶段

1. 等待 Iteration 061 的 3 seeds × 29 tasks × 6 variants fresh matrix 完成；
2. 以 task 为 cluster 计算 paired bootstrap CI，并形成失败轨迹图谱；
3. 固定 task-level development/held-out split，Codex 看不到 held-out 名称；
4. 最多两轮单假设规则修改，每轮保留 prompt、Codex 日志、前后代码快照、测试和在线结果；
5. 每个候选在固定 seller、同 task/seed 上与当前 champion 配对，BuyerScore 不提高则不晋级；
6. 冻结规则 champion 后，训练一个小型、可拒绝、能精确回退 champion 的 residual/gate；
7. 训练模型仅使用 development 轨迹，语言模型、seller、env、scorer 均冻结；
8. learned variant 先过 held-out task gate，再做一次 fresh all-29 confirmation。

## 低成本训练约束

- 允许 CPU logistic/linear/small MLP ensemble 或 conservative AWR；禁止微调 30B LLM；
- label 直接来自 official BuyerScore；split unit 是 task_path；
- runtime feature 只能来自公开历史、belief/candidate 属性和 buyer 自己的 utility；
- seller private truth 只能用于离线诊断，不能序列化为 inference feature；
- 不确定/OOD 时必须 exact fallback 到冻结 champion。

## 运行与监控

```bash
bash /work5/qixint/universal_reward_research/iteration_062_agenticpay_score_autoloop/run_detached.sh
tmux capture-pane -pt agenticpay_score_autoloop:0 -S -100
tail -f /work5/qixint/universal_reward_research/iteration_062_agenticpay_score_autoloop/events.jsonl
cat /work5/qixint/universal_reward_research/iteration_062_agenticpay_score_autoloop/LIVE_STATUS.json
```

`events.jsonl` 是全过程账本；每轮目录保留代码快照、修改说明、命令日志、结果与 promotion gate。
`LIVE_STATUS.json` 每分钟更新完成数、8002 健康状态、后台会话状态与距上次进展时间；若矩阵
会话意外退出且 8002 正常，监督器会使用 resume-safe 脚本继续缺失记录。
