# Negotiation Final：Belief-Usable Planning for Language Negotiation

这是当前研究的**唯一后续开发目录**。旧代码、历史报告和完整原始实验记录保存在同级的 `../negotiation_past/`；模型权重、conda/venv 和仍需调用的上游 AgenticPay 代码没有搬动。`/work5/qixint/negotiation` 是指向本目录的兼容软链接。

项目目标：在保持对手/卖方固定的比较中，只改 focal buyer（或 focal agent），让对手 belief 真正改变 probe、offer、concession、accept/continue 决策，并由 action lock 和 validator 生成合法自然语言行动。最终指标是 benchmark 官方 buyer reward/BuyerScore，不是 belief accuracy。

给下一位 Codex/agent 的完整交接说明见 [AGENT_README_CN.md](AGENT_README_CN.md)。所有结果、代码结构和发表路线见 [完整中文项目报告](docs/shared/NEGOTIATION_BELIEF_PLANNER_PROJECT_COMPLETE_SUMMARY_AND_PUBLICATION_ROADMAP_CN.md)。

## 目录

```text
framework/                 统一 schema、belief store/updater、planner、engine、验证事件
Simple_Env/                单 buyer/seller/商品；V5.9 与 V6.3；Direct/CoT/full
CaSiNo_Env/                自然语言 3-issue camping allocation；belief-usable chooser
AgenticPay_Env/            多商品、多 seller、多 buyer、多 issue；V37/V55/V60
NegotiationArena/          buy–sell/resource exchange；Direct/OS/framework 比较
experiments/               OpenAI-compatible model client、AgenticPay runner、低成本训练
tests/                     统一框架和各环境测试
tools/                     AgenticPay 完整 trajectory 查看器
supplementary/             我们的 ANL 与 LLM-Deliberation 集成代码（非主结果）
docs/{shared,simple,casino,agenticpay,negotiationarena,auxiliary}/
                            按方法和环境精选的报告
docs/presentations_final/   按时间排列的五份主要可编辑 PPT
runs/{simple,casino,agenticpay,negotiationarena}/
                            本地关键 run；Git 忽略，不在公开仓库中
```

代码保持上述顶层 Python package 名称，故实验入口和 import 不会因文档归类而失效。注意：`AgenticPay_Env/buyer/universal_framework.py` 保留 V1–V60 的实现以复现历史实验；**接下来应只编辑 V60、共享核心及明确标出的 adapter/validator，而不是继续修改所有历史 variant**。代码焦点见 [CODE_MAP_CN.md](CODE_MAP_CN.md)。

## 当前最可信结果

| Benchmark | 方法 | 官方 focal 指标 | 相比 baseline | 状态 |
|---|---|---:|---|---|
| Simple 128 × 3 | V6.3 | reward **0.5130** | +0.1066 vs CoT，CI [0.0581, 0.1560]；+0.0666 vs full，CI [0.0203, 0.1118] | 完整本地 split |
| AgenticPay only_multi_seller 29 × 3 | V60 | BuyerScore **30.7184** | +13.89 vs CoT；+13.64 vs full；+8.04 vs Direct（后者 CI 跨 0） | 强子集结果 |
| CaSiNo official test 100 | chooser | P1 score **22.17–22.30** | 比 Direct 高约 3.6–4.8，但 agreement 略降 | 全 split，单 rollout |
| NegotiationArena | role-dependent | focal reward | buyer/resource-second 优于 OS，seller/resource-first 落后 | 不能称全设置领先 |

**不能引用为正式全集结果：** AgenticPay 旧 V60 all-116/all-231 均含 25 个已定位 adapter instrumentation errors；Task5 修复 smoke 已过，但修复后的 full family/all-116/all-231 尚未干净重跑。focal multi-buyer 的 V60 旧运行全部遇到模型服务连接失败。详情见 [AgenticPay 计划](docs/agenticpay/EXPERIMENT_PLAN_CN.md)。

## 安装与依赖

```bash
git clone git@github.com:TQXoi/negotiation.git negotiation_final
cd negotiation_final
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
bash scripts/bootstrap_upstreams.sh
```

`bootstrap_upstreams.sh` 下载原始 AgenticPay/CaSiNo。上游代码保留其自身 license，不属于本文方法。当前机器的 `benchmarks/AgenticPay` 是指向 `/work5/qixint/benchmarks/AgenticPay` 的**本地软链接**；新机器应删除/替换该失效软链接后按 bootstrap 脚本重新下载。不要把上游目录、模型权重或 venv 上传 GitHub。

模型客户端从环境变量读凭据。OpenAI-compatible 本地服务可使用非秘密 placeholder：

```bash
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://127.0.0.1:8002/v1
```

不要在代码、run config、报告、PPT 或 shell history 中写真实 API key。

## 验证

当前机器无模型调用的完整相关测试：

```bash
PYTHONPATH="$PWD/NegotiationArena:$PWD" python -m pytest -q \
  tests Simple_Env/tests NegotiationArena/tests AgenticPay_Env/tests
```

2026-09-07 审计结果：`181 passed`。未设置 `PYTHONPATH` 时 NegotiationArena 两个测试存在 collection import 问题；应在下一次 packaging 修改中解决。

发布前运行：

```bash
bash scripts/check_publication.sh
git status --short
git check-ignore -v runs/simple/simple_full128_r3_formal_s0_42/config.json
```

`runs/`、raw calls、model traces、checkpoints、`.env` 与 token 文件由 `.gitignore` 排除。精选报告/PPT 和 reviewed aggregate analysis 可发布；完整原始轨迹只在本地归档。

## 开始下一轮工作

1. 阅读 [AGENT_README_CN.md](AGENT_README_CN.md) 的 P0–P3 计划与 claims ledger 规则。
2. 先完成修复后的 AgenticPay 29-task family、all-116/all-231、focal multi-buyer paired validation；不要先训练新 planner。
3. 同时冻结 Simple V5.9/V6.3 与 CaSiNo chooser，在不重复调 test 的条件下补三 rollout 和因果 ablation。
4. 再将 CaSiNo/NegotiationArena 的环境专用 belief/planner 接入共同 `framework` interface，验证同一 final method 的跨环境有效性。
5. 每一轮实验同步更新 `docs/shared/CLAIMS_LEDGER_CN.md`，只把证据足够的主张写入论文。

## 许可证

本项目新增代码按仓库 MIT license；CaSiNo 数据为 CC BY 4.0，其余 benchmark 保留上游 license。详见 `THIRD_PARTY.md`。
