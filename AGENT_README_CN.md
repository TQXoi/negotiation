# 给下一位研究 Agent 的交接说明

**日期：2026-09-15。** 本文是继续实验与同步撰写论文的工作说明，不是新的实验结果。先读根目录 `README.md`，再读 `docs/shared/NEGOTIATION_BELIEF_PLANNER_PROJECT_COMPLETE_SUMMARY_AND_PUBLICATION_ROADMAP_CN.md`。权威代码只在本目录；`../negotiation_past/` 只用于查证历史过程。

## 一、要完成什么

我们希望形成一个尽可能 universal 的自然语言 negotiation framework：

1. 对方 belief 不只是 prompt 中的文字画像，而是有证据、置信度和时效性的可更新状态；
2. planner 在 legal candidate set 上使用 belief，能改变 probe、价格/条款、让步、接受或继续谈判；
3. renderer 只把已选结构化动作转换为合理的文字，不允许偷改策略；
4. 错误/不可识别的 belief 可以被拒绝，并安全退化到强基线；
5. 同一方法原则在 Simple/AmazonHistoryPrice、CaSiNo、AgenticPay 与 NegotiationArena 的不同 topology 上成立；
6. 最终优化各环境官方 focal reward/BuyerScore，并保持 agreement、协议正确性及可复现性。

实验约束：主要只修改 buyer/focal agent，seller/partner 固定。不能在某方法条件下偷偷修改 seller prompt、scorer、max turns 或任务参数。内部 belief calibration 只作机制指标。

## 二、论文 story 与可主张的 novelty

建议工作标题：**Decision-Calibrated Belief-Usable Planning for Executable Language Negotiation**。

核心研究命题：自然语言谈判中，“更准确的 opponent belief”与“更好的谈判决策”之间没有自动关系。我们提出一个证据约束的 belief-to-action interface，使 belief 只在可识别、决策相关且不会越过安全边界时影响行动；再通过 action locking、contract validation 和 belief intervention 验证它是否真的提高官方收益。

候选贡献：

- **Factorized, evidence-calibrated belief**：偏好、reservation/aspiration、response policy、seller identity 和 regime 分开表示；区分明示事实、行为证据、角色先验与自由推断；允许 abstain。
- **Belief-usable planner**：belief 进入候选 rank、probe value、concession frontier 和 accept/continue 边界，而不是只进入最终语言 prompt。
- **Safe executable interface**：canonical offer、per-opponent state、action lock、issue-role/IR gate、protocol validator 与 minimal repair。
- **Causal evaluation**：frozen/shuffled/wrong/oracle/no belief、action flip、opponent switch/change point、reward delta 的完整链。

注意 prior art 冲击：Opponent Simulation 直接在 NegotiationArena 上优化对手响应；Game-Theoretic LLM、K-Level、EPO 等已有 opponent modeling/strategic reasoning 想法。不能只以“我们也维护 belief”作为 novelty。必须体现：跨自然语言合同 topology、可拒绝 belief、安全执行、belief usefulness 因果验证和同一冻结 planner 定义。

## 三、当前成果与严禁过度宣称的地方

| 环境 | 冻结候选 | 已完成的强证据 | 不能说什么 |
|---|---|---|---|
| Simple | V5.9 base / V6.3 conservative AWR | 128 场景 × 3 rollouts；V6.3 reward 0.5130，显著高于 CoT 0.4065/full 0.4464 | V6.3 相对 V5.9 的训练增量已显著（其 CI 跨 0） |
| CaSiNo | belief-usable LLM chooser | official test 100；P1 22.17–22.30，比 Direct 高约 3.6–4.8 | 准确 continuous belief 已被证明是主要增益来源（top-1 约 0.38–0.40） |
| AgenticPay | V60 multi-seller policy | only_multi_seller 29 × 3 seeds；BuyerScore 30.7184，显著高于 CoT/full | 修复后 all-116/all-231 已完整领先；V60 vs Direct CI 尚跨 0 |
| NegotiationArena | 研究级 V3.3/V5/V8 | buyer 与 resource-second 的部分设置超过 Opponent Simulation | 一个 unified variant 四设置通胜；V8 所有角色都比 frozen/shuffled 强 |

AgenticPay 旧 all-116/all-231 的 25 个 adapter error 和 focal multi-buyer 的 API failures 是**无效实验行**，不得计作模型失败/成功。Task5 的在线修复 smoke 已过，但全集仍需补跑。

## 四、代码应从哪里看

### 4.1 统一核心

```text
framework/schemas.py                  CanonicalState/Offer/Observation/Belief/Candidate/Decision
framework/engine.py                   belief update → candidate rank → selection validation → locked render
framework/belief.py                   结构化、语义、证据、readiness/aspiration/updater
framework/events.py                   verified observations/provenance
framework/planner.py                  EV、frontier、probe、deadline、terminal-safe planner
framework/conservative_awr_planner.py Simple V6.3 conservative residual
framework/trainable_planner.py        Planner V2 trainable decision boundary
```

当前 `UniversalNegotiationEngine` 的共享 interface 已成形。下一步最重要的代码修改是把各环境自己的候选和 protocol 保留在 adapter 中，而让**一个** planner 接口、belief factor 定义和可靠性 gate 承担跨环境策略。不要把 AgenticPay clause template 直接塞进 `framework/planner.py`。

### 4.2 各环境

```text
Simple_Env/buyer/universal_framework.py
    SimpleEnvAdapter、V5.9、V6.3 与 frozen/shuffled/wrong/oracle 对照
Simple_Env/buyer/factory.py
    variant 注册；当前仍有大量历史 variant，先使用 CODE_MAP 中的 focus set
Simple_Env/eval.py
    frozen scenario + fixed seller paired runner

CaSiNo_Env/buyer/belief_usable_planner/
    belief_model.py、planner.py、llm_chooser.py、generator.py、buyer.py
CaSiNo_Env/eval.py
    Partner-Base 对比入口

AgenticPay_Env/buyer/universal_framework.py
    AgenticPayAdapter、contract candidate、per-seller state、V37/V55/V60
AgenticPay_Env/buyer/issue_classifier.py
    独立可拒绝 issue-role/settlement-eligibility 分类
AgenticPay_Env/buyer/variants.py
    variant registry；V60 全名见下
AgenticPay_Env/environment/all_tasks.py
    all-231 和 focal-only runner；近期 instrumentation 修复处
AgenticPay_Env/eval.py
    single28/all_tasks/multi_agent CLI
experiments/run_agenticpay_single28_framework.py
    原始 AgenticPay agent 接入、official BuyerScore instrumentation
experiments/model_clients.py
    本地/OpenAI-compatible/其他 provider 调用；只从环境变量取 key
tools/view_agenticpay_trajectory.py
    完整任务配置、合同、交互、结果查看

NegotiationArena/arena_integration/decision_calibrated_agent.py
    belief/reliability-gated V2–V8
NegotiationArena/arena_integration/paper_aligned_opponent_simulation.py
    Opponent Simulation 对照
NegotiationArena/arena_integration/run_opponent_simulation_setting.py
    四 focal setting runner
```

辅助代码 `supplementary/ANL_2024_2025`、`supplementary/LLM_Deliberation_integration` 依赖各自原仓库。ANL 没有自然语言；LLM-Deliberation 有 role-name preference leakage，都不宜成为论文主结果。

### 4.3 正式结果与 PPT

- `docs/simple/PRIMARY_METRICS.md`：Simple 全测试 reward 与 CI；
- `docs/agenticpay/README_CN.md`：V60 近期流水线说明；
- `docs/agenticpay/EXPERIMENT_PLAN_CN.md`：all-116/all-231/focal 清理计划；
- `docs/negotiationarena/V8_CONFIRMATORY_ANALYSIS.md`：belief intervention；
- `docs/shared/OPENREVIEW_2026_NEGOTIATION_RELATED_WORK_AND_NOVELTY_STORY_CN.md`：相关工作；
- `docs/presentations_final/`：按 2026-07-27 至 2026-08-17 顺序排列的五份 editable PPT；
- `runs/`：本地 selected raw run，Git 忽略；完整原始记录在 `../negotiation_past/raw_research/`。

## 五、统一方法的下一次修改方向

### 5.1 最小可发表版本

每个环境 adapter 只提供：

```text
parse private/public state
→ canonical issue/offer and per-counterparty observations
→ legal action candidates
→ role-specific feasibility/IR/protocol constraints
→ action-locked rendering
```

跨环境共享组件只负责：

```text
factorized belief + evidence provenance + reliability/abstention
→ posterior predictive response per candidate
→ utility/risk/timeout/information-gain scoring
→ safe action selection and fallback
```

推荐候选级目标形式：

\[
Q(a)=\mathbb E[R_{\mathrm{official}}\mid a,b]
-\lambda_{\mathrm{risk}}\,\mathrm{Downside}(a,b)
-\lambda_{\mathrm{time}}\,\mathrm{TimeoutRisk}(a)
+\lambda_{\mathrm{IG}}\,\mathrm{DecisionRelevantIG}(a,b).
\]

这些 \(\lambda\) 和安全阈值应在 dev 上选择、跨环境冻结或只允许少量预定义 role/topology normalization；不可在最终 test 上按环境/任务事后调参。若 benchmark reward 不可直接在候选阶段计算，使用其 legally available own utility 与 calibrated response proxy，而不是偷看隐藏对手 utility。

### 5.2 当前实现的具体缺口

1. CaSiNo/NegotiationArena 仍有环境自己的 belief/planner；需要接入 shared interface 或在论文中降低 universal claim。
2. AgenticPay V60 `universal_framework.py` 很大；应把合同语法、issue-role、seller routing 和 planner rank 分拆，并以 regression tests 固定现有行为后再改。
3. Simple V6.3 checkpoint 是本地 JSON，在 `checkpoints/simple/`，被 Git 忽略。新机器需安全转移 artifact 或重训；没有 checkpoint 可先复现 V5.9。
4. `pyproject.toml` 尚未包括 NegotiationArena package，测试需临时 `PYTHONPATH`；下一轮包装修复。
5. 旧 benchmark 先验泄漏/单 proposing shortcut 要写成 benchmark validity audit，不可以拿它证明 active belief。

## 六、下一轮实验顺序

### P0：先使 AgenticPay 全任务实验有效

1. 运行 unit tests 与 V60 `multi_products_multi_seller/Task5` smoke；
2. 修复后的该 family **29/29**，检查 zero runner errors 和合同完整性；
3. V60、Direct、CoT、historical full 在同一 model config/seller/manifest 做 **all-116 paired**；
4. 扩展到 **all-231 paired**；
5. focal-only multi-buyer（`--focal-buyer-index 1`），其他 buyer 保持 upstream native；确保 endpoint 健康并自动 resume；
6. 至少 3 seeds/rollouts，task-cluster bootstrap CI；模型服务错误单独计数，不能当 no-deal。

CLI 骨架，务必先核对 `python -m AgenticPay_Env.eval --help` 及本机模型 alias：

```bash
python -m AgenticPay_Env.eval \
  --suite agenticpay_all_tasks \
  --model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --buyer-model 'openai@http://127.0.0.1:8002/v1:MODEL_NAME' \
  --seller-model 'openai@http://127.0.0.1:8003/v1:MODEL_NAME' \
  --agenticpay-task-suites multi_products_multi_seller \
  --buyer-variants 'repo_native,direct_prompt,cot_prompt,full_framework,universal_framework_v60_resilient_multiseller_settlement' \
  --limit 1 --seed 20260915 \
  --output-dir runs/agenticpay/family29_smoke
```

`--limit 1` 成功后去掉它，切换到 `only_multi_seller`、四个 multi-seller family、最后 `all`。使用同一个 frozen seller；不要依照 test trajectory 改 V60。

### P1：补跨环境主结果

- Simple：在新的独立 split、至少 3 rollouts 下比较 Direct/CoT/full/V5.9/V6.3。V6.3 必须传 `--continuous-planner-params-json checkpoints/simple/cross_environment_awr_checkpoint.json`；
- CaSiNo：official test 100 × 至少 3 rollouts，paired P1/joint/agreement/CI；
- NegotiationArena：先冻结一个 method 定义，然后四设置一致汇报；把 Opponent Simulation 当直接 baseline；
- 第二 backbone：至少一个不同开放权重模型或 GPT-4o，固定 seller pairings。

### P2：机制和泛化

最少 ablation：no/static/frozen/shuffled/wrong/oracle belief、no probe、no reliability gate、no validator/action lock、no AWR residual。必须连同 action flip 和 final reward 报告；如果 frozen/shuffled 不弱于 learned，不能硬说 learned belief 是贡献。

在 dev 上做 repeated-opponent、opponent-switch/change-point、held-out issue ontology 和 ambiguous contract clause。选取有 oracle-verified behavioral identifiability 的 profile suite；probe 的收益应扣除轮次/风险成本。

### P3：训练，只有在 P0/P1 后开始

低成本优先：

1. 从 AgenticPay 真实 trajectory 建 verified issue-role 与 settlement eligibility 标签；训练可拒绝 classifier，不让它覆写事实性合同条款；
2. 从 accept/reject/counteroffer 训练 calibrated response predictor，而不是 SFT 模糊的 `flexibility_delta` JSON；
3. conservative offline AWR 只在 legal candidate set 中做小幅 residual rerank，保持 base fallback；
4. critic/训练目标必须是 Simple buyer reward 和 AgenticPay official BuyerScore；
5. 所有 labels 与轨迹按 train/dev/test task family 划分，不让同一 seller profile/contract template 泄漏到 test。

## 七、如何边实验边写论文

不要等所有实验完成才开始写论文。按以下节奏同步推进：

1. **每一次代码修改先写假设。** 在 dev ledger 写：预期会改变哪类 candidate/action、为什么提高官方 reward、哪些失败类型可能变多；保存 commit SHA 和配置。
2. **先 smoke，再 dev，再冻结 test。** Smoke 只验证协议，dev 用于调方法，test 只跑预注册对比。看过 test trajectory 的修改必须作为新版本，下一次用新 held-out split，不能继续把同一 test 当确认实验。
3. **结果与论文 claim 一一对应。** 更新 `docs/shared/CLAIMS_LEDGER_CN.md`：主张、证据路径、样本/CI、负结果、是否可写进摘要。
4. **Method 章节随代码更新。** `framework/` 的接口、factorization、candidate objective、validator/fallback 是论文算法主体；每次改接口要同步修改 method 图和伪代码。
5. **Results 章节以官方指标为中心。** Simple reward、CaSiNo P1/joint、AgenticPay BuyerScore、NegotiationArena focal reward；其他指标作为机制图/附录。
6. **Related Work 区分具体差异。** Opponent Simulation 必须直接 baseline；ANL/LLM-Deliberation 的限制应写成 benchmark-selection rationale。
7. **保留失败 case study。** 成功 probe、错误 belief 被 gate 拒绝、多 seller 正确合同、错误继续/timeout 各一条；脱敏后再进入论文附录。
8. **最终冻结一套方法。** 不准把每环境各自最佳历史版本事后拼成一个 Ours。若最终参数确需 role/topology normalization，应在 method 中预先定义，并在所有实验中一致使用。

## 八、安全、迁移与后台实验

- `runs/`、`checkpoints/`、provider calls、models 与 `.env` 都被 Git 忽略；不可用 `git add -f` 强行加入。
- 每次 push 前执行 `bash scripts/check_publication.sh`，再审查 staged filenames 和 PPT 内 XML 是否含 token。
- 不要删除 `../negotiation_past/`；它保存全部历史记录，README 提供分类。
- 断网时必须使用独立 server-side session/调度器，写逐 episode JSONL 和 heartbeat，保存 resume manifest；不要声称聊天线程本身能长期后台监控。
- 新机器先 bootstrap upstream、转移 checkpoint、检查模型服务 `v1/models`，再跑 unit tests 和 smoke。

## 九、完成发表的验收条件

最低条件：

- 四个主环境有同一方法定义和公平 baseline；
- AgenticPay all-116/all-231 零 infrastructure errors 且 paired CI 可解释；
- Simple/CaSiNo 多 rollout 全 split；
- causal belief controls 能把 reward 变化连接到行动变化；
- 第二 backbone/unseen seller 显示不是固定模型和固定任务模板特化；
- 代码、checkpoint artifact、manifest、分析脚本和 prompt 可复现；
- 论文摘要中的每个性能主张都能在 claims ledger 中找到证据。

如果复杂 AgenticPay 全集仍不超过 Direct，收缩论点：Simple/CaSiNo 为性能主结果，AgenticPay 为真实复杂 topology 的 stress test 和安全退化分析；不能凭 only_multi_seller 子集声称 universal winner。
