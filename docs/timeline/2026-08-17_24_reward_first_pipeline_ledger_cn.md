# Universal buyer reward research：不可覆盖流水线账本

本文件只追加，不用后续结论改写历史。两个唯一优化目标是：

1. Simple Env 原生 `reward`；
2. AgenticPay fixed-seller 官方 `buyer_score`。

seller、benchmark scorer 和 test labels 固定；belief/calibration/action-flip 等只作解释。

## Iteration 000：初始可信全集（运行中）

- 日期：2026-08-17；
- GPU：仅项目 GPU 1/2；8002=GPU1，8003=GPU2；
- Simple：128 scenarios × 3 rollouts × 5 variants = 1920 episodes；
- AgenticPay：28 tasks × 3 seeds × 5 variants = 420 episodes；
- 初始 framework source 已存档为 `iteration_000_full_baseline/code_snapshots/iteration000_runtime_source.tar.gz`；
- SHA256：`dc8222de5d9ad8cd02b3ed997caac1f8d2ebed72d5dd4683b40915b8d26aa0b7`；
- 预注册：`iteration_000_full_baseline/PREREGISTERED_PROTOCOL_CN.md`。

运行中发生但不改变方法的工程事件：

1. AgenticPay 原 formal run 因 12,288 context overflow 作废；新 context-safe run 从零开始，保留原生 completion budget，只对历史做有审计的 compaction。详见 `AGENTICPAY_CONTEXT_REPAIR_AUDIT_CN.md`。
2. Simple 一个 transient `RemoteDisconnected` 先记为 `final:false`，同 episode clean retry 成功；正式失败计数只统计 `final:true`。
3. 为遵守资源约束，停止 GPU3/4/5 的本项目服务/评测，保留进度后在 GPU1 的 8002 resume；详见 `GPU_RESOURCE_PAUSE_AUDIT_CN.md` 与 `SIMPLE_RESHARD_AUDIT.json`。
4. 分析工具增加 task/scenario cluster paired bootstrap、verified proposal event 与 AgenticPay action/render consistency；这些修改不进入正在运行的 buyer/seller runtime。
5. 第三分片完成时 monitor 曾显示 631/630：五个正式 variant 文件实际是 630 个唯一成功键，额外一行来自保留的 `final:false` transient failure audit。只修复 monitor/finalizer 排除该审计文件，raw 不删不改。详见 `SIMPLE_TRANSIENT_FAILURE_COUNT_AUDIT_CN.md`。
6. Simple 1920/1920 完成后，将最慢的 AgenticPay seed 20260817 从 GPU2/8003 clean-interrupt 后以同目录 resume 到空闲 GPU1/8002；另两 seed 保持 GPU2。只迁移未完成请求，不改已落盘 outcome。
7. 迁移时发现旧 resume 将 endpoint URL 作为 model identity，曾重建 seed17 JSONL。已立即停止，从每 task 日志中的 upstream Summary JSON 恢复 101 条原记录，保留 2 条新完成记录，direct Task3 明确重跑，并新增 opt-in 同 served-model endpoint resume 校验。详见 `AGENTICPAY_ENDPOINT_MIGRATION_RECOVERY_AUDIT_CN.md`。

未完成前不得根据 partial mean 宣布最佳方法。

## Iteration 001：action-consistent renderer（已预注册，未运行）

- 触发证据：AgenticPay V1 当前部分日志的 176 个可审计 turn 中，158 个 locked `offer` 被自然语言说成第一人称 “accept/agree”；
- 唯一候选改动：新增独立 V3，使 locked action 与 public preamble 确定性一致，并验证 action semantics；
- 不改 belief、candidate、planner、seller、scorer；
- 预注册：`iteration_001_action_consistent_renderer/HYPOTHESIS_CN.md`；
- 状态：等待 Iteration 000 全集与正式 clustered 分析完成。

## 后续候选（尚未预注册，不得合并修改）

- proposal-backed accept/continue terminal safety；
- AgenticPay belief/planner core 与 Simple V5.9 对齐；
- posterior-predictive seller-feasibility lower bound；
- 只有 online 主 reward 支持后才单独进入下一 iteration。

## 2026-08-17 状态追加：Iteration 000 完成

- Simple 1920/1920、AgenticPay 420/420；正式数据无运行失败；
- Simple：V6.3 `reward=0.5130`，对 CoT/full 的 paired clustered CI 均高于 0，
  但对 V5.9 的差值 `+0.0149`、CI 跨 0；
- AgenticPay：direct `buyer_score=32.9531`，V1 `15.7307`，V2 `18.1775`；
  universal variants 尚不合格；
- 正式分析：`iteration_000_full_baseline/analysis/PRIMARY_METRICS.md`、
  `VERIFIED_REWARD_EVENTS.md`、`AGENTICPAY_RENDERER_DIAGNOSTICS.md`。

## 2026-08-17 状态追加：Iteration 001 已启动

- 新 variant：`universal_framework_v3_action_consistent`；
- 唯一方法改动：deterministic action-aware preamble + action semantics validator；
- V1/V2 不覆盖；14 项相关测试通过；
- smoke：Task1，2 rounds，`buyer_score=82.0286`，0 failures，公开 action 一致；
- 正式 dev：固定 8 tasks × 3 seeds，buyer-only、native fixed seller；运行仅使用
  8002/GPU1 与 8003/GPU2；
- 版本与 SHA：`iteration_001_action_consistent_renderer/SOURCE_MANIFEST.sha256`；
- 原始运行：`iteration_001_action_consistent_renderer/runs/`。

## 2026-08-17 状态追加：Iteration 001 dev 完成

- 24/24，0 failures；V3 `buyer_score=33.1317`，V1 paired `19.7585`；
- V3 − V1 `+13.3732`，task-clustered CI `[-7.9182,+39.9159]`；
- action mismatch 从 paired V1 `158/177` 降为 V3 `0/329`；
- 但同子集 direct `50.0790`、CoT `45.9631`、full `34.6891`，V3 未过强
  baseline 门槛；
- deal 62.5%、timeout 37.5%、accept selected/available `12/236`、late deferral
  53；
- 结论：保留 V3 作为 protocol-correct base，不扩跑全集；进入 Iteration 002
  单变量 accept/continue 修复。

## 2026-08-17 状态追加：Iteration 002 dev 完成

- 新 variant：`universal_framework_v4_terminal_accept_guard`；唯一方法改动为
  public-proposal-backed accept/continue boundary；
- 服务中断后从已有 JSONL 保守恢复；最终 8 tasks × 3 seeds = 24/24，0 failures；
- V4 `buyer_score=41.5588`，V3 `33.1317`，full `34.6891`，direct `50.0790`，
  CoT `45.9631`；
- V4 − V3 `+8.4271`，task-clustered CI `[-6.0281,+25.2507]`；V4 − full
  `+6.8697`，CI `[-25.0962,+36.4600]`；
- deal rate `87.5%`、timeout `12.5%`、late accept deferral `0`、action mismatch
  `0/280`、buyer-IR violations `0`；
- 预注册 candidate 晋级门槛通过，但仍低于 direct/CoT，不能作为最终方法；
- 轨迹瓶颈：Task6 重复选择已经正式被拒绝的 133.82，虽有更高安全候选仍直至 timeout；
  下一轮只增加 verified response frontier 与 V4 terminal guard 的组合，不改 belief、seller
  或 scorer。完整结论见 `iteration_002_terminal_accept_guard/RESULTS_AND_NEXT_STEPS_CN.md`。

## 2026-08-17 状态追加：Iteration 003 dev 完成并拒绝

- 新 variant：`universal_framework_v5_verified_response_frontier`；唯一方法改动是用公开
  counter/reject 形成 response frontier，惩罚仍停留在已拒绝区域、且存在 buyer-IR successor
  的 offer；不读取 seller private utility；
- Task6 smoke 修复重复 133.82 的行为并缩短到 6 轮，但 seller 主动提出的合同违反其隐藏 IR，
  official score 仍为负；该 oracle truth 未进入 deployed buyer；
- 固定 8 tasks × 3 seeds = 24/24，0 runtime failures；V5 `buyer_score=39.0328`，V4
  `41.5588`，V5 − V4 `-2.5260`，task-clustered CI `[-37.9495,+35.6155]`；
- deal `95.83%`、timeout `4.17%`、buyer IR violation `0`、action mismatch `0/180`；
  frontier 在 73 turns 对 279 candidates 实际生效，但 evaluator-only seller-private feasibility
  mismatch 达 `11/24`；
- Task3/12 分别相对 V4 `+96.51/+52.20`，但 Task2/8/10 分别
  `-67.34/-28.04/-74.86`；只沿 response/price frontier 会诱发对手侧不可行的过早成交；
- 按预注册 `mean(V5)>mean(V4)` gate 拒绝 V5，不扩跑 AgenticPay full28，不迁移 Simple；
  V4 仍是 AgenticPay frozen reference；
- 下一候选只能是使用重复、完整 public seller proposal 作为 counterparty-feasibility anchor 的
  单变量 guard；不得偷看 seller cost/utility。完整结论见
  `iteration_003_verified_response_frontier/RESULTS_AND_NEXT_STEPS_CN.md`。

## 2026-08-17 状态追加：Iteration 004 targeted smoke 完成并拒绝

- 新 variant：`universal_framework_v6_public_proposal_feasibility_guard`；唯一机制是在 V5 上用
  公开 seller proposal 的 exact repeat/price plateau 触发 early exact-proposal accept-vs-counter
  comparison；不读取 seller private utility；
- 15 项 unit tests 通过；targeted Task2/3/8/10/12、seed20260817 完成 5/5，0 runtime、
  0 action semantic failure、0 buyer IR failure；
- V6 mean official buyer score `18.4289`；旧 V5 五任务记录 `35.8744`；但 native seller
  采样/请求顺序不同，该差值只作 smoke gate，不作 paired causal claim；
- 核心 reachability：36 turns 中 guard considered/override 均为0；真实轨迹中的 proposal
  stability 与 rejection-streak gate 没有同时满足；
- score-success mismatch 4/5；Task8/12 证明 seller public proposal 本身也可能违反隐藏
  feasibility，盲信最后 proposal 不是可行解；
- 按预注册 gate 拒绝 V6，不运行8×3、full28或Simple，不在本 iteration 内调阈值；
- 下一步先做公开 trajectory feature 的 posterior-predictive bilateral-feasibility offline
  identifiability audit；只有可区分正/负候选后才新建 online iteration。完整结论见
  `iteration_004_public_proposal_feasibility_guard/RESULTS_AND_NEXT_STEPS_CN.md`。

## 2026-08-17 状态追加：Iteration 005 可识别性审计完成，未通过跨域 gate

- 冻结 280 条 agreement records、28 个 task groups，按 task grouped out-of-fold；public feature
  audit 无 private leakage；
- all-public：ROC-AUC `0.6831`、balanced accuracy `0.6685`、Brier `0.2339`、negative recall
  `0.6746`；相对 price-only balanced accuracy 提升 `+0.1746`，task-clustered 95% CI
  `[+0.0453,+0.3130]`；
- domain split 暴露不可泛化：multi-issue negative recall `0.8163`，但 price-only 仅 `0.1786`，
  低于预注册的每域 `0.50` gate；
- 决策：不把当前 feasibility classifier 接入 planner，不在线调 V6 阈值；V4 仍为 AgenticPay
  frozen reference；下一独立候选转向 action-locked strategic naturalization，feasibility 模型等待
  更多跨 seller/repeated-opponent labels 后重审；
- 完整结果见
  `iteration_005_public_feasibility_identifiability/AUDIT_RESULT_AND_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 006 V7 smoke 完成并拒绝

- 新 variant：`universal_framework_v7_action_locked_strategic_naturalization`；冻结 V4
  belief/candidates/planner，只让 LLM根据最新 public seller rationale 生成 OFFER preamble；
- 首次 run 在 2 个完整 task 后发现 protocol `:g` precision 导致 false lock repair，以及生成文本
  提及 trusted renderer；原 run 全部保留并标为 implementation-invalid，不进入 reward gate；
- canonical protocol comparator 与 mechanism-vocabulary filter 修复后，相关测试 `20/20`，从全新
  retry1 目录完成 Task2/3/6/8/12；
- retry1 mean BuyerScore `42.2629`，历史 V4 同任务 `29.7015`，descriptive delta `+12.5615`，
  CI `[−15.9233,+54.5309]`；deal 100%、timeout 0、runtime/lock/semantic/buyer-IR failure 均0；
- Task6/8 仍为 seller-private infeasible mismatch；strategic free-form preamble 实际启用率仅
  `20/37=54.05%`，低于预注册 70% gate；
- 按规则拒绝 V7，不运行 8×3/full28/Simple；V4 继续 frozen；下一独立候选为 constrained
  rhetorical-act label selector + deterministic action-locked template。完整报告见
  `iteration_006_action_locked_strategic_naturalization/RESULTS_AND_NEXT_STEPS_CN.md`。

## 2026-08-17 状态追加：Iteration 007 V8 smoke 通过，正式 dev 已启动

- 新 variant：`universal_framework_v8_constrained_rhetorical_selector`；冻结 V4 的 belief、候选、
  planner 与终止边界，LLM 只从五个公开修辞标签中选择，trusted code 映射为 action-locked 模板；
- Task2/3/6/8/12、seed20260817 完成 5/5；BuyerScore `26.1165`，历史同任务 V4
  `29.7015`；deal 100%、timeout 0；
- runtime/lock/semantic/buyer-IR failure 全0；37/37 OFFER turns 标签有效，使用2种标签，但
  `FIRM_BOUNDARY=36/37`，存在明显选择塌缩；
- score-success mismatch 3/5；Task3 无 mismatch 得 `94.1480`，满足预注册正向可达性要求；
- smoke 的 catastrophe/reachability gate 全部通过，因此已使用 8002/GPU1、8003/GPU2 启动固定
  8 tasks ×3 seeds dev；只有 dev mean 超过 V4 且不由 mismatch/单任务驱动才可晋级 full28；
- 完整 smoke 决策见
  `iteration_007_constrained_rhetorical_selector/SMOKE_RESULT_AND_DEV_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 007 V8 dev 完成并拒绝

- 固定8 tasks ×3 seeds 完成24/24；V8 BuyerScore `27.7394`，V4 `41.5588`，差值
  `−13.8194`、task-clustered CI `[−46.4700,+23.9013]`；亦低于 direct `50.0790`、CoT
  `45.9631`、旧 full `34.6891`；
- deal 100%、timeout 0、协议/语义/buyer-IR failures 全0，但 score-success mismatch
  `12/24=50%`；Task2/6/8/10 均出现 seller-private infeasible agreement；
- selector 156/156 有效，但 `FIRM_BOUNDARY=150/156`，标签选择明显塌缩；
- 三个 seeds 的逐任务 BuyerScore 和轮数完全相同，当前 greedy 配置没有产生有效 rollout 差异；
  统计仍按8个 task clusters，不能将24条视为独立样本；
- 按 dev 晋级条件拒绝 V8，不扩跑 full28/Simple；回退 V4，下一 iteration 测试 public-only LLM
  proposal candidate + frozen belief-aware planner，而非继续修改 public rhetoric；
- 完整结果见 `iteration_007_constrained_rhetorical_selector/DEV_RESULTS_AND_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 008 V9 smoke 完成并拒绝

- 新 variant：`universal_framework_v9_llm_proposal_candidate`；V4 全部冻结，只增加 public-only
  structured LLM proposal candidate，经 schema/hard ceiling/buyer-IR 检查后进入同一 planner；
- Task2/3/4/6/10 完成5/5；BuyerScore `28.1817`，历史同任务 V4 `42.9420`，差值
  `−14.7603`、CI `[−44.2808,0]`；
- 65/65 parsed，58/65 buyer-IR-valid，49 distinct candidates 注入，但 planner selected `0`；
  Task3/6 均未改善，按机制可达性与 reward gate 拒绝，不进入8×3；
- 诊断：LLM 候选通常更接近 seller，但 V4 的 p_accept/utility 仲裁将其排在5–15位；候选覆盖与
  belief-aware arbitration 必须联合工作；
- 额外 buyer calls 与 seller 共用 endpoint 会改变请求序列，后续 variant 必须使用独立 buyer/seller
  endpoints，并在同配置重跑 V4 control；
- 完整结果见 `iteration_008_llm_proposal_candidate/SMOKE_RESULTS_AND_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 009 V10 controlled smoke 完成并拒绝

- 新 variant：V9 candidate pool + stalled-state belief-grounded candidate-ID arbitrator；buyer=8002、
  seller=8003，并在同双端点配置 fresh 重跑 V4 control；
- V10 BuyerScore `−2.1478`，fresh V4 `28.1817`，paired delta `−30.3295`、task-clustered CI
  `[−57.2011,−3.9299]`；V10 deal40%、timeout60%、mismatch40%；
- arbitrator 60/60 有效、53次 override，协议与buyer-IR failures全0，但Task2新增mismatch，
  Task4/10由高分成交变timeout，Task3/6无改善；
- 结论：合法 candidate-ID arbitration 不等于策略安全；仅靠 prompt 无法校准 belief-to-action；
  按 gate 拒绝，不进入8×3；
- 下一 iteration 只增加 frozen-reference safe-improvement trust region，作为后续 residual offline
  RL/AWR 的部署安全边界；
- 完整结果见 `iteration_009_belief_grounded_candidate_arbitrator/SMOKE_RESULTS_AND_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 010 V11 safe-improvement smoke 完成并拒绝

- V10上只增加 frozen-reference residual trust region；42次审查中拒绝20次、允许5次；
- V11与fresh V4在Task2/3/4/6/10上的BuyerScore均为`28.1817`，逐任务/outcome完全相同；
- safe gate完全消除V10灾难性退化，但Task3/6无改善、reward delta为0；按预注册拒绝，不扩跑8×3；
- 保留trust region作为后续learned residual policy的部署安全壳；停止继续手调prompt/阈值，转向以
  Simple reward与AgenticPay BuyerScore为直接标签的跨环境residual training；
- 完整结果见 `iteration_010_safe_improvement_arbitrator/SMOKE_RESULTS_AND_DECISION_CN.md`。

## 2026-08-17 状态追加：Iteration 011 reward-labelled residual AWR 离线通过

- 汇总524条去重真实AgenticPay trajectory，并按scenario group切分train/dev；真实轨迹decision
  contexts为train2613、dev738，terminal action使用official BuyerScore，nonterminal使用一步transition
  加V4 next-state value，未执行action使用cross-fit response model；
- 三成员ensemble AWR离线aggregate delta `+0.03192`、cluster bootstrap CI
  `[+0.01712,+0.05099]`；真实AgenticPay域delta `+0.00838`、CI
  `[+0.00117,+0.01852]`，harmful override `5.96%`；预注册离线gate通过；
- 部署审计修复旧AWR默认Behavioral base与本轮V4标签base不一致的问题：V12显式使用
  `ProposalBackedTerminalPlanner`，再叠加ensemble LCB和opt-in safe-improvement trust region；旧variant
  语义不变，相关25 tests通过；
- 当前只证明可以进入online smoke，不能宣称BuyerScore提升；已准备fresh V4 vs V12的固定5-task
  双端点命令与paired审计脚本；8002/8003当前无响应，尚未运行在线实验；
- 完整记录见
  `iteration_011_reward_labeled_residual_training/OFFLINE_RESULTS_AND_ONLINE_PLAN_CN.md`。

## 2026-08-17 状态追加：V12 exact deployment audit拒绝，Iteration 012/V13预注册

- 将V11 trust region纳入exact deployment audit后，V12 aggregate override仅`2.61%`，
  `24.14%` contexts的LCB候选被二次拒绝；delta降到`+0.00042`且CI
  `[−0.00015,+0.00126]`，真实AgenticPay与Simple change-point域轻微为负，V12拒绝且不在线；
- Iteration012只关闭冲突的手写trust region，保留V4 terminal base、相同checkpoint、ensemble LCB、
  OOD penalty、initial-jump guard、action lock与fixed native seller；注册为
  `universal_framework_v13_reward_labeled_lcb`；
- V13对应LCB deployment offline aggregate delta `+0.03192`、CI
  `[+0.01712,+0.05099]`，真实AgenticPay delta `+0.00838`、CI
  `[+0.00117,+0.01852]`；25 tests通过，已准备fresh V4 5v5 smoke；
- 8002/8003当前无响应，online official BuyerScore仍待验证；完整记录见
  `iteration_012_reward_labeled_lcb_deployment/OFFLINE_GATE_AND_STATUS_CN.md`。

## 2026-08-17 状态追加：V13完成Simple跨环境接入，在线因服务停止阻塞

- 同一reward-labelled checkpoint已注册为Simple variant
  `universal_framework_v6_4_reward_labeled_lcb`；Simple保留Behavioral base，AgenticPay保留V4
  terminal-safe base，共享residual feature/checkpoint/LCB abstention，不使用环境名称特征；
- Simple constructor/runtime identity验证通过，5-scenario validation split dry-run通过；已准备V5.9 vs
  V6.4 smoke，以及CoT/full/V5.9/V6.4 validation40×3 paired命令；
- 连续三个goal continuation检查中，8002与8003始终connection refused，当前执行环境同时无法访问
  NVIDIA driver；因此AgenticPay 5v5 official BuyerScore、Simple smoke以及后续全集验证无法启动；
- 所有待运行命令和promotion gate已写入
  `iteration_012_reward_labeled_lcb_deployment/OFFLINE_GATE_AND_STATUS_CN.md`，恢复服务后可直接续跑。

## 2026-08-17 状态追加：V13双环境online smoke完成，Simple validation40×3启动

- 服务恢复后自动完成AgenticPay fresh V4 vs V13固定5-task smoke：V13 BuyerScore `39.8242`、
  V4 `28.1817`，delta `+11.6424`、task win3/5，但CI
  `[−30.8312,+57.3497]`；V13 deal60%、timeout40%，63 turns中29次override，协议/IR failures全0；
- AgenticPay提升高度异质：Task3从`−0.446`升至`97.03`，Task2从`17.55`退化为timeout
  `−2.73`；仅为smoke证据；
- Simple前5 validation scenarios：V6.4 reward `0.25282`、V5.9 `0.24735`，delta
  `+0.00547`、CI `[−0.10306,+0.11166]`；deal100% vs60%，但两个已有deal场景buyer surplus降低；
- 双环境mean primary reward非负且无协议错误，已启动CoT/full/V5.9/V6.4的Simple
  validation40×3 paired运行；完整smoke结果见
  `iteration_012_reward_labeled_lcb_deployment/ONLINE_SMOKE_RESULTS_CN.md`。

## 2026-08-17 状态追加：Simple validation40中期诊断（冻结、不改代码）

- 前4个完整scenario×3 paired时，V6.4−V5.9 reward `−0.20236`，interim clustered CI
  `[−0.31935,−0.11948]`；V6.4 override rate `43.86%`，显著高于offline Simple
  static/repeated的`9.17%/15.83%`；
- 暂定问题为residual activation train/deploy calibration shift，而非协议或candidate legality；
- 当前scenario集中于相邻beauty子域，继续冻结完成40-scenario验证，不根据中期结果手调threshold；
- 记录见`iteration_012_reward_labeled_lcb_deployment/INTERIM_DIAGNOSTIC_NO_CODE_CHANGE_CN.md`。

## 2026-08-17 状态追加：Iteration 013 evidence-gated residual 已预注册并隔离实现

- 随 online validation 增长，已配齐的前34个paired episodes中V6.4每局均发生override，且34/34
  都在无直接对手证据的首轮override；首轮selected action相对V5.9 base平均降低buyer normalized
  utility `0.1092`，显示LCB residual发生train/deploy activation shift；
- 预注册唯一变量：`direct_response_count==0`时learned residual必须abstain并严格返回frozen base；
  获得至少一次直接响应后，checkpoint、LCB、OOD、候选集与其他阈值完全不变；规则不读取环境名；
- 共享planner新增默认关闭的`require_direct_evidence`，历史V6.3/V6.4/V12/V13语义不变；新注册
  Simple `universal_framework_v6_5_evidence_gated_lcb`与AgenticPay
  `universal_framework_v14_evidence_gated_lcb`；只修改buyer；
- 新机制及相关回归测试共`55 passed`，Simple dry-run和AgenticPay registry检查通过；尚无online
  reward结果，不宣称改进；
- 已配置三个零GPU watcher：完整480局后生成final CI/override诊断；仅在完整结果满足预注册机制
  gate时串行运行Simple与AgenticPay paired smoke；smoke结束后自动分析native reward、official
  BuyerScore、cluster bootstrap CI、DealRate、TimeoutRate与override；
- 当前主验证继续冻结运行；完整记录见
  `iteration_013_evidence_gated_residual/PREREGISTERED_PLAN_CN.md`和
  `iteration_013_evidence_gated_residual/IMPLEMENTATION_STATUS_CN.md`。
