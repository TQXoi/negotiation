# Iteration 007 — V6 Decision-Relevant Value of Information（条件预注册）

状态：**仅预注册，尚未修改 framework，尚未运行实验。** 当前正式 V3.3/V4/V5
矩阵仍在运行；必须在全部 56 个 cells 达到 100/100 latest episodes、0 current
errors 后，用冻结的分析脚本判定是否触发本 iteration。这样避免看到最终结果后任意改
目标、阈值或主指标。

## 1. 初步失败模式（不作为最终结论）

截至正式矩阵未完成时，已经完成的 Buyer/Seller cells 显示：

1. V5 的 acceptance Brier/ECE 明显好于 V4，但 focal reward 没有同步改善；
2. Buyer V4 的 decision-time posterior MAE 小于 frozen，但 frozen reward 反而更高；
3. V4 会用 `sqrt(remaining_episodes) × posterior entropy × one-step candidate IG`
   放大 probe bonus；该 bonus 没有验证 probe 是否真的会改变未来最优 action；
4. trajectory 中存在 posterior 偏差被 planner 转成结构性不可行报价的例子，但这不是
   唯一问题：即使 posterior 更准，过度 probing 也可能牺牲当前已可接受的 deal；
5. oracle 仍有 reward headroom 时，说明 belief inference 仍有改进空间；但
   continuous 不优于 frozen 时，说明 planner utilization 同时是独立瓶颈。

因此下一步不把 novelty 写成“更精确的 opponent belief”，而是：

> A belief is useful only when its uncertainty can change a future decision and the
> expected value of that change exceeds the opportunity cost of eliciting information.

中文表述：**把可校准的 opponent belief 转化为 decision-relevant、机会成本受限、
可因果检验的信息价值。**

## 2. 正式结果触发门槛

完整结果生成后，依次检查：

### Gate A：Calibration–utility gap

若 V5 相对 V4 在至少两个 settings 降低 Brier 或 ECE，但 paired reward 没有正向改善，
则确认“预测校准不等价于策略效用”，触发 V6 planner 改造。

### Gate B：Continuous–frozen reversal

若 V4 continuous 在至少一个 setting 的 paired reward 低于 frozen，且该 setting 的
probe rate、`IG>0` rate 或 accept optimism gap 较高，则触发 decision-relevant VOI。

### Gate C：Belief causality

若 correct continuous 不优于 wrong-confident 和 shuffled，或 action-flip 非零但 reward
方向与 belief quality 不一致，则不得声称当前 planner 有效使用 belief；V6 必须先修复
该因果链，不能只报告总体 reward。

### Gate D：Oracle headroom

- Oracle 明显优于 V4：inference 与 planner 都需要保留为研究对象；
- Oracle 与 V4 相同但均弱于 baseline：主要瓶颈是 planner/action space；
- Oracle 仍弱于 baseline：不再通过提高 belief 精度解释失败，应重做 decision rule。

### Gate E：Opponent switch

只使用 difference-in-differences：

`(switch_post − switch_pre) − (no_switch_post − no_switch_pre)`。

若 raw post−pre 改善但 controlled effect 消失，则不能讲 adaptation story；它只是自然
episode learning。若 continuous 的 controlled recovery 不优于 frozen/wrong/shuffled，
则当前 continuous update 没有得到 switch 因果支持。

## 3. V6 方法假设

V4 当前近似 bonus：

```text
V4_bonus(a) = IG(a) × sqrt(remaining_horizon) × posterior_entropy
```

问题是 entropy reduction 本身没有 utility；只有它改变未来 action 才有价值。V6 改为
posterior-predictive decision value：

```text
EVI(a) = Σ_y p(y | a, belief)
           [max_b V(b | belief updated by y) − max_b V(b | current belief)]

DR_VOI(a) = credibility(a)
            × identifiable_mass(a)
            × action_flip_probability(a)
            × max(0, EVI(a))

probe_bonus(a) = min(
    effective_horizon × DR_VOI(a),
    opportunity_cost_cap,
    posterior_value_range_cap
)
```

其中：

- `y ∈ {ACCEPT, COUNTER, REJECT/EXIT}`，概率来自当前 posterior predictive；
- 每个 `y` 都执行一次 shadow posterior update 和 shadow planner，不调用新 LLM；
- `action_flip_probability` 只奖励可能改变 future locked action 的观测；
- `credibility` 区分对 locked offer 的直接 response 与 endogenous/free-form counter；
- `identifiable_mass` 抑制不同 θ 给出近似相同 response 的不可辨识 probe；
- `opportunity_cost_cap` 保证放弃一个正 utility observed offer 的代价不会被
  `sqrt(horizon)` 无界放大；
- 所有分量必须写入 trace，允许逐决策重算。

## 4. 不允许的实现捷径

1. 不用 buyer price=43、seller WTP=63 或 resource true weights 写 planner heuristic；
2. 不依据正式 test reward 手调某个 setting 专属阈值；
3. 不增加额外 LLM chooser 调用来模糊归因；
4. 不改变 opponent prompt、model、temperature、seed、protocol repair 或 episode 数；
5. 不用最终 reward 训练或更新同一 test episode 的 belief；
6. 不以 action flip 本身作为成功；flip 必须带来方向正确的 paired reward。

## 5. 实验阶段

### Stage 0：离线 trajectory replay

在现有 V4/V5 decisions trace 上重算 V6 shadow action，不调用模型：

- V6 vs V4 action flip；
- 被阻止的 probe 数量；
- 接受已观察正 utility deal 的恢复率；
- 结构性不可行 proposal rate；
- oracle shadow regret；
- 按 episode 1–5、6–10、11–15、16–20 分层。

进入在线测试的最低条件：

- 不增加结构性不可行 proposal；
- 在 continuous<frozen 的 setting 中减少低价值 probe；
- oracle belief 下不劣化 shadow objective；
- 单元测试覆盖每个 posterior outcome branch。

### Stage 1：独立 smoke（不复用正式主结果）

- 4 settings × 3 methods（V4、V5、V6）；
- 2 runs × 20 episodes；
- 新 seed；
- 输出到新的 `research_iterations/iteration_007.../runs/`；
- 当前正式目录严格只读。

### Stage 2：confirmatory 5×20

若 smoke 通过，运行：

- V6 continuous；
- V6 frozen；
- V6 wrong-confident；
- V6 shuffled；
- V6 oracle；
- V4 与 Opponent Simulation 使用正式冻结结果作为预注册 baseline；
- 如 model stochastic service 配置变化，则必须重跑 baseline，不得跨配置比较。

### Stage 3：opponent-switch

仅在 Stage 2 中 V6 的 correct belief 因果链成立后运行；主要指标为 controlled
difference-in-differences，而非 raw post−pre。

## 6. 主指标与成功标准

主指标全部按 5 个 run 配对：

1. focal reward delta vs V4、V5、Opponent Simulation；
2. agreement 与 joint utility（防止只把 surplus 抢到 focal）；
3. correct − wrong/shuffled reward contrast；
4. continuous − frozen reward contrast；
5. oracle headroom；
6. Brier、ECE、belief MAE/coverage；
7. decision-relevant action flip 与 realized regret；
8. controlled switch shock/recovery；
9. model calls、turns、format success。

V6 只有同时满足下列条件才进入论文主方法：

- 至少两个 action spaces/settings 的 paired reward 不劣于 V4，至少一个有稳定正效应；
- wrong/shuffled 明显损害 reward 或 realized decision value；
- continuous 相对 frozen 不再出现系统性反转，或能由预注册机制指标解释；
- calibration 不以降低 agreement/joint utility 为代价；
- improvement 不是额外 LLM calls 或 protocol repair 带来的；
- opponent-switch controlled effect 支持再推断，而非只支持自然学习。

## 7. 可能的论文 story

若 gates 与 V6 confirmatory 均成立：

1. 现有 opponent simulation/strategy reflection 主要优化语言策略；
2. 单独维护 opponent belief 不足，甚至更准的 belief 也可能因错误 VOI 造成更差决策；
3. 我们提出 continuous, calibrated belief，并首次把其用途约束为
   decision-relevant information value；
4. 通过 oracle/wrong/shuffled/frozen/action-flip 和 opponent-switch DiD 建立从
   belief quality → action change → realized utility/adaptation 的因果证据链；
5. 在 single-price 和 combinatorial resource negotiation 两种自然语言 action spaces
   上验证，而不是依赖某个 environment-specific price rule。

若 gates 不成立，应如实缩小 story：把贡献定位为 benchmark diagnosis 与 belief
causality evaluation protocol，而不是声称 planner 性能提升。
