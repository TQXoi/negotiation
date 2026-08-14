# Iteration 009 — V8 Reliability-Gated Factorized Belief

状态：**预注册；本文件创建时 V8 尚未实现或运行。**

## 1. 冻结证据

V7 confirmatory 已完成 32/32 cells、3200/3200 episodes、0 errors。

- Buyer/Seller continuous reward 不优于 frozen；
- resource-first V7 比 V5 五个 paired runs 全正；
- wrong-confident preference 在两个资源角色造成巨大 reward/joint collapse；
- shuffled preference 通常接近 V7/frozen；
- oracle 在 Buyer/resource-second 保留稳定 headroom；
- response-policy interventions 在相同状态下几乎不产生 action flip。

## 2. 方法假设

V7 learned posterior 不应在弱证据下直接控制 planner。V8 保留 joint
preference-policy posterior，但另外保存 episode anchor/meta posterior，并按证据可辨识性构造：

```text
b_safe(theta) = rho_theta * b_learned(theta) + (1-rho_theta) * b_anchor(theta)
b_safe(phi)   = rho_phi   * b_learned(phi)   + (1-rho_phi)   * b_anchor(phi)
```

- locked offer 的 observed ACCEPT/COUNTER/REJECT 是 direct response evidence；
- opponent-initiated offer 只弱增加 preference trust，不增加 policy trust；
- semantic claim 只作有界弱 preference evidence；
- `rho_phi` 只能由 direct response evidence 增长；
- 大 posterior drift 在证据不足时受到额外折扣；
- oracle preference 仅作为 diagnostic upper bound，可绕过 preference gate；
- wrong/shuffled stress 仍经过 safety gate，用于测量 mitigation，而非复现 V7 collapse。

所有权重对 Buyer/Seller/resource 相同，不读取 evaluator truth、角色名称或 realized test reward。

## 3. Stage 0 gate

1. 零 direct evidence 时 safe posterior 更接近 anchor 而非 learned point mass；
2. direct response 增长时 preference/policy trust 单调增加；
3. opponent offer 不增加 policy trust；
4. wrong confident 在低 trust 时被 anchor mixture 稀释；
5. oracle preference 可达到 point-mass upper bound；
6. action 合法且 non-negative self utility；
7. 不新增 LLM call；
8. 全部既有 integration tests 通过。

## 4. Stage 1 offline replay gate

在 V7 固定 trajectory 上只做 one-step action shadow：

- 记录 safe-vs-V7 与 safe-vs-anchor action flips；
- Buyer 至少出现非零保护性 flip；
- resource-first 不能所有决策退化为 anchor/frozen；
- 不使用旧 trajectory 的 realized reward 为未执行 action打分。

## 5. Stage 2 online smoke

新 seed；4 settings × (V5, V7, V8) × 2 runs × 20 episodes。通过条件：

- 至少两个 settings V8 reward 不低于 V7；
- Buyer 相比 V7 改善；
- resource-first agreement/joint 不发生明显 collapse；
- 至少两个 settings 有非零 reliability-gated action flip；
- 0 infrastructure errors。

通过后，仅在两个 resource settings 加 V8 wrong/shuffled stress。

## 6. Stage 3 confirmatory

若 smoke 通过，用新 seed 运行 5×20：V5、V7、V8、V8 frozen、V8 wrong、V8 shuffled、V8 oracle。

主 contrasts：V8−V7、V8−frozen、V8-wrong−V7-wrong、V8−oracle headroom，以及 gate→action→reward/joint mediation。

## 7. Opponent switch

只在 preference causality 已通过的 resource roles 实现真正 private-preference switch，并配 no-switch control。旧 `--opponent-switch-policy` 只切换 prompt/agent instance，不作为 preference switch。

## 8. 禁止事项

- 不按 setting 调 gate 参数；
- 不读取 true private parameter 控制 online action；
- 不新增 chooser/rollout LLM calls；
- 不以 calibration 替代 reward/joint；
- 不以 episode 当独立统计单位；
- 不因某个角色失败而事后删除。
