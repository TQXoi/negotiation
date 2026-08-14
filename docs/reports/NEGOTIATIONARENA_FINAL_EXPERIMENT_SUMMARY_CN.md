# NegotiationArena 最终实验总结（2026-08-13）

## 完成情况

- 旧正式 baseline/framework matrix：36 cells，3600 episodes，0 current errors；
- 旧 opponent instance/policy switch：20 cells，2000 episodes，0 errors；
- V7 factorized confirmatory：32 cells，3200 episodes，0 errors；
- V8 reliability-gated confirmatory：28 cells，2800 episodes，0 errors；
- toward-prior private-preference switch pilot：6 cells，600 episodes，0 errors；
- away-from-prior identifiable switch/control：8 cells，800 episodes，0 errors。

本轮新增并完成 4200 个正式/因果 episode（V8 2800 + 两轮 private switch 1400），无 infrastructure error。所有模型调用、协议修复、interaction log、game state、framework decision、episode JSONL、summary 与 paired analysis 都保留在各 iteration 目录。

## 核心结果

V8 的最强结果是 Resource-first：reward 22.265，高于 V5 18.455；相对 V5 的五-run paired delta 为 +3.810，描述性 95% t interval [1.492, 6.128]。V8-wrong 为 18.660，显著避免了 V7-wrong 旧结果 7.0 的崩溃。

但 V8 在 Buyer/Seller 上分别只有 7.240/11.120，低于 V7 8.560/15.180；Resource-second learned belief 21.675 也不胜 frozen 25.125 或 shuffled 25.665。Stationary setting 中 continuous 不胜 frozen。

真实 preference switch 的最终 strict-control test 中，Resource-second V8 的自身 reward DiD 为 +4.480 [.070, 8.890]，但相对 no-cross 的额外 DiD 为 +6.830 [−8.845, 22.505]；同时 preference MAE 从 .386 上升到 .479。因此只有 payoff signal，没有正确 belief tracking 的机制证据。

## 最终方法定位

当前最可信的 novelty 不是“更强 negotiation prompt”，而是：

1. continuous、factorized opponent belief；
2. belief-conditioned executable planner 与 action lock；
3. evidence-provenance reliability gate；
4. same-state wrong/shuffled/oracle/frozen/no-cross interventions；
5. calibration、action flip、private-preference switch 和 paired DiD 组成的 causal evaluation protocol。

这套方法能够揭示 reward 提高是否真的来自正确 belief。当前结果表明 reliability-bounded use 已有证据，universal payoff 与 non-stationary belief correctness 尚未解决。

## 下一步

优先构建可辨识的 held-out utility profile suite 和 hypothesis-separating active probes；随后实现 regime-mixture/change-point belief，而不是继续调 planner reward weights。CaSiNo 用于验证 natural-language multi-issue preference ranking；NegotiationArena 用于可控因果干预；ANL 仅作为 structured control。

详细代码结构、全表和 paired 区间见 `NEGOTIATIONARENA_V8_FORMAL_RESULTS_AND_PREFERENCE_SWITCH_PLAN_CN.md`。
