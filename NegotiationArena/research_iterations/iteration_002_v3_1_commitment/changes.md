# Code changes

- 新增 `ReciprocalCommitmentBeliefPlannerAgent`，V2/V3.0 保持不变。
- 最近五个成功 episode 的 own reward 中位数减 1 作为 soft commitment floor。
- 若当前 offer 低于 floor 且存在满足 floor 的正 safeguarded-value candidate，则禁止立即接受并 counter。
- Resource opening 的 exploit family 从全 frontier 选取；其他 family 保留 V3 response-support safeguard。
- 新增 `framework_v3_1` / `framework_v3_1_frozen` CLI。
- 新增 commitment 与 opening-frontier 两项测试。

## Pilot evaluator audit

Resource/First episode 8 表面显示 focal `reward=-12`，但 decision trace 证明 focal 从未接受负效用 offer。根因是 `PaperRepeatedTradingGame.after_game_ends()` 在 `WAIT` 后收到 `ACCEPT` 时向全部历史反向搜索旧 Trade，导致 BLUE 接受了自己的旧 counteroffer；summary 又把无有效前序 trade 的 ACCEPT 误标为 agreement。

修复为：

- ACCEPT 只作用于紧邻上一手、且由另一 player 发出的 Trade；
- 若上一手是 WAIT/NONE，ACCEPT 不形成 agreement，reward 为双方 0；
- 新增 stale-own-counteroffer 回归测试；总测试数增至 29。

该问题影响所有历史 Resource 方法的少数 WAIT→ACCEPT 轨迹。旧目录保持不改，修复后结果必须写入带 `strict_acceptance` 的新目录；论文表格不能混用旧 evaluator 与新 evaluator。

## Pre-experiment test record

首次运行 28 项测试时，commitment test 将 `[16,16,15]` 的中位数减 1 错写为 14，实际应为 15；实现输出 15。该次 failed-test 未触发任何模型实验。随后只修正测试断言并重新验证，算法代码未改。
