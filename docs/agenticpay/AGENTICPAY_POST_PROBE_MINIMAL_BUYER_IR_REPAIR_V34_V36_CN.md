# AgenticPay post-probe minimal buyer-IR repair：V34–V36

更新时间：2026-08-25

## 1. 当前方法

当 one-issue/non-crossing probe 后收到完整 seller counter 时：

1. 若 counter 对 buyer 已 IR，V33 latch 直接结算；
2. 若不 IR，V34 从 seller counter 出发，价格不变，只改变一个字段，以最小 buyer-utility gain 达到
   1% buyer reserve；每局一次；
3. 若字段是 day/month/minute 等计数，V35 将连续数学解沿 buyer-favorable 方向量化到自然整数单位；
4. 若 seller 紧接 repair 公开拒绝并明确提出 cost/risk/burden 和 compensation/price adjustment，V36
   把 buyer reserve 以上的 utility slack 向下量化为美分价格补偿，非价格条款不变，每局一次。

全过程只使用 buyer private utility 与 seller public behavior；不读取 seller hidden utility，不降低 buyer IR，
不修改 native seller、environment 或 scorer。候选由 trusted code 构造，planner selection validator 强制处理
这两个 phase-boundary action，renderer 锁定结构化合同。

## 2. 代码

- 核心 adapter/候选/validator/renderer：`negotiation/AgenticPay_Env/buyer/universal_framework.py`
- Variant 注册：`negotiation/AgenticPay_Env/buyer/variants.py`
- 实验 wiring：`negotiation/experiments/run_agenticpay_single28_framework.py`
- 代码级测试：`negotiation/tests/test_universal_framework.py`
- V34/V35/V36 原始实验、协议和 metrics：`universal_reward_research/iteration_034_*` 至 `iteration_036_*`

## 3. 最关键结果

六困难合同任务 fresh paired gate：V35 22.79、5/6 deal、1 timeout；V36 **26.88、6/6 deal、0 timeout**，
即 official native BuyerScore +4.09，0 mismatch、0 error。Task5 的自然单位 repair 与 Task10 的公开补偿
repair 均已成交。Task10 单项从 -2.73 提升到 21.83。

这证明 repair 在 targeted trajectory 上能把公开反应转换为 buyer-IR、可执行、可审计的后续动作；但
all-28 证明该机制尚未泛化。

## 4. All-28 最终验证

| Variant | BuyerScore | DealRate | TimeoutRate | Mismatch |
|---|---:|---:|---:|---:|
| CoT prompt | 23.57 | **78.57%** | **21.43%** | 8 |
| V36 | **25.95** | 71.43% | 28.57% | **3** |

总体 paired delta 为 +2.39，但 task-bootstrap 95% CI [-16.74, 20.34]。三项 price-only task
平均 +74.01；25 项 contract task 反而平均 **-6.21**，95% CI [-23.97, 10.51]，win rate 44%。
因此表面的总体提升主要来自 price-only task，不能证明 multi-issue planner 提升。

全量失败的主因是 phase state 陈旧：Task5、Task20 在卖方给出更新 counter 后仍反复发送 probe 后的旧
counter/repair；Task22 在卖方已明确接受 exact contract 后仍重复发送合同。三个成交 mismatch 还说明
公开接受文本不能替代 hidden seller-IR。V36 all-28 gate 失败，保留为审计版本，不晋升为当前最优通用
framework。下一版应做 rolling latest-counter、turn-id invalidation、one-shot latch 与 loop detector，而不是
继续调 concession 数值。

完整逐任务指标：
`universal_reward_research/iteration_036_publicly_compensated_post_probe_repair/metrics/ALL28_NATIVE_CONFIRMATION.json`。
