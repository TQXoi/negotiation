# NegotiationArena Parser-Safe V2 说明

日期：2026-08-05

## 1. 问题根因

原始 `ExchangeGameDefaultParser.parse_proposed_trade()` 假定每个 `|` 分段都严格包含：

```text
Player <name> Gives <resource>: <integer>
```

并直接执行：

```python
resources = player.split("Gives")[1].strip()
```

Qwen 输出存在三类偏差：

1. 使用逗号代替 `|`；
2. closing tag 写成 `</newly proposed` 等不完整形式；
3. `<reason>` 太长，900 tokens 截断导致 trade 只剩 `Player RED`、`ZUP:` 等片段。

第三类是主要来源。原 prompt 允许任意长 reasoning，但 protocol action 位于 reasoning 后面，因而容易在 action 完成前耗尽 token budget。

此外，旧 `_protocol_valid()` 没有严格校验 Buyer/Seller 的 `PROPOSAL` trade grammar；Direct agent 与 opponent agent 又不经过该 guard，畸形动作最终进入原 parser 并产生 `IndexError`。

## 2. V2 修复原则

- 不修改原始 `negotiationarena/parser.py`，保留 paper/raw protocol；
- `--protocol-mode raw` 仍为默认值，旧实验语义不变；
- 新实验使用独立目录和 `--protocol-mode normalize_retry`；
- deterministic normalization 只能规范已明确出现的动作，不猜缺失 price/bundle；
- 无法恢复时最多进行一次低温 serialization retry；
- repair 的原输出、最终输出、hash、类型和额外调用全部落盘；
- 所有方法和 opponent 使用相同 repair policy。

## 3. 处理流程

```text
raw model response
  → strict protocol validator
  → deterministic canonicalization（若 price/bundle 唯一可恢复）
  → one low-temperature format-only retry（若仍无效）
  → original NegotiationArena parser
```

Parser-safe system reminder 还要求：

- `<reason>` 不超过 100 words；
- 所有 closing tags 完整；
- Buyer/Seller proposal 必须精确使用 `|`；
- 最后一个 required tag 后不再输出文本。

## 4. 新增记录

每个 episode 增加：

```json
{
  "focal_protocol": {
    "invalid_raw": 0,
    "deterministic_repairs": 0,
    "retry_calls": 0,
    "repair_failures": 0
  },
  "opponent_protocol": {}
}
```

模型调用保存于 `model_traces/**/calls.jsonl`；repair 对照保存于：

```text
model_traces/run_XX/Player_*/protocol_repairs.jsonl
```

`summary.json` 同时报告：

- `format_success_rate`；
- `all_episode_agreement_rate`；
- `all_episode_mean_focal_reward`（错误局按 0）；
- raw invalid、deterministic repair、retry 和 failure 总数；
- 原有 valid-only 指标。

正式比较应以 all-episode reward 为主，valid-only reward 只作诊断。

## 5. 运行命令

先只跑 Buyer Direct 的 20 局 parser pilot：

```bash
cd NegotiationArena

SETTINGS=buyer METHODS=direct RUNS=1 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/oppsim_parser_safe_v2_direct_pilot \
bash scripts/run_opponent_simulation_parser_safe_v2.sh
```

通过条件：

- `format_success_rate >= 0.95`；
- `protocol_repair_failures <= 1/20`；
- 所有 repair 均可在 `protocol_repairs.jsonl` 中审计。

然后运行三方法 Buyer pilot：

```bash
SETTINGS=buyer RUNS=1 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/oppsim_parser_safe_v2_buyer_n20 \
bash scripts/run_opponent_simulation_parser_safe_v2.sh
```

最后才扩大至正式设置：

```bash
SETTINGS="buyer seller resource_first resource_second" \
RUNS=10 EPISODES_PER_RUN=20 \
OUTPUT_ROOT=arena_runs/oppsim_parser_safe_v2_formal_n10x20 \
bash scripts/run_opponent_simulation_parser_safe_v2.sh
```

不要把 parser-safe V2 续跑到 `oppsim_qwen30b_formal_n10x20`；该目录必须永久保留为 raw V1。

## 6. 验证

```bash
cd NegotiationArena
PYTHONPATH=. python -m pytest -q \
  tests/test_repeated_integration_unittest.py tests/test_arena_integration.py
```

当前结果：`13 passed`。另有离线 end-to-end 检查确认 canonicalized response 能被原始 `BuySellGameDefaultParser` 解析为：

```text
RED:  X=1
BLUE: ZUP=59
```
