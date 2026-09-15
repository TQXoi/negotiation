# 辅助环境集成代码

- `ANL_2024_2025/`：我们编写的 belief/planner/runner 代码，分别对接 ANL 2024 与 ANL 2025 原仓库；无自然语言交互，主要用于结构化 planner/oracle/frozen 对照。
- `LLM_Deliberation_integration/`：我们编写的 environment、roles、tool 与 P1 framework harness；必须在原 LLM-Deliberation source checkout 中或通过明确的 Python path 使用。该环境 role 名有偏好先验泄漏，不适合作为主论文的 interactive belief 证据。

这里**不包含完整第三方仓库**。原上游代码与 raw run 在 `../../negotiation_past/third_party/external_negotiation_envs/`；新机器需重新获取原论文仓库并固定 commit/license。迁移这两个辅助集成时先跑源论文 smoke，不能将本目录单独宣称为自包含 benchmark。
