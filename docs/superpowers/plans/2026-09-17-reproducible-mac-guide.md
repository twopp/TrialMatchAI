# 中文 Mac 可复现使用与发布文档实施计划

**目标：** 让新的使用者只依赖仓库内容，就能在 Apple Silicon Mac 上复现已经验证的 UIE、BGE-M3、Qwen MPS 和 DeepSeek 匹配流程。

**设计：** `docs/mac-chinese-setup.md` 承载完整中文教程，`config.mac.example.json` 提供不含密钥和机器路径的可复制配置，README 提供入口，`.gitignore` 阻止本地配置被误提交。现有主配置和 Linux 默认流程保持不变。

**设计说明：** `docs/superpowers/specs/2026-09-17-reproducible-mac-guide-design.md`

## 任务 1：提供安全的示例配置

- 新增 `config.mac.example.json`，使用当前验证过的 BGE-M3、UIE、Qwen Transformers/MPS 和 DeepSeek 设置。
- 将 `config.mac.json` 和 `.venv-uie/` 明确加入仓库忽略规则。
- 使用配置加载器验证示例配置。

## 任务 2：编写完整中文教程

- 新增 `docs/mac-chinese-setup.md`。
- 记录从克隆分支、安装主环境、创建 UIE 环境、下载和验证模型，到导入试验、匹配患者、查看报告的命令。
- 每段命令解释用途；覆盖 ClinicalTrials.gov 和自有规范化 JSON。
- 增加性能预期、隐私边界和常见错误排查。

## 任务 3：连接 README 并增加配置回归测试

- 在 README 快速入口和运行自有数据区域链接中文 Mac 教程。
- 新增测试，确保公开示例能够通过 `TrialMatchSettings` 校验，并固定关键后端和安全字段。

## 任务 4：验证、提交和发布

- 运行 JSON、链接、秘密信息和 Git 差异检查。
- 运行 Ruff 与完整 pytest。
- 提交文档与测试。
- 推送当前 `feature/deepseek-api` 分支到 `origin`，核对远程分支指向最新提交。
