# Llumnix reproduction

本目录提供 `Llumnix-native`（真实 KV-cache live migration）和
`Llumnix-queue`（现有 KServe 兼容、只重排排队请求）两种模式。与 HPA 完全同构的
Llumnix-queue 三阶段启动命令见仓库根目录 `HPA_STYLE_START.md`；native 和扩展调试命令
见 `RUN_SHARESCALE_LLUMNIX.md`；机制说明和结果边界见
`REPRODUCE_TOKENSCALE_LLUMNIX.md`。
