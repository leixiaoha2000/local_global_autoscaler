# Local / Global Autoscaler

面向混合交互式与批处理大语言模型服务的局部和全局弹性管理实验项目。
代码包含 Kubernetes / KServe 部署配置、vLLM sidecar 控制器、Redis 角色管理、负载生成与基线比较。

## 项目内容

- `local_scaler/`、`local_scaler.py`：局部控制器和服务监控。
- `global_scaler/`、`global_scaler.py`：全局控制器及预热实例池管理。
- `compare/`：HPA、Knative、Chiron、TokenScale 与 Llumnix 相关基线和评测工具。
- `xiaorong/`：仅局部或仅全局控制的消融实验。
- `load/`、`shiyong/`、`draw/`：负载生成、小型工作负载 CSV 和绘图脚本。
- 顶层 YAML 与 `config/`：模型部署和资源配置。
- `HAMi/`、`llumnix-ray/`：随项目保留的第三方源码；其许可证见对应目录。

## 运行说明

各组件的依赖见对应目录中的 `requirements.txt`。
基线运行步骤见 [TokenScale 与 Llumnix 场景化复现说明](REPRODUCE_TOKENSCALE_LLUMNIX.md)、
[TokenScale 说明](compare/tokenscale/README.md) 和 [Llumnix 说明](compare/llumnix/README.md)。

运行前请按实际环境调整模型路径、服务地址、Kubernetes 命名空间和 GPU 资源配置。
GPU 服务实验需要自行准备 Kubernetes / KServe、GPU 环境及所需模型和数据。
