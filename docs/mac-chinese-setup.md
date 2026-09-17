# Apple Silicon Mac 中文模型方案使用教程

这份教程用于在 M 系列 Mac 上运行已经验证过的中文临床试验匹配组合：

| 环节 | 模型或服务 | 运行位置 |
| --- | --- | --- |
| 试验和患者的向量检索 | `BAAI/bge-m3` | 本地 CPU |
| 中文医疗实体抽取 | `uie-medical-base` | 本地 CPU，独立 Paddle 环境 |
| 入排条件相关性重排 | `Qwen/Qwen3-Reranker-0.6B` | 本地 Apple MPS |
| 最终入排资格评估 | DeepSeek API | 在线服务 |

UIE 只在试验库首次构建或试验内容改变时处理试验条件；日常匹配患者会复用已经保存的结果。BGE-M3 和 Qwen 在本机运行。启用 DeepSeek 后，候选试验的入排标准和患者摘要会发送到配置的 DeepSeek 服务。

本项目属于测试和研究工具，生成结果不是医疗决定。首次运行请使用合成或去标识数据；在传输真实患者数据前，请确认已取得适当授权并满足所在机构的隐私与合规要求。

## 1. 准备环境

需要：

- Apple Silicon Mac（M1 或更新）；
- Python 3.11；
- `git` 和 [`uv`](https://docs.astral.sh/uv/)；
- 建议至少保留 10 GB 可用磁盘空间；
- 一个有效的 DeepSeek API Key。

如果尚未安装 `git`、`uv` 和 Python 3.11，可以在已安装 Homebrew 的 Mac 上执行：

```bash
brew install git uv
uv python install 3.11
```

已经安装这些工具的电脑可以跳过这一步。没有 Homebrew 时，请按照 [`uv` 官方安装说明](https://docs.astral.sh/uv/getting-started/installation/)安装。

克隆包含中文 Mac 方案的分支：

```bash
git clone --branch feature/deepseek-api https://github.com/twopp/TrialMatchAI.git
cd TrialMatchAI
```

第一条命令下载代码并直接切换到目标分支；第二条进入项目目录。

安装主项目及 PyTorch、Transformers 等本地模型依赖：

```bash
uv sync --frozen --extra llm
```

`--frozen` 要求严格使用仓库中的 `uv.lock`，避免不同电脑解析出不同依赖版本。`--extra llm` 安装 BGE-M3 和 Qwen 所需的模型运行库。

确认 Mac 的 MPS 加速可用：

```bash
uv run python -c "import torch; print('PyTorch:', torch.__version__); print('MPS:', torch.backends.mps.is_available())"
```

最后应显示 `MPS: True`。如果是 `False`，Qwen 会回退到 CPU，仍可运行但速度会明显变慢。

## 2. 下载 BGE-M3 和 Qwen

```bash
uv run hf download BAAI/bge-m3
uv run hf download Qwen/Qwen3-Reranker-0.6B
```

两条命令分别把向量模型和条件重排模型下载到 Hugging Face 的本地缓存。下载中断后重复同一命令即可续传，不需要删除缓存重来。

可以这样确认缓存已经存在：

```bash
du -sh ~/.cache/huggingface/hub/models--BAAI--bge-m3
du -sh ~/.cache/huggingface/hub/models--Qwen--Qwen3-Reranker-0.6B
```

BGE-M3 大约占用 4.3 GB，Qwen3-Reranker-0.6B 大约占用 1.1 GB；实际大小可能随模型版本变化。

## 3. 创建 UIE 独立环境

Paddle 与主项目的 PyTorch 依赖分开安装，避免两个模型框架发生版本冲突：

```bash
uv venv .venv-uie --python 3.11

uv pip install --python .venv-uie/bin/python pip setuptools

uv pip install \
  --python .venv-uie/bin/python \
  --index-strategy unsafe-best-match \
  --extra-index-url https://www.paddlepaddle.org.cn/packages/stable/cpu/ \
  "paddlepaddle==3.3.0" \
  "paddlenlp==3.0.0b4" \
  "aistudio-sdk==0.2.6"
```

第一条命令创建专供 UIE 使用的 Python 3.11 环境。第二条补齐 PaddleNLP 导入时需要的基础安装工具。第三条从 PyPI 和 Paddle CPU 软件源选择兼容版本；`aistudio-sdk==0.2.6` 用于避免新版 SDK 与这个 PaddleNLP 测试版接口不兼容。

验证依赖：

```bash
.venv-uie/bin/python -c "import paddle, paddlenlp; print('Paddle:', paddle.__version__); print('PaddleNLP:', paddlenlp.__version__)"
```

看到 `Paddle: 3.3.0` 和 `PaddleNLP: 3.0.0b4` 即表示环境可用。`No ccache found` 和 `Setuptools is replacing distutils` 是警告，不是运行失败。

首次执行下面的中文抽取测试时，会自动下载 `uie-medical-base`：

```bash
.venv-uie/bin/python - <<'PY'
from paddlenlp import Taskflow

extractor = Taskflow(
    "information_extraction",
    schema=["疾病", "药物", "基因", "基因突变"],
    model="uie-medical-base",
    device_id=-1,
)
print(extractor("患者诊断为非小细胞肺癌，EGFR L861Q突变，拟使用奥希替尼。"))
PY
```

结果中应包含“非小细胞肺癌”“EGFR L861Q”或“奥希替尼”等实体。模型通常保存在 `~/.paddlenlp/taskflow/information_extraction/uie-medical-base/`。

## 4. 创建本地配置和 DeepSeek 密钥

复制仓库提供的公开示例：

```bash
cp config.mac.example.json config.mac.json
```

`config.mac.json` 已被 Git 忽略，可以根据本机目录修改而不会被误提交。默认配置使用 UIE、BGE-M3、Qwen MPS 和 DeepSeek，并把生成数据放在仓库的 `data/` 与 `results/` 下。

创建只保存在本机的密钥文件：

```bash
touch .env
chmod 600 .env
open -e .env
```

在打开的文件中只添加下面一行，替换成自己的密钥并保存：

```dotenv
DEEPSEEK_API_KEY=替换成自己的密钥
```

不要把密钥写进 `config.mac.json`，也不要提交 `.env`。如果不准备向在线服务发送任何患者内容，可以把 `config.mac.json` 中的 `rag.enabled` 改成 `false`；此时仍会完成检索和重排，但不会生成 DeepSeek 资格评估。

验证公开配置能够加载：

```bash
uv run python - <<'PY'
from trialmatchai.config.config_loader import load_config

config = load_config("config.mac.json")
print("实体抽取:", config["entity_extraction"]["backend"])
print("向量模型:", config["embedder"]["model_name"])
print("条件重排:", config["model"]["reranker_model_path"])
print("重排设备:", config["LLM_reranker"]["device"])
print("资格评估:", config["rag"]["backend"])
PY
```

预期分别显示 `uie`、`BAAI/bge-m3`、`Qwen/Qwen3-Reranker-0.6B`、`mps` 和 `deepseek_api`。

## 5. 准备临床试验库

### 方案 A：从 ClinicalTrials.gov 导入

先用少量记录试跑，不写入本地数据库：

```bash
uv run trialmatchai update-registry \
  --config config.mac.json \
  --keyword "non-small cell lung cancer" \
  --max-studies 10 \
  --dry-run
```

确认查询内容正确后，去掉 `--dry-run` 正式下载、规范化并更新索引：

```bash
uv run trialmatchai update-registry \
  --config config.mac.json \
  --keyword "non-small cell lung cancer" \
  --max-studies 10
```

ClinicalTrials.gov 原始响应保存在 `data/registry/raw/`，规范化试验保存在 `data/trials_jsons/`，运行记录保存在 `data/registry/runs/`。

### 方案 B：使用自己的试验 JSON

每个试验保存为一个 UTF-8 JSON 文件，放入 `data/trials_jsons/`。下面是最小实用示例：

```json
{
  "nct_id": "LOCAL-001",
  "brief_title": "EGFR 突变非小细胞肺癌研究",
  "official_title": "一项评估研究药物治疗 EGFR 突变非小细胞肺癌的研究",
  "condition": ["非小细胞肺癌"],
  "overall_status": "RECRUITING",
  "phase": ["PHASE3"],
  "sex": "ALL",
  "minimum_age": "18 Years",
  "maximum_age": "80 Years",
  "eligibility_criteria": "Inclusion Criteria:\n- 年龄18至80岁\n- 经病理确认的非小细胞肺癌\n- 存在EGFR L861Q突变\nExclusion Criteria:\n- 存在未控制的活动性感染"
}
```

`nct_id` 也可以是自己的安全试验编号，不要求必须以 `NCT` 开头。`eligibility_criteria` 是最重要的字段，应明确区分入选和排除标准。

放好文件后构建试验库：

```bash
uv run trialmatchai build \
  --config config.mac.json \
  --force-prepare \
  --reindex
```

这一步会让 UIE 抽取中文医疗实体、BGE-M3 生成向量并创建 LanceDB 索引。首次构建会加载模型，速度比后续增量更新慢。试验较多时，这是可能持续数小时的离线任务，不应在每次患者匹配前重复执行。

检查构建状态：

```bash
uv run trialmatchai build --config config.mac.json --status
```

## 6. 导入并匹配患者

FHIR JSON 示例：

```bash
uv run trialmatchai e2e \
  --config config.mac.json \
  --input /path/to/patient.fhir.json \
  --format fhir \
  --reingest \
  --rematch
```

把 `/path/to/patient.fhir.json` 替换为真实文件路径。`--reingest` 要求重新读取患者文件，`--rematch` 要求重新运行匹配；首次测试可以保留，日常重复运行时可以去掉以复用已有结果。

纯文本病历也可以运行：

```bash
uv run trialmatchai e2e \
  --config config.mac.json \
  --input /path/to/patient.txt \
  --format text \
  --reingest \
  --rematch
```

一次处理多个患者时重复提供 `--input`：

```bash
uv run trialmatchai e2e \
  --config config.mac.json \
  --input /path/to/patient-1.fhir.json \
  --input /path/to/patient-2.fhir.json \
  --format fhir \
  --reingest \
  --rematch
```

运行过程依次完成索引检查、患者导入、候选试验检索、Qwen 条件重排、DeepSeek 资格评估和报告生成。

## 7. 查看结果

默认输出结构如下：

```text
results/
├── <患者编号>/
│   ├── ranked_trials.json
│   ├── report.html
│   └── ...DeepSeek 评估中间文件
└── index.html
```

- `ranked_trials.json` 是机器可读的排序和评估结果；
- `<患者编号>/report.html` 是单个患者报告；
- `results/index.html` 是多患者汇总入口。

在 Mac 上打开汇总页面：

```bash
open results/index.html
```

如果只需要重新生成报告，不重新匹配：

```bash
uv run trialmatchai report --config config.mac.json --all
```

## 8. 离线运行和缓存

BGE-M3、Qwen 和 UIE 下载完成后，可以禁止 Hugging Face 在线访问：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
uv run trialmatchai e2e \
  --config config.mac.json \
  --input /path/to/patient.fhir.json \
  --format fhir
```

这个设置只让 Hugging Face 模型离线加载。只要 `rag.enabled=true`，DeepSeek 资格评估仍然需要网络。

## 9. 性能预期

在经过验证的 M4 MacBook Air（32 GB 统一内存）上：

- 29 条试验标准的首次 UIE 抽取和索引构建约 2 分 50 秒；
- 单名患者的 Qwen 条件重排约 1.5 至 2 分钟；
- DeepSeek 时间取决于网络、候选试验数量和 API 服务负载。

这些数字只用于估算，不是性能保证。试验数量、条件长度、内存和 Mac 型号都会影响耗时。

## 10. 常见问题

### `No module named 'setuptools'` 或 `No module named 'pip'`

重新补齐 UIE 环境工具：

```bash
uv pip install --python .venv-uie/bin/python pip setuptools
```

### `cannot import name 'download' from 'aistudio_sdk.hub'`

安装已经验证的 SDK 版本：

```bash
uv pip install --python .venv-uie/bin/python "aistudio-sdk==0.2.6"
```

### UIE 启动或抽取失败

检查 `.venv-uie/bin/python` 是否存在，并重新运行第 3 节的抽取测试。构建时 UIE 发生技术错误会明确记录警告，并根据示例配置回退到 `regex`；回退表示流程可以继续，不代表 UIE 已经正常工作。

### `No criteria documents found`

说明索引阶段没有找到处理后的入排标准。先确认 `data/trials_jsons/` 中存在有效试验 JSON，再执行：

```bash
uv run trialmatchai build --config config.mac.json --force-prepare --reindex
```

### `DEEPSEEK_API_KEY` 缺失

确认仓库根目录存在 `.env`，且包含非空的 `DEEPSEEK_API_KEY=...`。如果暂时不使用在线评估，将 `config.mac.json` 中的 `rag.enabled` 改成 `false`。

### Hugging Face 离线模式提示无法检查模型

如果模型已经下载，这类预检警告通常不影响从本地缓存加载。若随后出现找不到模型文件，先移除 `HF_HUB_OFFLINE=1` 和 `TRANSFORMERS_OFFLINE=1`，联网重新执行第 2 节下载命令。

### MPS 内存不足或 Qwen 运行失败

先关闭占用大量统一内存的应用，再重试。项目遇到不受支持的 MPS 运算时会尝试回退 CPU；也可以把 `config.mac.json` 中 `LLM_reranker.device` 改成 `cpu`，代价是匹配速度明显降低。

## 11. 更新代码

以后更新当前分支：

```bash
git switch feature/deepseek-api
git pull --ff-only origin feature/deepseek-api
uv sync --frozen --extra llm
```

更新后如果实体模型、向量模型或构建逻辑发生改变，请重新运行 `trialmatchai build --force-prepare --reindex`。不要删除 `.env`、`config.mac.json` 或模型缓存，除非确实要重建本地环境。
