# PaperBanana · RFS research-figure engine

[Original PaperBanana](https://github.com/dwzhu-pku/PaperBanana) / PaperVizAgent 的增强 Fork。保留 **Retriever → Planner → Stylist → Visualizer → Critic** 五智能体，以及原有 Vanilla、Polish、统计绘图和界面入口；新增与 [Research Figure Studio](https://github.com/douxl-lab/research-figure-skill) 的版本化互通接口。

## 科研图默认版式

与 RFS 的 [真实论文机制图规范](https://github.com/douxl-lab/research-figure-skill/blob/main/research-figure/references/journal-mechanism-style.md) 对齐：**独立图的标题和图号置于论文图注，图片内部不加 1–6 阶段编号或 A/B/C 面板字母，不堆叠彩色卡片和装饰性横幅**。需要详细解释的总体架构、路由机制与证据融合，默认分别绘制，按完整 FigureSpec 传递；仅在用户或期刊明确要求组合子图时使用 panel letters。所有计算符号必须有代码或公式依据，尤其不得套用未经验证的多类 Dirichlet 公式解释多标签预测。

## 新版：统一 FigureSpec v1

两端直接使用同一份 `schema_version: 1.0.0` 科学规范。模块、方向与类型明确的连接、分组、标签、公式、完整原始数据、来源和不可变约束不会在接口处缩减成提示词摘要。科学指纹保护这些字段；仅 `design` 允许在约束内优化。

`skill/exchange.py` 提供离线协议描述、请求校验、授权生成和产物校验。RFS 在调用前核对协议指纹，生成后将 **图片 + 同一 FigureSpec + 详细文字设计说明** 交接给原生重建后端。本仓库不直接生成可编辑 PPTX，也不声称脚本成功就证明图片科学正确。

## 安装

完整生成环境使用 Python 3.12；账号模型、额度和网络需自行配置。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python skill/run.py --help
```

只做协议交换/校验，无需模型 SDK 或凭据：

```bash
python -m pip install -r requirements-exchange.txt
python skill/exchange.py describe
```

模型能力来自 `configs/model_registry.yaml`。将模板复制到忽略的 `configs/model_config.yaml` 或使用所选 provider 的环境变量；密钥不得进入 FigureSpec、Prompt、仓库、日志或输出包。文档记录的模型不保证你的账号可用。配置优先级为显式 CLI → 环境变量 → YAML → 注册表默认；不因另一个 key 存在而静默切换服务。

## 与 RFS 配对使用

先在 RFS 创建 canonical request，再在本仓库运行：

```bash
python skill/exchange.py validate-request --request /project/new-request/request.json
python skill/exchange.py run --request /project/new-request/request.json --allow-external
python skill/exchange.py verify-bundle --spec /project/new-request/figure-spec.json --bundle /project/new-request/generated/bundles/CANDIDATE
```

`describe`、`validate-request`、`verify-bundle` 都不调用模型。`run` 要求显式允许发送当前科研材料及可能的付费调用，并复用原 CLI 与五智能体。替换 CANDIDATE 为实际候选目录名。已有生成目录不会被覆盖。

配对入口仅处理方法示意图；真实数值图由 RFS 的固定数据绘图路径处理。完整命令和恢复边界见 [Skill 运行参考](skills/paperbanana/references/exchange.md)。

## 与 RFS 的 TikZ 精密矢量路线适配

新增离线 `vector-handoff`：验证 PaperBanana 已生成候选图的文件哈希和科学指纹，向新目录导出原始 PNG、设计说明和完整、未篡改的 FigureSpec v1，供 RFS 选择 TikZ 本地矢量编译。PaperBanana 保留原有五智能体，不重复实现 TeX 编译；双方的 schema 和 validator 不变。

~~~bash
python skill/exchange.py vector-handoff --spec /project/F01.json \
  --bundle /project/request/generated/bundles/CANDIDATE \
  --out /project/vector-reference
python /PATH/research-figure/scripts/rfs.py run \
  --spec /project/vector-reference/figure-spec.json \
  --engine tikz --out /project/F01-tikz
~~~

交接命令**不调用外部模型，也不生成 TeX/PPTX**。RFS 独立生成 `.tex`、矢量 PDF/SVG 和 PNG；输出不会自动与 PaperBanana 位图实现像素级一致，仍须审核科研语义和实际版式。详见 [RFS TikZ 工作流](https://github.com/douxl-lab/research-figure-skill/blob/main/research-figure/references/tikz-workflow.md)。

## 独立生成及旧入口

```bash
python skill/run.py --content-file method.txt --caption "Scientific architecture overview" --task diagram --render-mode rfs --retrieval-setting none --num-candidates 2 --max-concurrent 2 --output output/figure.png --rfs-output-dir output/rfs
# 已有完整科学规范：追加 --rfs-file /project/figure-spec.json
python app.py
streamlit run demo.py
```

`--render-mode legacy` 保留旧图片契约。`--retrieval-setting none` 不下载参考集；`--offline` 仅禁止参考数据集下载，不会把 API 变成本地模型。恢复使用相同输入/配置与 `--resume --checkpoint-dir ...`；失败和超时不能标记为成功，服务端超时可能已经计费。

## 输出与编辑重建

成功候选包含 `manifest.json`、`image.png`、`prompt.txt`、`design.md` 和 `figure_spec.json`，可能另有原图、system prompt、原始输入规范及绘图代码。PNG 保留原始字节/透明度；失败候选可保留上一张有效图片，但状态仍为失败。

消费端同时检查文件哈希、实际 MIME 和原始科学指纹。哈希不是签名或科学真实性证明。`design.md` 是生成模型设计说明；后续必须观察实际图片，写出绑定图片与规范指纹的重建笔记，再由 paper-fig / Sivia / draw.io 执行。需分别核验原生对象与实际导出的渲染。

## 测试、安全与公开范围

```bash
python -m pip install -r requirements-dev.txt
python -m pip check
python -m pytest -q
python scripts/check_publication.py
python -m compileall -q agents utils rfs skill skills main.py app.py demo.py
```

协议回归使用真实导出/导入代码和合成图片，不调用付费模型；CI 运行完整软件测试。离线测试不证明真实生成质量、科研结论或原生 PPTX 可编辑性。统计图旧路径会执行模型生成 Python，超时/子进程不是安全沙箱；只在可信单用户隔离环境运行。详见 [SECURITY.md](SECURITY.md)。

源码、公开 README、许可证、Skill 运行参考可提交；密钥、IDE/Agent 私有配置、内部审查/开发状态文档、未发表论文数据和生成产物不可提交。`.gitignore` 不移除旧 Git 历史；本次清理不改写历史。

## Skill 与许可

标准入口：`skills/paperbanana/SKILL.md`；旧 `skill/SKILL.md` 保留同一运行规则供现有集成使用。安装标准目录时需要通过 `PAPERBANANA_ROOT` 指向实际引擎仓库，不能把一份 SKILL.md 当完整运行时。

遵循 Apache-2.0，保留 Google LLC 和上游作者声明。原始工作为 *PaperBanana: Automating Academic Illustration For AI Scientists*（Dawei Zhu 等）；本 Fork 的协议与工程增强不代表上游或模型提供商背书。不捆绑 Sivia、paper-fig 受限第三方材料或字体。
