# PaperBanana RFS：模型与 API 只读核查

核查时间：2026-10-10 06:57–07:02 UTC。源码基线：`836455537e863b5a2f40dace487a782c0bc5ef94`。这份文件是审查与适配建议，尚未实施源码修改。

## 结论

应将 provider、endpoint、能力和参数限制收敛到显式 registry，并让 agents 调用按任务定义的公共接口。不能从模型名称包含 `image` / `gpt-image`、前缀或哪个 key 存在来猜测能力或选择 provider。官方文档公开列出的支持与用户账号真实可用性是两回事。

优先提供官方稳定 ID 的配置候选：Gemini 文本 `gemini-3.8-flash`，图片 `gemini-nano-banana-2.1` 或保守兼容的 `gemini-3.1-flash-image`；OpenAI 图片 `gpt-image-2.5-sunburst` / `gpt-image-2.5-flare`（可 pin 2026-09-08 snapshot），同时保留 `gpt-image-2`、`gpt-image-1.5`、`gpt-image-1` 等显式 profile。现有 `gemini-3.1-pro-preview` 可继续作为明确的 preview 选项，不应把所有旧选择静默改写为新型号。

这些名称来自本次实际读取的官方页面，非未来型号猜测。并未测试任何用户账号上的调用、计费、额度或模型可见性。

## 1. 官方证据：模型精确字段

以下摘录为短字段片段；同一来源的英文直接引用总计不超过 25 words。其余为字段转述。

| 模型/主题 | 官方确切字段或短摘录 | 审查判断 | 来源与日期 |
|---|---|---|---|
| Gemini 3.8 Flash | `Stable: gemini-3.8-flash`；`Structured outputs Supported` | 接受文本与图片，输出文本；input 1,048,576 / output 65,536；可作 vision critic 和 JSON 规划模型。支持 low/medium/high thinking；minimal 不支持 | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash)，Latest update 为 September 2026；[退役表](https://ai.google.dev/gemini-api/docs/deprecations)列 release 2026-09-02 |
| Gemini 3.1 Pro Preview | `Preview: gemini-3.1-pro-preview`；`Image generation Not supported` | 接受图片并输出文本，structured output 支持；input 1,048,576 / output 65,536。不要将“可读图”误认为“能生图” | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-pro-preview)，Last updated 2026-10-09 |
| Gemini 3.1 Flash Image | `Stable: gemini-3.1-flash-image`；`Structured outputs Not supported` | 文本/图片输入，文本/图片输出；input 131,072 / output 32,768；支持图片生成与对话编辑 | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-image)，Last updated 2026-10-09 |
| Gemini Nano Banana 2.1 | `Model code gemini-nano-banana-2.1`；`Output token limit 32,768` | 当前官方高效图片型号；文本/图片输出，structured output 不支持；1K/2K/4K；至多14张参考图。支持 minimal/medium/high thinking，不应复制 3.8 文本型号的限制 | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-nano-banana-2.1)；[退役表](https://ai.google.dev/gemini-api/docs/deprecations)列 release 2026-10-06 |
| Gemini 3 Pro Image | `Stable: gemini-3-pro-image`；`Input token limit 65,536` | 图片生成/编辑；output 32,768；structured output 不支持。与 flash 图片型号不是同一上下文上限 | [模型页](https://ai.google.dev/gemini-api/docs/models/gemini-3-pro-image)，Last updated 2026-09-03 |
| 两个现有 image preview 默认 | 退役表相应两行 shutdown date 均为 2026-06-25 | `gemini-3.1-flash-image-preview`、`gemini-3-pro-image-preview` 不适合作为当前受支持默认；当前目录列 stable IDs。表的日期说明是 earliest possible retirement，未实测旧 endpoint 是否还返回 | [Google 退役表](https://ai.google.dev/gemini-api/docs/deprecations)，本次读取2026-10-10 |
| GPT Image 2.5 Sunburst | `Image Input and output`；`Structured outputs Not supported` | 直接 Images API 生成/编辑，也可作为 Responses 的 image_generation tool；文本仅输入。模型 ID 包括 `gpt-image-2.5-sunburst-2026-09-08` | [模型页](https://developers.openai.com/api/docs/models/gpt-image-2.5-sunburst)，本次读取2026-10-10 |
| GPT Image 1.5 / 2 | 两页分别列 `v1/images/generations` 与 `v1/images/edits` | 这些型号有官方文档支持，可以保留精确 profile；不能因是旧型号就删除已有功能 | [1.5模型页](https://developers.openai.com/api/docs/models/gpt-image-1.5)、[2模型页](https://developers.openai.com/api/docs/models/gpt-image-2)，本次读取2026-10-10 |
| GPT 6.1 Sol | `128,000 max output tokens`；`Structured outputs Supported` | 图片输入、文本输出；Responses/Chat Completions（无工具）可用。reasoning low/medium/high/xhigh/max；none/minimal不支持 | [模型页](https://developers.openai.com/api/docs/models/gpt-6.1-sol)，本次读取2026-10-10；官方页无可靠 release 日期字段，未推断 |
| GPT 4.1 | `32,768 max output tokens` | 现有一刀切 50,000 输出上限可能超限；保留时需精确限制 | [模型页](https://developers.openai.com/api/docs/models/gpt-4.1)，本次读取2026-10-10 |

## 2. 官方 API 与参数约束

### Gemini

- `client.aio.models.generate_content(model, contents, config)` 对应官方 `models.generateContent`。该 API 仍有独立官方 REST 参考；无需为这次升级强制把所有已有调用迁移到 Interactions。[generateContent参考](https://ai.google.dev/api/generate-content)
- 当前面向图片和 structured output 的官方教程已主要展示 `client.interactions.create`；输入使用 typed text/image blocks，响应有 steps 与 output_image 等便利属性。其格式不能直接塞入现有 generateContent adapter。[图片教程](https://ai.google.dev/gemini-api/docs/image-generation)、[结构化输出教程](https://ai.google.dev/gemini-api/docs/structured-output)
- 结构化输出支持 JSON Schema 子集，需要文本模型；应将 RFS schema 保持浅层、稳定且在本地验证。current generateContent REST reference 把 `responseMimeType` / `responseSchema` / legacy JSON schema 字段标为 deprecated，推荐 `responseFormat`；已有 SDK 兼容方案仍可按已 pin 的版本使用 legacy 字段，但不要把新 REST 字段名未经验证当 Python SDK 参数。[generateContent参考](https://ai.google.dev/api/generate-content)
- 图片 `image_size` 需 canonical `1K` / `2K` / `4K`，文档明确 lowercase 会拒绝。不同型号尺寸集合不同，不得把 4K 或 0.5K 当全系列能力。响应 modalities 必须与该模型支持组合精确匹配；不能假设每个图像模型都接受 IMAGE-only。候选数量需按模型 profile 校验，官方参数参考没有给所有 Gemini 统一“最多8个”的保证。[图片教程](https://ai.google.dev/gemini-api/docs/image-generation)、[generateContent参考](https://ai.google.dev/api/generate-content)
- 图片输出可能包括文本和中间思考图；应归一化全部结果、区分最终图像与 thought，保留真实 MIME/尺寸，不能盲选首个 inline_data 并声称它必定是最终图片。[图片教程](https://ai.google.dev/gemini-api/docs/image-generation)
- Gemini models.get/list 可辅助核对 supportedGenerationMethods、inputTokenLimit、outputTokenLimit 等，但不能据此证明账号可用、图片编辑能力或 JSON schema 的完整支持。[Models参考](https://ai.google.dev/api/models)

### OpenAI

- 对独立单图生成/编辑优先使用 `AsyncOpenAI.images.generate` / `.images.edit`；后者必须传原图或参考图。Responses 图像 tool 是可选的对话路径，上层文本模型和图片 tool 的 model 是两个独立配置项。不能把 GPT Image model 当 Chat Completions 文本模型。[官方图片指南](https://developers.openai.com/api/docs/guides/image-generation)
- 图片生成参数 n 为1–10，返回 GPT Image base64；不要传 legacy response_format=url/b64_json。png/jpeg/webp 为图片 output_format，不是文本 JSON schema；透明背景只能结合 png/webp。[Images generate参考](https://developers.openai.com/api/reference/resources/images/methods/generate)
- 编辑至多16张输入图，multipart 图片文件是 SDK 图片编辑的既有路线；当前参考还提供 JSON image references。不要在旧 SDK 上无验证地混用 image 与 images 两套参数。input_fidelity：1/1.5支持 high/low、mini只有low、2应省略；2.5不能未经证据继承旧字段。[Images edit参考](https://developers.openai.com/api/reference/resources/images/methods/edit)
- GPT Image 1/1.5 的常用尺寸为1024x1024、1536x1024、1024x1536；2/2.5支持满足约束的 WIDTHxHEIGHT：边长为16倍数、比例不超过3:1，仍有模型-specific像素/边长上限。2的指南列最大边3840、总像素655360–8294400；超过2560x1440的输出官方标experimental。不能把用户选的21:9或4K无提示改成固定1536x1024。[图片指南](https://developers.openai.com/api/docs/guides/image-generation)、[Images generate参考](https://developers.openai.com/api/reference/resources/images/methods/generate)
- 文本 structured output 在 Responses 使用 text.format、Chat Completions 使用 response_format 的 json_schema/strict 配置；schema 根对象、必需字段、additionalProperties=false需适配其支持子集。refusal、incomplete/length、无内容必须是独立结果状态，不能用 json_repair 当作成功。[Structured outputs指南](https://developers.openai.com/api/docs/guides/structured-outputs)
- 最新 GPT-6 官方指南要求在 reasoning effort 非 none 时去掉 temperature/top_p等；GPT-6.1-sol不支持none。因此当前“所有 OpenAI 请求都传 temperature”会出错。由精确 profile 控制这些参数。[GPT-6指南](https://developers.openai.com/api/docs/guides/latest-model)
- 模型是否可用取决于用户项目、区域/组织、tier和额度；图片指南提示可能需要组织验证。这些未实测，不应显示“已验证账号可用”。[图片指南](https://developers.openai.com/api/docs/guides/image-generation)

## 3. Timeout / retry：必须选定唯一责任层

- OpenAI SDK官方 README：默认重试2次；连接故障及408/409/429/5xx会重试；默认timeout10分钟，timeout参数以秒计。源码目前外层5 attempts叠加SDK默认重试，会让单样本产生更多请求和更长等待。[OpenAI SDK README](https://github.com/openai/openai-python)
- Google SDK `HttpOptions.timeout` 明确以毫秒计；`HttpRetryOptions.attempts`包括原请求，0或1表示不重试。types说明未指定attempts可为5，但 `_api_client.retry_args(None)` 使用不重试策略，因此不要脱离安装版本和options实际值说Google永远默认5次。[官方types源码](https://github.com/googleapis/python-genai/blob/main/google/genai/types.py)、[官方API client源码](https://github.com/googleapis/python-genai/blob/main/google/genai/_api_client.py)
- 建议应用层拥有retry：OpenAI构造客户端设置max_retries=0；Google显式HttpRetryOptions(attempts=1)。公共RequestPolicy用秒表示，Google adapter转换为毫秒。每次尝试 transport timeout与整个任务total_deadline分开；采用asyncio.timeout/wait_for并正确传播CancelledError。UI取消不能只让客户端等待结果而后台继续无限重试。
- 只重试瞬时连接/超时/限流/服务端故障，遵循Retry-After + capped exponential backoff + jitter。400参数/schema、401/403认证权限、404型号不存在、quota/billing、refusal/safety终止，不重复付费尝试。超时后是否服务端已生成无法保证，日志需标outcome_unknown；不要承诺exactly-once。多候选为独立sample，不要混淆retry与采样补齐。
- 上述SDK引用来自当前公开main文档，并非本地已安装版本的运行保证。环境只检测package metadata：google-genai、openai、anthropic均NOT_INSTALLED；httpx为0.28.1。requirements目前未pin这些SDK。实施应锁定兼容版本并对其参数签名做离线测试。

## 4. 已发现源码风险

基线行号如下，均为静态检查，未触发provider调用。

1. generation_utils.py:175–179依据image/nanoviz字串猜输出类型；:748–781依据前缀/已初始化key猜provider。增加OpenRouter key会让原生Gemini默认改路由，甚至把不明模型送往Anthropic。
2. generation_utils.py:798–804转换只提供max_completion_tokens，Claude helper :299读取max_output_tokens，统一Anthropic路径会在请求前KeyError。
3. visualizer_agent.py:126–165、vanilla_agent.py:96–133独立猜gpt-image。`openrouter/openai/gpt-image-*`也包含gpt-image，会先进入直连OpenAI分支并携带错误ID。
4. polish_agent.py:169–200和demo.py:191–282编辑路径没有统一OpenAI支持；后者OpenRouter输入为type/image/data/mime_type，但现有转换器只识别source/base64或image_base64，原图会被丢掉。
5. legacy_generation_options.py:24–53输出小写1k/2k/4k，直接进入GeminiImageConfig；应在adapter边界规范化，而不是破坏legacy UI/data。
6. visualizer/vanilla固定OpenAI尺寸1536x1024，不反映用户比例/分辨率；图片agents常用max_output_tokens=50000，超出多个Gemini图片型号32768上限。
7. config.py:53–75、template、demo.py:450–479重复硬编码模型；各配置读法ENV/YAML优先级不一致；demo预设残留退役preview。应由同一registry产生默认、提示和UI候选。
8. 广泛except Exception后重试并返回字符串Error：会掩盖永久参数错误、把错误当可用描述或base64继续传递；Gemini会原地修改config.candidate_count，产生共享状态副作用。
9. 文本候选收集按part数量而非candidate数量，可能把同一candidate的多个文字part计作多个sample；同时丢失finish_reason、usage、request_id、safety/refusal信息。

## 5. 建议公共合约（设计，未实施）

### Registry

`ModelProfile`包含：provider、exact_model_id、endpoint（gemini_generate_content / gemini_interactions / openai_responses / openai_chat / openai_images / anthropic_messages / openrouter_chat）、input/output modalities、operations（text/vision/structured_text/image_generate/image_edit）、schema mode、output/input token限额、采样/推理支持参数、图片尺寸/比例规则、response modality组合、候选上限、status、来源URL与核查日期。

统一 `resolve_model(model_ref, operation, overrides) -> ResolvedModel`：先读明确role/provider和精确registry项，再读用户显式覆盖。旧字符串支持确切alias映射，`openrouter/<vendor>/<id>`只作为显式provider语法，不据model substring推能力。未知模型需提供provider+endpoint+capabilities/profile；无法确定能力时提前给明确错误。不能根据key存在更换provider。用户自定义profile标user-declared，不能显示officially-verified。

默认策略：新建配置选择官方stable IDs；旧文件用户明确选择仍保留，退役项给迁移建议。缺key报该provider缺配置，不跨provider fallback。跨provider重试必须是用户配置的有界fallback链，并保存实际provider/model。

### 内容与结果

- `TextBlock(text)`、`ImageInput(bytes, mime_type)`为唯一内部形式。legacy source/base64、image_base64和demo data/mime_type先归一化，检测真实图像格式；禁止默默忽略未知block或伪写jpeg。
- `GenerationRequest(operation, model_ref, content, system_prompt, schema, sample_count, generation_options, policy)`不依赖Google SDK types。RFS roles（planner/critic/renderer/editor）明确需要能力。
- `GenerationResult(samples, provider, model_id, request_id, usage, finish_reason, elapsed, attempts)`；图片sample保存bytes+mime+width+height；JSON sample保存validated对象。Failure带category、retryable、safe_message。legacy wrappers可保留List[str]兼容，但内部不能用Error作成功内容。
- 公共 `generate_text` / `generate_structured` / `generate_image` / `edit_image`；visualizer/vanilla/polish/demo都走这组接口。schema处理是文本adapter职责，图片编辑原图必须传递；不能用图片模型补JSON规划。
- `validate_request`在外部请求前检查能力、尺寸、token、参数冲突。超限输出cap可明确policy下clamp并记录warning，不能默默失真；不支持的分辨率/比例应解释或要求用户明确允许最近支持值。
- 保留Anthropic、OpenRouter和已有Gemini ADC/Vertex路径作为单独provider/backend，不把OpenAI兼容API等同OpenAI模型能力。此次官方核查仅Google/OpenAI；Anthropic/OpenRouter已有能力应保留并用回归测试确认，未给未经官方核查的新能力背书。

## 6. 离线验收与后续真实验证

离线至少覆盖：显式provider不受key组合改变；substring不触发能力；精确alias与unknown override；Claude token映射；OpenRouter嵌套ID；三个legacy图片block无丢失；大写K；OpenAI生成与编辑分别调用且编辑带原图；Gemini modalities与输出上限；JSON refusal/length/invalid-schema；永久错误单次终止；SDK retry关闭；retry预算、timeout/取消传播；多candidate与多part归一化；无凭据、payload或原图泄露的日志；UI默认来源一致。

需另行使用用户授权的真实账号进行付费 smoke tests：文本vision、strict JSON schema、Gemini text-to-image与image-edit、OpenAI images.generate与images.edit、多输入图、尺寸/格式选项、实际rate limit/timeout、各候选型号项目可见性、组织验证及收费结果。这些本次全部未运行。未读取任何凭据内容或环境变量值，也未发起认证或生成请求。

## 实施环境补充

上述 NOT_INSTALLED 记录属于只读审查时的系统解释器，不能当作后续测试环境状态。升级验收另建 Python 3.12 隔离环境，实际安装 google-genai 2.29.0、openai 3.28.0、anthropic 1.13.0、Gradio 6.30.0。完整版本在 requirements-tested.txt，实际执行结果及实时调用边界见 docs/TEST_RESULTS.md。
