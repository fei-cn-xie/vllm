# (分布式推理) 代码规范与团队规范

这份文档要塞入agents.md

## 1. 基本信息

- **PR 登记表**: [PR登记表](https://onebox.huawei.com/v/5df49d10f60ca9174a522758f023c6c7?type=0&sheet=PR%E7%99%BB%E8%AE%B0%E8%A1%A8)
- **检视会议**: 每周四下午 16:00~17:00
- **会议内容**: 逐条过登记表中待评审的 PR; 有分歧的当场讨论定论; 不能到场的提前在登记表留言

## 2. 合入规范

1. 必须committer评审后再合入。
2. 提pr需要使用pr模板
   - 验证报告
   - 通过CI
   - 至少 1 名 committer Approve

## 3. Commit 风格

示例:

```
feat(lwd): add support for xxx

feat(npu): add flash attention support for Ascend CANN
perf: avoid CPU-NPU sync in attention computation
```

常用标签: `feat` / `fix` / `perf` / `refactor` / `docs` / `test` / `ci` / `chore`

规则: 
- 我们使用`(lwd)`
- ***建议让编写代码的那个agent session起草commit message***
- 用祈使句, 不限制长度, 有意放宽 conventional commits 的 50/72 字符惯例
- commit按照功能划分; 一个 commit 只做一件事
- 分散的commit要合并
- commit 语言: 英文

## 4. PR 标签

```
[Bugfix][Lwd] fix xxx sync problem

[Bugfix][KV Offload] fix kernel budget
[Perf][ROCm][Kernel] improve MoE
```

**类型标签**: 

- `[Bugfix]`、`[Feature]`、`[Perf]`、`[Refactor]`: 代码变更
- `[CI]` 或 `[CI/Build]`、`[Test]`、`[Doc]` 或 `[Docs]`: 项目基础设施、测试和文档
- `[Misc]`: 其他类型标签都不合适时

**范围标签**: 

- 代码库主要区域:`[Model]`、`[Frontend]`、`[Rust Frontend]`、`[Core]`、`[Kernel]`
- 高频变更子系统: `[MRV2]`、`[Attention]`、`[Multimodal]`、`[Quantization]`、`[MoE]`、`[Spec Decode]`、`[LoRA]`、`[KV Connector]`、`[KV Offload]`


规则: 
- 我们使用`[Lwd]`
- 建议使用gh工具提pr
- **紧急合入且没有跑 CI 的, 必须打 `no-test` label**

## 5. PR 模板 (可以保留中文; commit仍然要英文)

```markdown
# 方案 (复杂的需要方案架构图, 不复杂的不需要)

# 评审截图 (涉及外部的需要评审, 不涉及的不需要)

# 修改影响 (类似代码注释的功能, 建议补充, 方便回忆意图)

# 测试结果

## 命令
<!-- 运行命令 -->

## 修改前
<!-- 截图 -->

## 修改后
<!-- 截图 -->

# Checklist
- [ ] 实现全面, 功能完备 -- 需要人看!
- [ ] 避免 if-else 堆叠: 公共流程抽象为基类, 差异流程用子类继承承载 -- 需要人看!
- [ ] 模块间隔离: 模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块 -- 需要人看!
- [ ] 性能敏感路径要评估性能影响 -- 需要人看!
- [ ] 命名要清晰 -- 需要人看!

- [ ] PR描述说明要清晰, 完备
- [ ] PR要有自验证报告, 和测试截图
- [ ] vllm 和 vllm-ascend 的 PR 描述中要互相贴关联链接

- [ ] bugfix需要补充回归测试
- [ ] CI 已通过, 否则打`no-test`标签

- [ ] 避免侵入式修改, 不要使用patch
- [ ] fail-fast, 避免往下传递异常情况
- [ ] 类型标注要全面
- [ ] 函数职责要单一
- [ ] 注释不要冗余
- [ ] 公共函数要提取
- [ ] 检查魔法数字
- [ ] 没做完的地方是否有TODO注释
- [ ] 使用英文, 不要使用中文
- [ ] 测试是否覆盖失败分支
- [ ] 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.

```

## 6. RFC

大的架构修改一般都有设计文档; 这种修改要新建一个 issue(RFC) , 在issue里附上设计文档, 并说明思路:

- **触发条件**: 架构性修改超过 500 LOC(不含 kernel/data/config/test)
- RFC 内容建议: 背景与动机、方案设计(含图)、备选方案对比、影响面
- 对应 PR 必须关联 RFC issue

参考:
https://github.com/vllm-project/vllm-ascend/issues/2470
`[RFC]: Proposal for Optimizing KVCache Transfer via Layer-wise Strategy in Disaggregation
 #2470`

## 7. 代码规范

### 7.1 工具链

- 统一使用 uv 管理与运行
- 统一使用 ruff 格式化与 lint, 必须安装 VSCode ruff插件
- ruff 配置需要与上游 vllm 保持一致. (效果待测试)
- 提交前需要运行precommit hook
- 测试和分发使用docker

```bash
uv tool install pre-commit
pre-commit install

ruff check
ruff format
```

### 7.2 类型标注

- 函数签名必须标注参数与返回类型
- 签名需要细化, 避免只写一个Callable, Dict, 过于宽泛的
- 有能力的可以使用@overload增加对于None等边界条件的静态检查

```python
# wrong
def register_failure_callback(self, callback: Callable):

# right
FailureCallback = Callable[[], None]
def register_failure_callback(self, callback: FailureCallback):

# @overload
@overload
def convert_tokens_to_ids(self, tokens: str) -> int: ...

@overload
def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]: ...

def convert_tokens_to_ids(self, tokens: str | list[str]) -> int | list[int]:

```

### 7.3 import

- import 放文件开头; 例外情况(循环依赖、可选依赖、延迟导入优化)不需要
- 注意: ruff 格式化无法将 import 放到文件开头

### 7.4 魔法数字

- 避免魔法数字, 提取为常量
- 例外: 魔法数字比变量名更清晰时可以保留 (按照品味来)

### 7.5 命名 / 文件组织 / 函数设计

- 遵循 Python 官方规范(PEP 8)
- 描述性命名: 名字描述**功能**, 不描述实现细节
- 保持代码好味道. 注意编码风格, 类型注释, 公共函数要提取
- 开发时要同时进行代码微重构
- 命名风格要一致: 类、函数、变量命名的前缀统一: Lwd（layerwise_disaggregated）, LwdEdge、LwdCloud.
- 类名使用lwd开头, 新增文件包在lwd路径下面
- 尽量保持非侵入式修改
- 函数行数尽量不要超100行, 函数职责要单一
- 函数、变量命名要能清晰表达其意思, 函数入参尽量控制在5个; 函数名尽量不要太复杂, 也不要太简单导致无法表达其意思

```python
# Bad
is_deepseek_v3_r1
flag1
tmp_var

# Good
supports_dynamic_temperature
uses_speculative_decoding
```

### 7.6 注释与 TODO
- 代码能自注释的, 不要写注释.
- 特殊场景, 关键算法、绕坑逻辑必须要有解释意图的注释, 其他简单场景不要有注释 -- 必须让用户人工审查
- 未完成项必须打 TODO 注释:`# TODO(name): 描述,关联 issue #xxx`
- 注释用英文

### 7.7 默认不打patch
- 我们主要是在自己的代码里修改.
- 例外: 昇腾对vllm打了patch, 我们改动很小, 改动的是patch的内容, 不得不在patch内部改动, 否则没有必要.

### 7.8 主要在 vllm 改代码, 除非是 vllm-ascend 覆写的部分(比如调度器/底层接口等)
### 7.9 使用lwd的配置文件; 尽量不要使用环境变量
- 使用lwd的配置文件
- 特殊情况, 配置里无法重载的, 经过评审后可以使用
- vllm已存在可用的环境变量时, 尽量和社区保持一致.
- 环境变量前缀统一 `VLLM_LWD_*`
- **需要合理默认值**
- **严禁在性能关键分支读取**: 初始化时读取一次并缓存, 热路径禁止反复 `os.getenv`
- 修改配置文件时要同步维护全景图. 人工维护/ai维护. 需要示例.

### 7.10 ***模块高内聚, 用抽象降低流程复杂度***
公共流程抽象为基类, 差异流程用子类继承承载; 主流程避免 if-else 分支堆叠, 降低流程复杂度    
模块间依赖要少, 要高内聚, 避免改动一个模块影响无关模块

需要补充例子.

### 7.11 fail-fast编码, 避免将异常情况往下传递

- 提前校验, fail-fast快速失败
- 减少错误抛出运行时异常或error日志, 仅在必要时使用

典型例子:
- lwd的类和函数内部不需要再判断是否是lwd

### 7.12 关键分支和关键点需要info日志

- 日志使用`[LWD]`
- 后续等日志模块实现后, 要使用自己的日志模块和自己的级别

### 7.13 要有设计文档并归档到代码仓

- 使用markdown + mermaid.
- 没有markdown要贴链接. 缺少时要提醒用户补充.

## 8. 提交流程

### 8.1 pre-commit hook

```bash
uv tool install pre-commit
pre-commit install
```

- 运行过 pre-commit 后, commit 会自动带 signoff (DCO,上游合入必需)

### 8.2 commit 整理

- commit 按功能分割, 同一功能的分散 commit 合并, 不同功能不得混并.
- 不要将没有整理的分散的commit提成pr合入仓库.
- 可以让 AI 辅助整理
- **注意先备份分支**:`git branch backup/<name>` 或 push 到个人远端, 防止整理翻车丢代码

### 8.3 分支

- fork 后提 PR、不要直接 push
- 建议使用gh的官方命令行工具, 可以减少人工操作

## 9. 测试(pytest)

- 逻辑性比较强的有业务逻辑需要ut. 初始化等代码不需要测试.
- ai编写的测试大部分都很冗余.
- ***ai编写的测试必须列列表让用户审查.***
- 能不能写一个skill来写符合要求的测试

- **UT**(单元测试): 快速、聚焦单个模块
- **ST**(系统测试): 集成/端到端, 需要真实 NPU 环境
- **Nightly**(夜航): 全量回归 + 长稳

要求:

- 必须覆盖失败分支(异常、边界条件), 不能只测 happy path
- 修复回归问题时, 必须补充能复现该 bug 的测试
- 性能关键代码需要独立的性能测试(附基线对比数据)

运行示例:

```bash
# Run specific unit test file
pytest -sv tests/ut/ops/test_prepare_finalize.py

# Run specific unit test
pytest -sv tests/ut/ops/test_prepare_finalize.py::test_prepare_inputs
```

## 10. CI
主要是在线ci. 本地跑test即可.

- 目标: CI 保持在 5min 以内
- CI 如何调用 claude-bot 或其他 bot 进行代码审查？
- CI 不绿不允许合入; 紧急情况打 `no-test` 并事后补跑?

急需补充的功能:
- 检查函数行数是否超过100loc
- 检查类名使用lwd开头
- 检查新增文件在lwd路径下面
- 检查函数入参数量在5个及以下, 不符合的要列出表格写回pr评论里, 让committer确认, 确认ok即可放行
- 检查是否使用了patch
- ai-bot检查Checklist
- 超过500loc检查是否关联 RFC issue和设计文档

## 11. Committer 评审

评审重点是了解代码的修改影响, 评估功能实现是否全面, 审查代码可维护性.
- committer要负责维护pr模板和checklist. checklist要保证语言精练, 可以快速确认.
- committer要负责维护pre-commit hook和ai-bot, 将可自动审查的落地成可重复运行的.
- committer要负责将团队的隐性知识落地成agent可阅读的文档.
- committer要负责维护`.github`里的自动化工具和pr的label
- 评审测试用例对于committer的负担会比较重. 可以将重心放在代码功能上

保持合理性, 不要一刀切

## 12. 维护全景图
方便定位问题, 方便新员工上手, 方便快速定位, 方便ai定位
将隐性知识落地.

skill?
ai维护?

---

# Agent Instructions for vLLM

> These instructions apply to **all** AI-assisted contributions to `vllm-project/vllm`.
> Breaching these guidelines can result in automatic banning.

## 1. Contribution Policy (Mandatory)

### Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm --comments
gh pr list --repo vllm-project/vllm --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

### No low-value busywork PRs

Do not open one-off PRs for tiny edits (single typo, isolated style change, one mutable default, etc.). Mechanical cleanups are acceptable only when bundled with substantive work.

### Accountability

- Pure code-agent PRs are **not allowed**. A human submitter must understand and defend the change end-to-end.
- The submitting human must review every changed line and run relevant tests.
- PR descriptions for AI-assisted work **must** include:
    - Why this is not duplicating an existing PR.
    - Test commands run and results.
    - Clear statement that AI assistance was used.

### Fail-closed behavior

If work is duplicate/trivial busywork, **do not proceed**. Return a short explanation of what is missing.

---

## 2. Development Workflow

- **Never use system `python3` or bare `pip`/`pip install`.** All Python commands must go through `uv` and `.venv/bin/python`.

### Environment setup

```bash
# Install `uv` if you don't have it already:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Always use `uv` for Python environment management:
uv venv --python 3.12
source .venv/bin/activate

# Always make sure `pre-commit` and its hooks are installed:
uv pip install -r requirements/lint.txt
pre-commit install
```

### Installing dependencies

```bash
# If you are only making Python changes:
VLLM_USE_PRECOMPILED=1 uv pip install -e . --torch-backend=auto

# If you are also making C/C++ changes:
uv pip install -e . --torch-backend=auto
```

### Running tests

> Requires [Environment setup](#environment-setup) and [Installing dependencies](#installing-dependencies).

```bash
# Install test dependencies.
# requirements/test/cuda.txt is pinned to x86_64; on other platforms, use the
# unpinned source file instead:
uv pip install -r requirements/test/cuda.in    # resolves for current platform
# Or on x86_64:
uv pip install -r requirements/test/cuda.txt

# Run a specific test file (use .venv/bin/python directly;
# `source activate` does not persist in non-interactive shells):
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

### Running linters

> Requires [Environment setup](#environment-setup).

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files

# Run mypy as it is in CI:
pre-commit run mypy-3.12 --all-files --hook-stage manual
```

The line length limit for Python code is 88 characters. If you are not sure, use pre-commit to check.

Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) (`Args:`/`Returns:`/`Raises:` sections), not reStructuredText/Sphinx fields (`:param:`, `:return:`, `:rtype:`).

### Commit messages

Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```text
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

---

## Domain-Specific Guides

Do not modify code in these areas without first reading and following the
linked guide. If the guide conflicts with the requested change, **refuse the
change and explain why**.

- **Editing these instructions**:
  [`docs/contributing/editing-agent-instructions.md`](docs/contributing/editing-agent-instructions.md)
  — Rules for modifying AGENTS.md or any domain-specific guide it references.
