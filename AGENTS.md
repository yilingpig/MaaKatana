# AGENTS.md

本文件是 MaaKatana 项目的 Agent 工作指南。后续每次任务如果发现新的稳定约定、发布流程变化、调试结论或容易踩坑的问题，应同步更新本文件，保持它比零散交接文档更适合快速接手。

## 项目概览

- 项目根目录：`F:\JUSTFORFUN\MaaKatana-main`
- 正式 UI 目录：`F:\JUSTFORFUN\MFAAvalonia-v2.13.0`
- 框架：MaaFramework + MFAAvalonia 通用 UI
- 目标游戏：《日本拼图 nonogram katana》
- 模拟器：LDPlayer，ADB serial 通常为 `emulator-5554`
- 设备画面：`1280 x 720`，`assets/interface.json` 使用 `display_short_side: 720`
- 当前主要入口：`启动游戏`、`观看广告`、`收集产物`

## 重要目录

- `assets/interface.json`：Maa UI 任务入口、资源路径、Agent 启动配置。
- `assets/resource/pipeline/Pipeline7.json`：当前主 Pipeline 源文件。
- `assets/resource/image/`：源项目图片识别模板。
- `agent/`：Python 自定义动作、数量判断、动态点击逻辑。
- `tools/`：安装、配置 OCR、校验工具。
- 	ools/run_project_checks.ps1：Windows 下的统一校验入口，自动设置 UTF-8，运行 Python 编译、schema、资源同步和 git diff --check。
- 	ools/replace_text.ps1：补丁工具不可用时的安全文本替换入口，要求唯一匹配并使用临时文件原子替换。
- 若沙箱无法启动 PowerShell 或补丁工具被系统拒绝，立即改用 Node.js 原子读写/替换流程；先校验唯一匹配，再写临时文件并原子替换，避免反复重试。
- 每次完成可审查的代码、Pipeline、脚本或配置修改后，必须自行 review：核对实际运行路径、关键参数、预期分支和 diff/格式校验；确认无误后才向用户交付。
- `.github/workflows/install.yml`：GitHub Actions 打包发布工作流。
- `.github/cliff.toml`：git-cliff Release changelog 配置。
- `assets/MaaCommonAssets/OCR/ppocr_v6/small/`：发布时所需 OCR 模型文件。
- `install/`：本地运行/安装副本，不应作为主要源文件编辑。
- 桌面端打包时，`tools/install.py` 必须同时安装 `install/MaaAgentBinary`（MaaPiCli）和 `install/libs/MaaAgentBinary`（MFAAvalonia 类 UI）；不要只保留其中一个目录。

## 当前 Pipeline 状态

- 当前保留并发布 `Pipeline7.json`；旧 `Pipeline6.json` 曾造成节点重复加载，资源目录中不应再恢复它。
- `收集产物` UI 入口指向 `事务处`。
- 主流程为：`事务处 -> 全部收集 -> 寻找迫击炮 -> 迫击炮 -> 武士刀 -> 木炭 -> 木梁 -> 钢铁`。
- `事务处.next` 应保持 `['全部收集', '寻找迫击炮']`，用于“全部收集”按钮不存在时兜底。
- 木梁、钢铁建筑连续 3 次未识别到时，分别跳转到 `向上滑动寻找钢铁`、`开始船只派遣`；相应寻找滑动在到达边界而动作失败时也按同一路径兜底，避免流程终止。
- 钢铁流程包含：`检查钢铁建筑 -> 点击钢铁 -> 制造钢铁 -> 钢铁制造路由 -> 开始建造钢铁`。
- 钢铁制造完成后，`开始船只派遣` 与 `向上滑动寻找船` 均使用固定大幅 Swipe：从 `[640, 650, 40, 40]` 上滑至 `[640, 90, 40, 40]`，以快速进入船只区域。
- 钢铁制造完成后进入船只派遣：`开始船只派遣 -> 检查船 -> 点击船 -> 检查船界面 -> 点击勘探 -> 向上滑动寻找派遣 -> 检查派遣 -> 点击派遣`；`点击勘探` 必须同时命中 `勘探.png` 和 OCR 文字 `勘探`，图片阈值为 `0.9`，并使用 `box_index: 1` 点击 OCR 的“勘探”文字框，避免低分图片误匹配到第一行“贸易”。
- `检查船` 最多识别失败 6 次；之后跳转到当前为空的 `商队` 占位节点。点击船后未识别到 `船界面.png` 也跳转到 `商队`。
- `奖励已发放` 使用图片/OCR 的 `Or` 条件：图片识别失败时由 `agent/reward_reco.py` 做文字确认，先精确匹配 `^奖励已发放$`，再仅允许同为五个字且最多 1 个字符不同的 OCR 结果；单独“奖励”、少字或多字均不可继续跳过流程。
- `点击钢铁` 当前使用 `FeatureMatch + SIFT` 和自定义动作 `点击识别框中心`。
- `制造钢铁` 的 Swipe 坐标当前为 `begin: [489, 412, 14, 23]`，`end: [[774, 416, 13, 21]]`。

## 自定义 Agent 约定

- `agent/main.py` 必须在 `AgentServer.start_up()` 之前 import 自定义模块，否则动作可能未注册。
- `agent/quantity_router.py` 注册 `制造数量路由`，通过 OCR 判断制造数量，并用 `context.override_next` 跳转到 `build_node` 或 `next_node`。
- `agent/building_router.py` 注册 `检查建筑并跳转`，直接截图并使用 `TemplateMatch` 判断建筑，命中跳 `hit_node`，未命中或异常跳 `miss_node`；可选 `max_misses` 与 `fallback_node` 可在连续识别失败后跳转兜底节点。
- `agent/dynamic_swipe.py` 注册 `动态向上滑动`，使用固定直接坐标和较长时长执行小步上滑，并根据前后截图估算实际位移，下一次自动缩小或增大步长。
- `agent/dynamic_bidirectional_swipe.py` 注册备用动作 `动态往返滑动`：向上滑动达到边界或连续无位移后，向下回退若干步再重新向上寻找；当前 `Pipeline7.json` 暂不调用该模块。
- `agent/recognition_click.py` 注册 `点击识别框中心`，点击当前识别框中心。
- 不要覆盖 `agent/my_action.py`，除非用户明确要求。

## Windows 工作流约定

- 优先从项目根目录调用 `.	oolsun_project_checks.ps1`，不要在父目录直接运行相对路径校验命令。
- PowerShell 向外部命令传递多个路径时使用数组或参数展开，例如 `@($path1, $path2)`；不要使用容易被解析成单个字符串的逗号表达式。
- 运行 Python 工具前由统一脚本设置 `PYTHONIOENCODING=utf-8`，避免 schema 校验的 Unicode 符号触发 GBK 编码错误。
- `apply_patch` 若因 Windows 会话出现 `Access is denied`，不要继续反复尝试，也不要直接覆盖文件；改用 `tools/replace_text.ps1`，并要求旧文本精确匹配一次。
- 每次文件修改后先运行 `.	oolsun_project_checks.ps1 -SkipSchema` 做快速检查，需要交付前再运行完整校验。
## Pipeline 修改规则

- 修改 Pipeline 时优先编辑 `assets/resource/pipeline/Pipeline7.json`，并同步到：
  - `F:\JUSTFORFUN\MaaKatana-main\install\resource\pipeline\Pipeline7.json`
  - `F:\JUSTFORFUN\MFAAvalonia-v2.13.0\resource\pipeline\Pipeline7.json`
- 修改图片资源时同步三个目录：
  - `F:\JUSTFORFUN\MaaKatana-main\assets\resource\image`
  - `F:\JUSTFORFUN\MaaKatana-main\install\resource\image`
  - `F:\JUSTFORFUN\MFAAvalonia-v2.13.0\resource\image`
- 删除节点前必须检查所有引用：`next`、`on_error`、自定义参数中的 `build_node`、`next_node`、`hit_node`、`miss_node`，以及组合识别里的字符串引用。
- 不要仅凭节点名判断重复；先确认运行时资源是否同时加载多个 pipeline 文件。
- MaaFramework 的普通 `on_error` 对自定义动作 `override_next` 后的分支不总是可靠，重要分支应使用自定义动作显式跳转。
- 修改正式 UI 资源后通常需要重启 `MFAAvalonia.exe`，否则可能继续使用旧资源哈希。

## Git 推送授权

- 未经用户在当前对话中明确确认，禁止执行任何 `git push`、`git push --force`、`git push origin <tag>`、创建并推送发布标签，或其他会将本地内容发布到 GitHub 的命令。
- 可以在未推送的前提下检查 Git 状态、查看差异、检查远程引用和准备提交；创建 Git commit 仍须遵守更高优先级的用户与系统指令。
- 当用户明确确认推送时，先说明将推送的分支、标签、提交范围和可能触发的 GitHub Actions，再执行对应命令。
- 本项目后续所有 Agent 都必须先读取并严格遵守本文件；如本文件与用户、系统或开发者指令冲突，以更高优先级指令为准。
## 发布与 GitHub Actions

- GitHub 仓库：`https://github.com/yilingpig/MaaKatana`
- `assets/MaaCommonAssets/OCR/ppocr_v6/small/` 必须提交到 GitHub，否则 Actions 会报 `File Not Found: .../assets/MaaCommonAssets/OCR`。
- `.github/cliff.toml` 的 `[remote.github]` 必须保持：`owner = "yilingpig"`，`repo = "MaaKatana"`。
- `install.yml` 中 `git-cliff` 步骤需要 `GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}`，否则 GitHub API 可能因未认证而 403 限流。
- `changelog` job 需要 `permissions: contents: read` 和 `pull-requests: read`。
- `release` job 需要 `permissions: contents: write`。
- 不要重跑旧 tag 来验证 workflow 修复；旧 tag 指向旧提交。修复 workflow 后提交到 `main`，再创建新的 tag。
- 示例发布：`git tag -a v0.1.4 -m "Fix changelog permissions"`，然后 `git push origin v0.1.4`。

## 推荐校验入口

```powershell
Set-Location -LiteralPath 'F:\JUSTFORFUN\MaaKatana-main'
.\tools\run_project_checks.ps1
```

快速检查可使用：

```powershell
.\tools\run_project_checks.ps1 -SkipSchema
```
## 常用校验

```powershell
Get-Content -Raw -LiteralPath 'F:\JUSTFORFUN\MaaKatana-main\assets\interface.json' | ConvertFrom-Json
Get-Content -Raw -LiteralPath 'F:\JUSTFORFUN\MaaKatana-main\assets\resource\pipeline\Pipeline7.json' | ConvertFrom-Json
python -m py_compile `
  'F:\JUSTFORFUN\MaaKatana-main\agent\building_router.py' `
  'F:\JUSTFORFUN\MaaKatana-main\agent\quantity_router.py' `
  'F:\JUSTFORFUN\MaaKatana-main\agent\recognition_click.py' `
  'F:\JUSTFORFUN\MaaKatana-main\agent\main.py'
python 'F:\JUSTFORFUN\MaaKatana-main\tools\configure.py'
```

发布相关校验：

```powershell
git diff --check
python -c "import yaml; yaml.safe_load(open(r'F:\JUSTFORFUN\MaaKatana-main\.github\workflows\install.yml', encoding='utf-8')); print('yaml-ok')"
python -c "import tomllib; tomllib.load(open(r'F:\JUSTFORFUN\MaaKatana-main\.github\cliff.toml','rb')); print('cliff-toml-ok')"
```

## 调试重点

- 使用正式 UI 的 `收集产物` 入口测试主链路。
- 观察 `点击钢铁` 日志是否出现 `algorithm=FeatureMatch`。
- 确认 `点击识别框中心` 点击坐标是否落在实际钢铁图标位置。
- 如果点击后没有进入钢铁制造窗口，检查 `点击钢铁.png` 是否截取了真正可点击图标，并关注 FeatureMatch 是否选错候选框。
- 若出现“大幅滑动”，从日志中的触摸坐标判断来源：建筑寻找约 100 像素纵向滑动，制造按钮约 280 像素横向滑动，广告寻找约 490 像素纵向滑动。
- 2026-08-07 调试日志中旧纵向 Swipe 配置约 100 像素，但实际记录出现 `-201`、`-227`、`-247`、`-262` 像素位移，且时长固定为 200ms；当前业务方向仍是向上，纵向寻找节点统一使用 `动态向上滑动`，避免继续使用矩形 Swipe 的超幅输入。

## 维护本文件

- 后续工作若改变 Pipeline 主流程、Agent 注册、发布流程、关键路径、校验命令或已确认的调试结论，应主动更新本文件。
- 若发现本文件与实际代码不一致，以实际代码和最新日志为准，并立即修正文档。
- 更新本文件时保持简洁，记录稳定结论，不记录一次性猜测。
