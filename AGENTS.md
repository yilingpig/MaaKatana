# MaaKatana Agent 交接指南

本文件是项目唯一的长期交接与执行规范。每个新 Agent 开始工作前必须阅读；发现新的稳定结论、运行路径或发布流程后，应在交付前更新本文件。

## 1. 强制规则

- 未经用户在**当前对话**中明确确认，禁止执行任何 `git push`、标签推送、Release、`git push --force` 或其他发布操作。
- 每次修改代码、Pipeline、脚本或配置后，必须自行 review：核对运行副本、关键分支、节点/模板引用、格式与差异校验。
- PowerShell 沙箱或补丁工具被拒绝时，不要反复重试；改用 Node.js 的“唯一匹配校验 → 临时文件 → 原子替换”流程。
- 仅处理用户要求的范围；不要擅自提交本地工具目录、调试产物或无关改动。
- 对需求明确、修改范围可控且不涉及 GitHub 发布的事项，直接修改并自行校验；不要只给操作建议后停下。
- 若同一诊断、工具或实现路径在较长时间内没有实质进展，主动切换思路（例如改查日志、换用替代工具、缩小问题范围或采用更直接的实现），不要在同一失败点反复消耗时间。

## 2. 当前快照（2026-08-14）

- 项目根目录：`F:\JUSTFORFUN\MaaKatana-main`。
- 游戏：《日本拼图 nonogram katana》；设备画面 `1280 x 720`；LDPlayer serial 通常为 `emulator-5554`。
- 当前准备发布：`v0.1.7`，包含迫击炮/武士刀路由修复、商队多城市派遣和汽船派遣流程。
- `v0.1.7` 发布内容应包含 `agent/condition_router.py`、对应 Agent 导入、Pipeline7、任务入口及商队/汽船新增模板；本地 MPE 目录和未引用模板不得纳入发布。
- `MaaPipelineEditor-v1.7.5-stable/` 是本地编辑器目录，保持未跟踪；除非用户明确要求，不得加入 Git。
- 旧交接文档已由本文件取代，不要恢复。

## 3. 关键路径与运行副本

- Pipeline 源文件：`assets/resource/pipeline/Pipeline7.json`。旧 `Pipeline6.json` 曾导致重复节点加载，不要恢复。
- VS Code MaaFramework Support 实际加载：`install/resource`，因此运行 Pipeline 为 `install/resource/pipeline/Pipeline7.json`。
- 主编辑点始终是 `assets/`；修改 Pipeline 后同步到 `install/resource/pipeline/Pipeline7.json`，修改图片后同步到 `install/resource/image/`。
- 只有确认 `F:\JUSTFORFUN\MFAAvalonia-v2.13.0` 是当前实际运行目标时，才额外同步它的 `resource`；不要盲目覆盖。
- 入口：`启动游戏`、`观看广告`、`收集产物`、`商队`、`汽船` 及三个汽船备用城市。`收集产物` 指向 `事务处`；`商队` 进入 `点击商队`；默认 `汽船` 选择纽约，雷克雅未克、墨西哥和墨尔本只通过独立入口手动选择，不影响主链。

## 4. Pipeline 主链与最近修复

- 主链：`事务处 -> 全部收集 -> 寻找迫击炮 -> 寻找武士刀 -> 木炭 -> 木梁 -> 钢铁 -> 船只派遣`。
- `事务处.next` 必须保持 `['全部收集', '寻找迫击炮']`。
- **候选识别节点的 `on_error` 不会用于未命中跳转。** 若节点放在父节点 `next` 中且带 `recognition`，未命中只会使父节点产生 `NextList.Failed` 并重试；该节点自己的 `on_error` 不会执行。
- 最新修复：`寻找迫击炮` 与 `寻找武士刀` 均必须是无 `recognition` 的 DirectHit 自定义节点，动作均为 `检查建筑并跳转`：
  - `寻找迫击炮`：检查 `工作室.png`；命中 `点击迫击炮`，未命中 `寻找武士刀`。
  - `寻找武士刀`：检查 `铁匠铺.png`；命中 `点击武士刀`，未命中 `向上滑动`。
- 迫击炮/武士刀路由修复已纳入 `v0.1.7` 发布范围。验证日志应出现 `寻找迫击炮` 的 `Action.Starting`，而非不断出现 `事务处 NextList.Failed`。
- 木梁、钢铁建筑连续 3 次未识别时，分别跳 `向上滑动寻找钢铁`、`开始船只派遣`；两处寻找滑动失败时也走同样兜底。
- `制造钢铁` Swipe：`begin [489, 412, 14, 23]`，`end [[774, 416, 13, 21]]`。

## 5. 船只派遣

- 链路：`开始船只派遣 -> 检查船 -> 点击船 -> 检查船界面 -> 点击勘探 -> 向上滑动寻找派遣 -> 检查派遣 -> 点击派遣 -> 点击商队 -> 检查商队界面 -> 罗马/雅典/西安选择与交易 -> 点击商队派遣 -> 检查汽船 -> 点击汽船 -> 检查汽船界面 -> 点击纽约 -> 向上滑到底汽船 -> 点击汽船派遣 -> 返回公会`。
- `开始船只派遣` 与 `向上滑动寻找船` 均固定大幅上滑：`[640, 650, 40, 40] -> [640, 90, 40, 40]`。
- `检查船` 连续失败 6 次跳 `点击商队`；点击船后没有 `船界面.png` 也跳 `点击商队`。
- 已删除 `商队` 路由节点；`点击商队` 直接使用 `商队.png` 识别并点击，失败跳 `检查汽船`。
- `检查商队界面` 使用 `商队界面识别.png` 判断是否进入界面；命中后跳 `点击罗马`，未命中跳 `检查汽船`。汽船图标或 `汽船界面识别.png` 未命中时统一执行 `返回公会`，避免派遣占用时反复重试。
- `点击罗马`、`点击雅典` 与 `点击西安` 都必须同时命中对应的 `FeatureMatch + SIFT` 图片和 OCR 文字，使用 `box_index: 1` 点击 OCR 框；雅典位于罗马上一行，西安位于第二行且与雅典相差三行。
- 罗马、雅典或西安无 `雷雨.png` 时，固定上滑到底，然后依次点击 `财宝.png` 两次、`金锭.png` 两次和 `派遣.png`；商品节点使用 `FeatureMatch + SIFT` 并限制在对应商品 ROI。
- 罗马、雅典或西安出现雷雨时，固定大幅上滑并用图片/OCR 检查“这座城市暂时停止交易”；罗马命中后改选雅典，雅典命中后改选西安，西安命中后进入 `检查汽船`。雷雨但未识别到停止交易时下滑回到底部继续交易。
- 汽船城市与商队城市保持同一识别写法：对应城市图片使用 `FeatureMatch + SIFT`，同时用 OCR 验证文字并点击 OCR 框；默认纽约，三个备用城市共享后续上滑和派遣节点。
- 勘探在船界面第四行。`点击勘探` 必须同时命中 `勘探.png` 和 OCR 文本 `勘探`；图片阈值 `0.9`，`box_index: 1` 点击 OCR 文本框，避免误点第一行“贸易”。
- 船只派遣通常超过 12 小时且难以撤销；未获用户明确许可时，不点击最终“派遣”或测试后续商队流程。
## 6. 自定义 Agent 模块

- `agent/main.py` 必须在 `AgentServer.start_up()` 前 import 自定义模块。
- `agent/building_router.py`：`检查建筑并跳转`。直接截图 TemplateMatch；命中跳 `hit_node`，未命中或异常跳 `miss_node`；支持 `max_misses`、`fallback_node`。
- `agent/condition_router.py`：`图文条件并跳转`。同一截图同时检查模板和 OCR，默认任一命中即可通过 `hit_node` 跳转，未命中走 `miss_node`。
- `agent/quantity_router.py`：`制造数量路由`。OCR 判断制造数量，通过 `context.override_next` 跳转。
- `agent/dynamic_swipe.py`：`动态向上滑动`，用于普通小步向上寻找。
- `agent/dynamic_bidirectional_swipe.py`：备用往返滑动模块，当前 Pipeline 未调用。
- `agent/recognition_click.py`：`点击识别框中心`；钢铁使用 `FeatureMatch + SIFT` 配合此动作。
- `agent/reward_reco.py`：奖励识别只接受完整“奖励已发放”或同为五字且最多一个字符不同的 OCR 结果；单独“奖励”不得通过。
- 不要覆盖 `agent/my_action.py`，除非用户明确要求。

## 7. 调试与日志

- 插件日志：`C:\Users\yilin\AppData\Roaming\Code\User\workspaceStorage\74d19155edcdc76695ff9bf0e22adcb3\nekosu.maa-support\mse.log`。
- 框架日志：`install/debug/maafw.log`。
- 插件截图/连接连续失败时：确认 LDPlayer 已完全启动，关闭 MPE/正式 UI 等其他控制端，再在插件中重连；旧 PID 常是根因。
- 地图业务方向通常是**向上滑动**。常规寻找优先使用动态小步上滑；不要把大幅固定上滑误用于所有节点。

## 8. 安装、发布与 GitHub Actions

- 仓库：`yilingpig/MaaKatana`。
- 发布 OCR 必须保留：`assets/MaaCommonAssets/OCR/ppocr_v6/small/`。
- `tools/install.py` 桌面端必须同时复制 `install/MaaAgentBinary`（MaaPiCli）和 `install/libs/MaaAgentBinary`（MFAAvalonia）。
- `.github/cliff.toml`：`owner = "yilingpig"`，`repo = "MaaKatana"`。
- `install.yml` 的 git-cliff 需要 `GITHUB_TOKEN`；changelog job 需要 contents/pull-requests 读取权限，release job 需要 contents 写入权限。
- 自动打包只在 `v*` 标签或手动触发时运行，构建矩阵固定为 `win + x86_64`，Artifact 保留 3 天；普通分支和 PR 只运行 `check.yml`，避免重复生成跨平台 Artifact。
- 修复发布工作流后：提交到 `main`，创建**新标签**，不要重跑旧标签。需要出现在 Release changelog 的提交不得包含 `[skip changelog]`。

## 9. 校验与交付清单

1. 编辑 `assets` 后同步 `install/resource`。
2. 解析 Pipeline JSON；检查 `next`、`on_error` 以及 `build_node`、`next_node`、`hit_node`、`miss_node` 的全部引用。
3. 对改动的 Python 文件执行 `python -m py_compile`。
4. 发布相关改动解析 workflow YAML、`cliff.toml`，并运行 `git diff --check`。
5. 核对真实日志、完成 review，并明确未验证的风险。
6. 未获当轮明确授权时，不提交、不推送、不打标签、不发布。

推荐命令（项目根目录）：

```powershell
.\tools\run_project_checks.ps1
# 快速检查：.\tools\run_project_checks.ps1 -SkipSchema
```

## 10. 新对话首轮操作

1. 阅读本文件并执行 `git status --short`，确认未提交的迫击炮/武士刀修复仍在。
2. 确认源 Pipeline 与 `install/resource/pipeline/Pipeline7.json` 一致。
3. 重载 VS Code MaaFramework Support 或重启 VS Code，运行 `收集产物`。
4. 日志预期：全部收集未命中后，`寻找迫击炮` 出现 `Action.Starting` 并显式跳转；不应再循环 `事务处 NextList.Failed`。
5. 修复确认并校验通过后，先向用户报告；只有获取新的明确授权才进行 Git 操作。
