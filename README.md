# MaaKatana

MaaKatana 是一个基于 [MaaFramework](https://github.com/MaaXYZ/MaaFramework) 的个人自动化项目，用于自动操作《日本拼图 nonogram katana》中的部分游戏流程。

## 当前功能

- 启动游戏
- 进入公会并观看广告领取奖励
- 进入事务处并尝试全部收集
- 依次处理迫击炮、武士刀、木炭、木梁和钢铁制造流程
- 使用 OCR 判断制造数量，避免超过目标数量
- 使用自定义 Agent 完成建筑检查、数量路由和识别框点击

## 项目结构

- `assets/interface.json`：Maa UI 入口、任务名称和 Agent 配置
- `assets/resource/`：Pipeline、图片和其他运行资源
- `agent/`：Python 自定义动作与识别逻辑
- `tools/`：安装包生成和项目校验工具
- `.github/workflows/`：持续集成与自动发布配置

## 使用方式

本项目通过 MaaFramework 通用 UI 运行。请从 GitHub Releases 下载对应平台的安装包，解压后使用 MFAAvalonia 启动，并根据实际设备配置 ADB 连接。

当前主要入口：

| UI 任务 | Pipeline 入口 |
| --- | --- |
| 启动游戏 | `启动游戏` |
| 观看广告 | `进入公会` |
| 收集产物 | `事务处` |

## 开发与发布

本项目采用 MaaFramework ProjectInterface v2，并使用 GitHub Actions 自动检查、打包和发布。推送形如 `v0.1.0` 的 Git 标签后，发布工作流会生成各平台安装包。

发布前请确认 GitHub 仓库设置中的 Actions 权限为 `Read and write permissions`。

## 注意事项

- 本项目仅供学习和个人自动化使用。
- 请勿将账号信息、访问令牌、调试日志或其他敏感信息提交到仓库。
- 修改资源后应优先测试 `收集产物` 入口。
- MaaFramework 和 MFAAvalonia 的版本由发布工作流中的环境变量控制。

## 许可证

本项目使用 MIT License，详见 `LICENSE`。
