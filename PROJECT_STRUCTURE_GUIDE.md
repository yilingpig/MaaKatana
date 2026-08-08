# MaaKatana 项目结构解析

这份文档用于帮助你理解当前 MaaFramework 模板项目的组织方式，并作为创建自己项目时的参考。

## 一句话结论

这个项目不是传统意义上的前端或后端应用，而是一个 MaaFramework 衍生项目模板。

它的核心不是 `package.json`，也不是 `agent/main.py`，而是：

```text
assets/interface.json
assets/resource/pipeline/*.json
agent/*.py
```

也就是：

```text
项目外壳配置 -> 自动化任务流 -> 可选的 Python 自定义逻辑
```

## 根目录结构

```text
MaaKatana-main
├─ .github/              GitHub Actions、Issue 模板、发布配置
├─ .vscode/              VSCode 项目配置
├─ agent/                Python 自定义识别和动作扩展
├─ assets/               项目接口、资源、任务流、图片、OCR 模型
├─ deps/                 MaaFramework 相关 schema 和运行依赖占位
├─ docs/                 中文开发文档
├─ tools/                安装、配置、校验脚本
├─ package.json          Node 工具依赖，不是项目运行入口
├─ README.md             模板说明
└─ LICENSE
```

## 关键目录说明

### 1. `assets/`

这是最重要的目录，决定项目对外展示什么、有哪些任务、任务如何执行。

```text
assets
├─ interface.json
└─ resource
   ├─ pipeline
   │  └─ my_task.json
   ├─ image
   │  └─ empty.png
   └─ model
      └─ ocr
         ├─ det.onnx
         ├─ rec.onnx
         └─ keys.txt
```

#### `assets/interface.json`

这是项目总入口配置，主要定义：

- 项目名称
- 项目描述
- 联系方式
- 许可证
- 控制器类型，例如 Android ADB、Win32 桌面窗口
- 资源路径
- UI 中展示的任务列表
- 每个任务对应的 pipeline 入口
- 任务选项

如果你要做自己的项目，应该优先从这个文件开始改。

#### `assets/resource/pipeline/my_task.json`

这是任务流配置文件，定义具体任务节点。

当前示例里有：

```json
{
    "MyTask1": {},
    "MyTask2": {},
    "MyTask3": {},
    "MyTask4": {
        "recognition": "Custom",
        "custom_recognition": "my_reco_222",
        "action": "Custom",
        "custom_action": "my_action_111"
    },
    "OcrTask": {}
}
```

其中 `MyTask4` 演示了如何接入 Python 自定义识别和动作。

### 2. `agent/`

这个目录负责 Python 自定义逻辑。

```text
agent
├─ main.py
├─ my_action.py
└─ my_reco.py
```

#### `agent/main.py`

这是 AgentServer 启动入口。

它做三件事：

- 初始化 MaaFramework Toolkit
- 启动 AgentServer
- import 自定义 action 和 recognition 模块，让装饰器注册生效

通常不需要大改这个文件。新增自定义模块时，需要在这里 import。

#### `agent/my_reco.py`

这是自定义识别示例。

核心注册代码是：

```python
@AgentServer.custom_recognition("my_reco_222")
class MyRecongition(CustomRecognition):
    ...
```

这个 `"my_reco_222"` 必须和 pipeline 中的 `custom_recognition` 对应。

#### `agent/my_action.py`

这是自定义动作示例。

核心注册代码是：

```python
@AgentServer.custom_action("my_action_111")
class MyCustomAction(CustomAction):
    ...
```

这个 `"my_action_111"` 必须和 pipeline 中的 `custom_action` 对应。

### 3. `tools/`

这个目录不是业务逻辑，而是开发和发布辅助脚本。

```text
tools
├─ configure.py
├─ install.py
├─ requirements.txt
└─ validate_schema.py
```

#### `tools/configure.py`

用于配置 OCR 模型。

它会尝试从：

```text
assets/MaaCommonAssets/OCR
```

复制默认 OCR 模型到：

```text
assets/resource/model/ocr
```

注意：当前仓库里的 `assets/MaaCommonAssets` 是空目录，这说明它依赖 git submodule 或外部资源。

#### `tools/install.py`

用于打包安装。

它会把以下内容复制到 `install/`：

- MaaFramework 运行依赖
- `assets/resource`
- `assets/interface.json`
- `agent/`
- `README.md`
- `LICENSE`

#### `tools/validate_schema.py`

用于校验 JSON 和 JSONC 配置是否符合 schema。

它会校验：

- pipeline 配置
- interface 配置
- 可选的 task 配置

### 4. `deps/`

当前主要放 schema 文件：

```text
deps/tools
├─ pipeline.schema.json
├─ interface.schema.json
├─ interface_config.schema.json
├─ interface_import.schema.json
├─ custom.action.schema.json
└─ custom.recognition.schema.json
```

这些文件用于约束 `assets` 里的配置格式。

### 5. `.github/workflows/`

这里是 CI 和发布流程。

主要文件：

- `check.yml`: 校验资源和 schema
- `install.yml`: 下载 MaaFramework、打包项目、上传 artifact、发布 release
- `sync_schema_files.yml`: 同步 schema
- `mirrorchyan_release.yml`: Mirror 酱发布相关
- `mirrorchyan_release_note.yml`: Mirror 酱 release note 相关

如果你只是本地开发，前期不用急着改这里。

等你要发布自己的项目时，再重点修改：

- 包名
- release 名称
- GitHub 仓库地址
- 是否启用 Mirror 酱
- 是否打包通用 UI

## 运行关系

整体关系可以理解为：

```text
用户在通用 UI 中选择任务
        |
        v
assets/interface.json 找到 task entry
        |
        v
assets/resource/pipeline/*.json 找到任务节点
        |
        v
MaaFramework 执行识别和动作
        |
        v
如果节点使用 Custom，则调用 agent/*.py
```

示例：

```text
interface.json
└─ task entry: MyTask4

pipeline/my_task.json
└─ MyTask4
   ├─ custom_recognition: my_reco_222
   └─ custom_action: my_action_111

agent/my_reco.py
└─ @AgentServer.custom_recognition("my_reco_222")

agent/my_action.py
└─ @AgentServer.custom_action("my_action_111")
```

## 建立自己项目的推荐顺序

### 第一步：复制模板并改基础信息

优先修改：

```text
README.md
assets/interface.json
LICENSE
.github/ISSUE_TEMPLATE/*
```

把里面的模板名、仓库地址、联系方式、项目描述改成自己的。

### 第二步：确认资源依赖

检查：

```text
assets/MaaCommonAssets
assets/resource/model/ocr
```

如果 `assets/MaaCommonAssets` 为空，需要初始化 submodule 或手动准备 OCR 资源。

### 第三步：设计任务入口

在 `assets/interface.json` 里定义你的任务列表。

示例思路：

```json
{
    "name": "每日任务",
    "entry": "DailyTask"
}
```

这里的 `DailyTask` 要在 pipeline 文件里存在。

### 第四步：编写 pipeline

在：

```text
assets/resource/pipeline/
```

新建或修改任务流文件。

建议从最小任务开始：

```json
{
    "DailyTask": {}
}
```

先让任务能被识别到，再逐步增加识别、点击、OCR、条件分支等逻辑。

### 第五步：只有需要时再写 Python 扩展

如果普通 pipeline 配置无法满足需求，再写：

```text
agent/my_reco.py
agent/my_action.py
```

典型使用场景：

- 复杂图像判断
- 多步动态逻辑
- 需要调用上下文 API
- 需要根据识别结果动态改变后续任务

### 第六步：校验配置

改完配置后建议运行 schema 校验。

CI 中使用的是：

```bash
python tools/validate_schema.py --schema-dir deps/tools --resource-dirs assets/resource --exclude-dirs assets/resource/announcement --interface-files assets/interface.json
```

### 第七步：准备发布

发布前再处理：

```text
.github/workflows/install.yml
maatools.config.mts
package.json
```

重点确认：

- artifact 名称
- GitHub release 名称
- MaaFramework 版本
- MFAAvalonia 版本
- 项目名是否还残留 `MaaXXX`

## 最小改造清单

如果你只是想快速做一个自己的项目，最少需要改这些文件：

```text
README.md
assets/interface.json
assets/resource/pipeline/my_task.json
agent/my_action.py
agent/my_reco.py
```

其中 `agent/*.py` 不是必改项。只有 pipeline 用到 custom recognition 或 custom action 时才需要改。

## 不建议优先修改的内容

前期不建议一上来就改：

```text
tools/install.py
tools/validate_schema.py
deps/tools/*.json
.github/workflows/*
```

这些属于工具链和发布链路。除非你已经明确知道要改打包方式、schema 规则或 CI 行为，否则先保持默认。

## 重点记忆

这个模板的核心链路是：

```text
interface.json 负责让任务显示出来
pipeline/*.json 负责定义任务怎么跑
agent/*.py 负责补充 pipeline 做不了的逻辑
tools/*.py 负责配置、校验和打包
.github/workflows/*.yml 负责 CI 和 release
```

如果你要照它创建自己的项目，建议先把 `assets/interface.json` 和 `assets/resource/pipeline/` 跑通，再考虑 Python 自定义逻辑和发布流程。
