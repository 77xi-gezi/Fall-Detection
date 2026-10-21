# 单张图片人体跌倒状态识别

输入一张图片，立即输出标注图片、状态 JSON，以及控制台结果。
图片中有多个人时，每个人分别输出 normal（正常）、falling（疑似跌倒）或 fallen（倒地）。
没有检测到人体时输出 no_person（未检测到人体）。

> 当前 LSTM 权重原用于 30 帧连续画面，单图准确率尚未验证，已有误判。
> 代码提供可运行的实验流程，模型分数不代表经过验证的识别准确率。

## 从 GitHub 下载并启动

本仓库存放源码、模型和部署配置。图片检测网页需要在电脑或服务器上启动，
不会在 GitHub 仓库页面中直接运行。

本机验证环境为 Windows、Python 3.13、NVIDIA GPU，以及启用 Linux 容器的
Docker Desktop。Triton 配置使用 GPU；本地 Python 检测脚本也支持 CPU。

1. 安装 Python、Git 和 Docker Desktop，在 PowerShell 中下载项目：

   ```powershell
   git clone https://github.com/77xi-gezi/Fall-Detection.git
   cd Fall-Detection
   ```

2. 安装 Python 环境与依赖：

   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
   ```

3. 打开 Docker Desktop，等待引擎启动，然后双击项目中的 `start_browser.cmd`。
   首次启动会下载 Triton Docker 镜像，需要网络和足够磁盘空间。
4. 浏览器打开 **http://127.0.0.1:8080**，选择自己的图片并点击“开始检测”。

仓库包含 `yolo26s-pose.pt`、`models/checkpoints/best.pth`，以及
`deploy/triton/model_repository/` 下两个可直接加载的 ONNX 模型。
虚拟环境、Docker 镜像、原始图片数据和检测输出不随 Git 仓库上传。
使用脚本批量检测时，将自己的图片放入 `input/images/`。

## 浏览器图片检测

打开 Docker Desktop，双击项目中的 **`start_browser.cmd`**。
启动程序会确认 Triton 已就绪，然后自动打开 **http://127.0.0.1:8080**。
网页支持点击选择或拖入单张、多张图片，显示每个人的状态与模型分数，
支持放大标注图，以及下载标注图片、JSON、CSV 和完整结果 ZIP。
每张图片独立处理；当前模型的单图准确率限制同样适用，网页已说明。

每次最多 50 张、每张不超过 16 MB，总大小不超过 100 MB，分辨率不超过 2500 万像素。
结果保存在 `output/web_results/<本次任务编号>/`。原始上传文件处理后移除，
保留标注图和结果记录。关闭浏览器后结果文件仍然保留。
同批中有图片损坏时会显示该图片的错误，并继续检测其余图片。

也可以在 PowerShell 中启动，保留该终端窗口运行网页服务：

```powershell
# 在包含 start_browser.cmd 的项目目录执行
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_web.ps1
```

在 VS Code 按 F5 可选“浏览器：图片检测网页（Triton）”，此入口需要 Docker 中的
Triton 已经运行。关闭启动终端或按 Ctrl+C 可停止前台网页服务，Triton 容器仍可独立运行。

网页端口为 **8080**；`http://127.0.0.1:8000/v2` 是 Triton 服务器信息接口。
网页后端使用 Flask 与 Windows 可用的 Waitress，通过现有客户端调用 Triton，
不改变模型或原有检测流程。网页服务只监听本机地址。
项目环境安装脚本已包含网页依赖，也可单独安装或修复：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
```

上传与服务实现参考 [Flask 文件上传](https://flask.palletsprojects.com/en/stable/patterns/fileuploads/)
和 [Waitress 部署说明](https://flask.palletsprojects.com/en/stable/deploying/waitress/)。

## ONNX / Triton 服务

已导出 YOLO26s-Pose 和 LSTM 的 ONNX 模型，并部署到本机 Triton GPU 服务。
详细操作见 [Triton 部署说明](deploy/triton/README.md)。
Docker 服务运行时，可在 VS Code 按 F5 选择“Triton：文件夹批量图片”，
也可以在现有命令后加 `--backend triton`：

```powershell
.\.venv\Scripts\python.exe detect_folder.py --input input\images --backend triton
```

服务健康检查地址为 `http://127.0.0.1:8000/v2/health/ready`。

## 文件夹批量检测

将多张图片放入 `input/images/`，在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe detect_folder.py
```

也可以指定任意图片文件夹和结果文件夹：

```powershell
.\.venv\Scripts\python.exe detect_folder.py --input "D:\待检测图片" --output "output/batch_results"
```

VS Code 按 F5，选择“跌倒检测：文件夹批量单图（YOLO26s-Pose）”，输入文件夹路径。
每张图片独立检测，可来自不同场景，无需连续、同尺寸或达到 30 张；模型仅加载一次。
默认只读该文件夹中的图片，加 `--recursive` 可包含子文件夹，输出目录会从扫描中排除。

默认结果在 `output/batch_results/`：

- `images/`：标注图片，例如 `photo.jpg_result.png`。
- `results/`：每张图片的完整状态 JSON，例如 `photo.jpg.json`，含各个人物的结果。
- `summary.csv`：每张图片一行，包含状态、检测人数、输出路径及失败原因，可用 Excel 打开。
- `summary.json`：本次总图片数、成功／失败数及状态统计。

保留原扩展名用于区分同名 JPG 和 PNG；递归模式会保留子文件夹结构。
损坏图片会在汇总中记为 error 并继续处理其余图片；有失败时退出码为 1，
全部成功为 0，输入或模型初始化失败为 2。重复运行会覆盖同名结果文件，
汇总表记录本次处理结果。批量脚本沿用下文所述的模型和准确率限制。

## 运行

VS Code 打开本目录的 `Fall-Detection.code-workspace`，按 F5，选择
“跌倒检测：单张图片（YOLO26s-Pose）”，输入图片路径。

也可在项目根目录的 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe detect_falls.py --input input\photo.jpg
.\.venv\Scripts\python.exe detect_falls.py --input input\photo.jpg --output output\result.png
```

支持 JPG、JPEG、PNG、BMP、TIFF、WEBP 和中文文件路径。
默认自动使用 CUDA（GPU 可用时），也可传入 --device cpu。
--json-output 可指定状态 JSON 的保存路径。

默认输出 output/<图片名>_result.png 和 output/<图片名>_result.json。
JSON 的 persons 数组给出每个人的状态、模型概率、人体框和关键点；
顶层 state 汇总图片中的模型状态，优先级为 fallen > falling > normal。
控制台输出状态摘要，图片中标注人物编号、状态和最大类别概率。

默认姿态权重为 yolo26s-pose.pt，安装脚本会下载官方权重。当前已安装的
Ultralytics 8.4.177 支持该模型，无需更新 Python 或 CUDA。
模型说明：[Ultralytics YOLO26 Pose](https://docs.ultralytics.com/tasks/pose/)。

## 保留的技术流程

```
一张图片
  → YOLO26s-Pose：人体框与 17 个关键点（predict，不使用跟踪）
  → 原始 5 特征：身体角度、重心高度、重心变化、框宽高比、关键点置信度
  → LSTM：每个人输入形状为 (1, 1, 5)，一次观测
  → 三类模型概率，取概率最大的类别
```

连续帧、滑动窗口、BoT-SORT 跟踪、摄像头、视频和 RTSP 模式已删除。
不会复制同一张图片填充 30 帧。单图没有运动历史，重心变化特征固定为 0。

## 当前模型的适用范围

现有 best.pth 权重是在 30 帧序列上训练的。本程序复用它进行单时间步推理，
完成单图输入和状态输出，但尚未用有标签的单图数据验证准确率。
输出的三类概率是模型分数，不能当作经过单图校准的可信度；
单张图片也没有用于确认“正在跌倒”过程的运动信息。
本次运行验证中，input/demo.jpg 的倒地人物被判为 normal，属于已观察到的误判，
说明现有时序权重不能直接视为可用的单图分类权重。接口测试只检查数据流和输出，
并不代表真实识别准确率。
要获得可靠的单图分类效果，应使用有标签的单图数据重新训练或微调分类器。
当前程序没有新增规则评分，也没有声称时序模型的原有准确率适用于单图。

JSON 的 inference 字段明确记录 input_frames=1、motion_observed=false、
single_image_accuracy_validated=false。LSTM 结构和原始 5 特征计算方法保持不变。

input/demo.jpg 是运行演示图片，部分图像自带原项目标注，只用于检查运行和文件输出。

## 环境与检查

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pip check
```

现有 .venv 继续使用 PyTorch 2.10.0+cu126、torchvision 0.25.0+cu126。
requirements-windows.lock.txt 固定当前必要依赖；不再需要跟踪器的 LAP 依赖。

核心文件：

```
detect_falls.py            单图入口
src/image_io.py            单图读写
src/detector.py            YOLO26s-Pose 单图检测
src/feature_extraction.py  原始 5 特征
src/pipeline.py            单时间步分类和状态输出
models/fall_detector.py    LSTM 定义
```

当前仓库只保留图片检测功能，训练入口和连续帧模式不在本版本中。
本地开发备份属于 `.tools/`，不随仓库上传。
