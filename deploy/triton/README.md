# ONNX / Triton 部署

Docker Compose 服务名为 `fall-detection-triton`，使用
`nvcr.io/nvidia/tritonserver:26.08-py3`。本机验证版本为 Triton 2.72.0，
GPU 为 RTX 2060。默认配置将两个 ONNX 模型放到 GPU 0 执行。
仓库已经包含导出的 ONNX 模型，首次运行不需要重新导出。

## 浏览器入口

在项目根目录双击 `start_browser.cmd`，或执行 `scripts/start_web.ps1`。
浏览器打开 **http://127.0.0.1:8080** 上传图片或视频、查看与下载检测结果。
Triton 的 **http://127.0.0.1:8000/v2** 返回服务器信息，**8000** 供网页后端调用，
图片上传和结果展示由 **8080** 端口的网页服务负责。
详细使用方法见项目根目录 README 的“浏览器图片与视频检测”。

## 使用现有图片脚本

VS Code 按 F5，选择“**Triton：文件夹批量图片**”或“**Triton：单张图片**”。
已有本地 PyTorch 入口也保留，选择 Triton 入口才会请求服务器。

在包含 `start_browser.cmd` 的项目根目录打开 PowerShell，准备自己的图片后运行：

```powershell
# 单张图片
.\.venv\Scripts\python.exe detect_falls.py --input input\photo.jpg --backend triton

# 文件夹内的图片逐张独立检测
.\.venv\Scripts\python.exe detect_folder.py --input input\images --output output\triton_results --backend triton
```

标注图片、状态 JSON、批量 CSV 的格式与本地模式一致。
`--triton-url` 可指定服务器地址；当前导出模型要求 `--imgsz 640`。
`--backend triton` 使用服务器仓库里的固定权重，客户端 `--model` / `--yolo-model`
不会重新加载服务器模型。改变模型需要重新导出并重启服务。

## 启动与停止

Docker Desktop 应处于运行状态，并使用 WSL2 / Linux 容器后端。

```powershell
# 启动，已运行时重复执行即可
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\start_triton.ps1

# 检查健康状态，应返回 HTTP 200
Invoke-WebRequest http://127.0.0.1:8000/v2/health/ready

# 查看模型加载日志
docker compose -f deploy\triton\compose.yaml logs --tail 60

# 停止本项目容器
docker compose -f deploy\triton\compose.yaml stop
```

端口映射到本机 `127.0.0.1`：HTTP 8000、gRPC 8001、metrics 8002。
Docker 的 restart 策略为 unless-stopped，Docker 重启后会恢复之前运行的服务。
镜像及模型仓库持久保留，模型目录以只读方式挂载。

## 模型仓库与接口

```
model_repository/
  yolo26s_pose/
    config.pbtxt
    1/model.onnx
  fall_lstm/
    config.pbtxt
    1/model.onnx
```

| 模型 | 输入 | 输出 |
|---|---|---|
| yolo26s_pose | images，FP32，[1,3,640,640] | output0，FP32，[1,300,57] |
| fall_lstm | features，FP32，[人数,帧数,5] | logits，FP32，[人数,3] |

YOLO 使用 end-to-end 分支。每条检测为 xyxy 框、检测分数、类别 ID、17×3 关键点，
客户端按人体置信度过滤并还原到原图坐标。FP32 ONNX 使用 opset 17。
LSTM 的人数与时间维度可变：图片输入 1 个真实观测，视频输入 30 个连续观测。

HTTP 模型推理接口：

- `POST /v2/models/yolo26s_pose/infer`
- `POST /v2/models/fall_lstm/infer`

这些 Triton 接口接收张量，不能直接将 JPG 当作请求体；图片脚本负责图像预处理。
两个神经网络在服务器执行，图像预处理、5 特征提取、softmax、标注和文件保存由
Python 客户端执行。客户端实现位于 `src/triton_backend.py`。

## 重新导出与验证

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-triton.txt
.\.venv\Scripts\python.exe -X utf8 scripts\export_triton.py
docker compose -f deploy\triton\compose.yaml restart
.\.venv\Scripts\python.exe -X utf8 scripts\verify_triton.py --image input\photo.jpg
```

也可执行 `scripts/start_triton.ps1 -Export` 重新导出后启动。
导出清单位于 `export_manifest.json`，运行验证脚本后会生成 `validation_report.json`。
验证图片由使用者自行提供，原始图片和机器相关的验证报告不随仓库上传。
本次验证包括 ONNX 图检查、LSTM 的 1/2/5 人 × 1/30 帧输入、YOLO 与原始 PyTorch 的同预处理比较，
以及通过实际 Triton HTTP 接口完成单图和批量推理。

Docker Desktop 中预先下载的 `26.08-pyt-python-py3` 只有 PyTorch/Python 后端，
本项目因此使用同版本的 `26.08-py3` 完整镜像，它包含 ONNX Runtime 后端。

ONNX 转换和服务部署只验证数值一致性，未重新训练单图分类器。
现有 LSTM 在单图模式下的已知误判和准确率限制仍然存在，详见项目主 README。

官方资料：[Triton Quickstart](https://github.com/triton-inference-server/server/blob/main/docs/getting_started/quickstart.md)、
[HTTP 二进制张量协议](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_binary_data.html)、
[Docker Desktop GPU 支持](https://docs.docker.com/desktop/features/gpu/)。
