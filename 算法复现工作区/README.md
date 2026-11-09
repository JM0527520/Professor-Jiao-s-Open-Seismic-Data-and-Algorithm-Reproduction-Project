# 公开地震数据与算法复现工作区

主线：**公开数据核对 → 已有算法复现 → 建立基线 → 验证一项改进 → 迁移到实际工区。**

当前已完成数据验收、作者模型小规模预测，并准备好训练兼容实现、数据使用规则、完整评价及服务器入口。已在本机完成32³小裁块的梯度更新和优化器恢复检查；完整128³训练与全区预测暂存等待服务器。原始数据从同级“公开数据集”只读取得。全部代码、配置、环境、缓存和结果都保存在本工作区。

## 查看成果

- [11月9日准备完成记录](outputs/服务器前准备完成记录.md)：本轮规则、训练代码、本机验证及暂存任务。
- [数据使用与评价规则](docs/数据使用与评价规则.md)：无效、未知、确认负样本和参考正标签如何使用。
- [作者训练流程与兼容说明](docs/作者训练流程与兼容说明.md)：实际batch=2、普通BCE、训练参数及恢复方式。
- [服务器运行手册](docs/服务器运行手册.md)：建议资源、环境安装及后续逐项执行命令。
- [第1步：数据验收报告](outputs/audit/数据验收报告.md)：全部FaultSeg3D配对体，Thebe分区、数值、补零和标签核对。
- [第2步：作者算法预测报告](outputs/predictions/作者算法预测报告.md)：两个合成验证体和一个Thebe验证区域的预测结果。
- [Thebe三向预测对比图](outputs/predictions/thebe_validation_demo_comparison.png)：地震、参考标签、预测概率、叠加对比。
- [数据与算法依据](docs/数据与算法依据.md)：作者实现来源、两代模型的差异、轴序、空间分区与标签含义。
- [验收与运行记录](outputs/本轮完成记录.md)：运行检查、数据问题与后续衔接。

11月3日记录作为历史保留；当前状态以11月9日准备完成记录为准。

## 安装环境

已在本机建立`.venv`，Python 3.12，TensorFlow 2.18.1。Windows原生TensorFlow采用CPU推理；本轮小规模检查已经足够，不依赖4GB显卡进行训练。

已有环境可直接使用以下运行命令。新机器需Python 3.12，在本目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
```

也可使用已安装的uv：

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-lock.txt
```

`requirements-lock.txt`保存本次实际使用的完整版本；`requirements.txt`是运行直接依赖，`requirements-dev.txt`增加检查工具。搬到Linux服务器时重新创建环境并按其GPU驱动配置TensorFlow；本次Windows环境不作为已经验证的服务器GPU训练环境，也不复制`.venv`。

## 运行

在本目录中，先验收，再预测：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m fault_workflow audit
.\.venv\Scripts\python.exe -X utf8 -u -m fault_workflow predict
```

一次运行两步：

```powershell
.\scripts\run_local.ps1 -Stage all
```

运行回归检查：

```powershell
.\scripts\run_local.ps1 -Stage verify
```

本机准备入口（不运行服务器任务）：

```powershell
.\.venv\Scripts\python.exe -X utf8 -u -m fault_workflow.prepare_cli prepare
.\.venv\Scripts\python.exe -X utf8 -u -m fault_workflow.prepare_cli smoke
.\.venv\Scripts\python.exe -X utf8 -u -m fault_workflow.prepare_cli preflight
```

`prepare`沿用已验收资料，重新逐道扫描训练/验证区，输出规则应用清单、演示掩膜及基线协议。`smoke`仅执行CPU32³小裁块工程检查。完整训练和评价需要显式`--server-work`、TensorFlow GPU及完整128³ GPU检查通过，本轮没有执行。

入口均从`configs/local.json`读取数据和输出位置，路径相对本工作区；可使用`--config`指定另一份JSON配置。搬机器或更改原数据后必须重跑验收，预测前会检查数据位置、大小及修改时间。读写分离检查会拒绝把结果或缓存放进原始数据目录。

## 目录与输出含义

| 目录 | 用途 |
|---|---|
| `fault_workflow` | 数据读取、核验、作者模型加载、分块预测及绘图代码 |
| `configs` | 数据位置、作者检查点、预测样例、分块参数 |
| `tests` | 轴序、标准化、损坏文件、补零、分区保护、分块覆盖及输出检查 |
| `docs` | 数据和算法使用依据 |
| `scripts` | 本地运行入口 |
| `server` | 待服务器验证的Linux GPU依赖和Dockerfile |
| `cache` | 一个Thebe验证区域与来源坐标，不是完整训练缓存 |
| `outputs/audit` | 验收报告、逐剖面统计、异常标签位置、三向图与完整剖面图 |
| `outputs/predictions` | 作者预测报告、概率体、逐样例元数据和对比图 |
| `outputs/logs` | 实际运行与检查日志 |
| `outputs/preparation` | 规则应用、原代码训练协议、异常清单及CPU小裁块检查 |

`*_probability_model.npy`使用模型轴序`(垂向采样,横向轴2,分区剖面轴)`，`*_probability_raw.npy`还原为原始存储轴序。两者都是float32概率，保留[0,1]全部数值，未做阈值裁剪。

`*_amplitude_support_model.npy`只是振幅支持范围，保留波形内部零值，不能代替完整人工解释范围掩膜。Thebe窗口的真实来源坐标见`thebe_validation_demo_metadata.json`。

Thebe训练资料存在大量补零。完整包络规则已确定：训练区1,308个、验证区23,514个正标签位于振幅支持范围外，均排除监督和评价、待人工核查，原始标签不改。训练区1,308个已包含原先213个全零道正标签。其余未标注位置保持未知，缺少完整解释范围时不开展Thebe监督微调，不报告其现场Precision/Dice。

下一阶段在服务器建立正式基线，再针对实际问题验证一项改进，最后接入实际工区。当前完成了本机小裁块训练检查，未执行正式基线训练、方法改进或工区迁移。
