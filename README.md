# llamafactory-mlu
(star🌟)对0.9.0的llamafactory适配MLU370，可以在算泥网上直接使用，仅作个人分享。如果需要其他适配MLU的库包，也可以联系我。



# 🚀 寒武纪 MLU370：LLaMA-Factory 魔改版快速部署指南

> **资源包说明**：本教程基于 `fzfz666/llamafactory-mlu` 提供的魔改源码包。该包已彻底解决 MLU 算子转换、Gradio 5.x 界面兼容性、BF16 硬件拦截以及 `ipc_collect` 导致的显存崩溃等核心问题。

## 一、 环境基准检查 (Pre-check)
在部署前，请确保你的系统环境满足以下唯一要求：
1.  **硬件**：寒武纪 MLU370 系列加速卡。
2.  **系统驱动**：执行 `cnmon` 能够正常看到卡信息。
3.  **Python 版本**：**必须是 3.10**（驱动强绑定）。
4.  **底层框架**：已安装寒武纪官方版 PyTorch (`torch_mlu`)，执行 `python -c "import torch_mlu"` 不报错。

---

## 二、 下载与解压魔改包

直接从 GitHub 下载你封装好的全套魔改源码：

```bash
cd /mnt/workspace  # 切换到你的持久化存储目录

# 1. 下载压缩包 (使用 ghfast 加速)
wget https://ghfast.top/https://github.com/fzfz666/llamafactory-mlu/raw/main/LLaMA-Factory_mlu_Source_Only.tar.gz

# 2. 解压
tar -xzvf LLaMA-Factory_mlu_Source_Only.tar.gz

# 3. 进入目录 (此时你应该能看到四个 _mlu 结尾的文件夹)
cd Cambricon_LLM_Env
```

---

## 三、 一键配置 Python 依赖环境

这一步是关键！我们要先安装基础依赖，然后将我们的“魔改版源码”强制挂载到 Python 环境中。

```bash
# 1. 配置阿里源加速
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/

# 2. 锁定安装 Gradio 4.x (防止界面乱码)
pip install "gradio<5.0.0"

# 3. 安装 LLaMA-Factory 的通用运行依赖 (如 datasets, trl, rouge-chinese 等)
cd LLaMA-Factory_mlu
pip install -e .[metrics]
cd ..

# 4. 【核心步骤】强制将环境重定向到我们的 MLU 魔改源码（注意本项目中提供的源码不包含以下三个库的源码，因为太大了上传不上来哈！你可以联系我，或者是直接去算你网使用对应的镜像，到时只需要再重新下载llama factory即可使用。 ）
# 这一步会覆盖掉刚才安装的官方版，确保 import 时调用的是 MLU 适配版
cd transformers_mlu && pip install -e . && cd ..
cd peft_mlu && pip install -e . && cd ..
cd accelerate_mlu && pip install -e . && cd ..
```

---

## 四、 运行微调 (以 Qwen2.5-0.5B 为例)

### 1. 启动 WebUI 界面
```bash
cd LLaMA-Factory_mlu
GRADIO_SERVER_PORT=80 llamafactory-cli webui
```
*   **浏览器访问**：输入服务器 IP。
*   **必选设置**：
    *   **计算精度**：必须选 `fp16`。
    *   **FlashAttention**：必须关闭。
    *   **模型路径**：填入你的 Qwen2.5 存放路径。

### 2. 纯命令行快速验证 (推荐)
如果你想直接看进度条，执行这个脚本：

```bash
cd LLaMA-Factory_mlu

CUDA_VISIBLE_DEVICES=0 python src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path /你的路径/Qwen2.5-0.5B-Instruct \
    --finetuning_type lora \
    --template qwen \
    --dataset_dir data \
    --dataset alpaca_zh_demo \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 5 \
    --save_steps 100 \
    --output_dir saves/qwen2.5_mlu_test \
    --fp16 True \
    --plot_loss True \
    --flash_attn disabled
```

---

## 五、 避坑自查表 (README 核心摘要)

1.  **为什么不让用 BF16？**
    寒武纪 MLU370 在新版 Transformers 下开启 BF16 会触发 NVIDIA Ampere 架构检查。强制使用 **FP16** 即可完美解决且 Loss 正常。
2.  **为什么界面全是英文/乱码？**
    说明 Gradio 自动升级到了 5.x。请务必执行 `pip install "gradio<5.0.0"`。
3.  **报错 `AttributeError: logging...`？**
    不要在 `src/llamafactory/extras` 这种子目录下运行命令，回到 `LLaMA-Factory_mlu` 根目录运行即可。
4.  **找不到数据集？**
    检查 `LLaMA-Factory_mlu/data/dataset_info.json` 是否存在。本魔改包已内置该文件，确保对应的 `.json` 数据文件也在该目录下。

---

### 给 GitHub 读者的建议：
这份代码包是经过实测的“寒武纪大模型生存指南”。如果你在 MLU370 上微调 Qwen2.5 遇到任何算子不支持的问题，请直接使用本项目提供的 `transformers_mlu` 等适配库。
