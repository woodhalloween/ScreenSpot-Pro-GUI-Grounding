# ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use
![Contributions welcome](https://img.shields.io/badge/contributions-welcome-orange.svg?style=for-the-badge)
[![Research Paper](https://img.shields.io/badge/Paper-brightgreen.svg?style=for-the-badge)](https://likaixin2000.github.io/papers/ScreenSpot_Pro.pdf)
[![Huggingface Dataset](https://img.shields.io/badge/Dataset-blue.svg?style=for-the-badge)](https://huggingface.co/datasets/likaixin/ScreenSpot-Pro)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-8A2BE2?style=for-the-badge)](https://gui-agent.github.io/grounding-leaderboard)

## 📢 Updates
(Feb 21 2025) We're excited to see our work acknowledged and used as a benchmark in several great projects: [Omniparser v2](https://www.microsoft.com/en-us/research/articles/omniparser-v2-turning-any-llm-into-a-computer-use-agent/), [Qwen2.5-VL](https://arxiv.org/pdf/2502.13923), [UI-TARS](https://arxiv.org/pdf/2501.12326), [UGround](https://x.com/ysu_nlp/status/1882618596863717879), [AGUVIS](https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/issues/2), ...

## Set Up

Before you begin, ensure your environment variables are set:

- `OPENAI_API_KEY`: Your OpenAI API key.

### Environment Setup

#### Option 1: Simple pip installation (Recommended)
```bash
# Using requirements.txt (Recommended)
pip install -r requirements.txt

# Or install all required dependencies manually
pip install torch>=2.0.0 transformers>=4.40.0 pillow==10.2.0 tqdm==4.66.2 accelerate==1.5.2 qwen-vl-utils==0.0.10 torchvision==0.21.0 --upgrade jinja2

# Or install individually
pip install torch>=2.0.0 transformers>=4.40.0 pillow==10.2.0 tqdm==4.66.2 accelerate==1.5.2 qwen-vl-utils==0.0.10 torchvision==0.21.0
pip install --upgrade jinja2

# Download required datasets
python download_dataset.py
```

#### Option 2: Poetry installation
1. Install Poetry:
```bash
pip install poetry
```

2. Update Poetry lock file:
```bash
poetry lock
```

3. Install dependencies:
```bash
poetry install
```

4. Activate virtual environment:
```bash
poetry shell
```

5. Download required datasets:
```bash
python download_dataset.py
```

6. Configure Git (if you plan to commit changes):
```bash
git config --global user.name "あなたの名前"
git config --global user.email "あなたのメールアドレス"
```

### Requirements
- Python version: >=3.10, <3.12
- PyTorch: >=2.0.0
- Sufficient disk space for models and datasets

## Evaluation
Use the shell scripts to launch the evaluation. 
```bash 
bash run_ss_pro.sh
```
or
```bash 
bash run_ss_pro_cn.sh
```

### Running Qwen2VL Model
To evaluate using the Qwen2VL model, you can run the following command:
```bash
PYTHONPATH=$PWD python eval_screenspot_pro.py --model_type qwen2vl --screenspot_imgs "./data/ScreenSpot-Pro/images" --screenspot_test "./data/ScreenSpot-Pro/annotations" --task "all" --language "en" --gt_type "positive" --log_path "./results/qwen2vl.json" --inst_style "instruction"
```

The evaluation supports automatic checkpointing - if interrupted, you can run the same command again to resume from where it left off. Progress is automatically saved every 10 tasks to a `.partial` file.

### Important Notes
- When using the simple pip installation (Option 1), FlashAttention2 is not required and the model will run without it
- The `PYTHONPATH=$PWD` environment variable is needed to properly locate modules in the project directory
- You may see some warning messages during execution, but they will not affect model performance
- The Jinja2 upgrade is necessary to resolve potential template processing issues

# Citation
Please consider citing if you find our work useful:
```plain
@misc{li2024screenspot-pro,
      title={ScreenSpot-Pro: GUI Grounding for Professional High-Resolution Computer Use}, 
      author={Kaixin Li and Ziyang Meng and Hongzhan Lin and Ziyang Luo and Yuchen Tian and Jing Ma and Zhiyong Huang and Tat-Seng Chua},
      year={2025},
}
```