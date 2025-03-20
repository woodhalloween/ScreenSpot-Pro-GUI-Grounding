import os
from huggingface_hub import hf_hub_download, snapshot_download

# データ保存先ディレクトリ
output_dir = "../data/ScreenSpot-Pro"

# ScreenSpot-Proデータセットをダウンロード
print("ScreenSpot-Proデータセットをダウンロードしています...")
snapshot_download(
    repo_id="likaixin/ScreenSpot-Pro",
    local_dir=output_dir,
    repo_type="dataset",
    ignore_patterns=["*.git*", "README.md", "*.jpg"],  # 不要なファイルを除外
)

print(f"データセットが {output_dir} にダウンロードされました") 