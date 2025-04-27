#Script 1
import os
import json
import ijson
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_download

# === CONFIGURATION ===
DATASET_NAME = "neulab/PangeaInstruct"
CACHE_DIR = "downloads"
JSON_FILE_NAME = "PangeaIns.json"
TARGET_TASKS = [
    'cambrian', 'ALLaVA-4V', 'allava_vflan', 'MTVQA', 'nvlr2-llava', "translation",
    'ChartQA', 'Viet-ShareGPT-4o-Text-VQA', 'Viet-OCR-VQA', 'Viet-Doc-VQA', 'table-vqa', 'doc-vqa',
    "laion-caption", "NuminaMath-CoT", "OpenHermes-2.5", 'text_only', 'ocr', 'cultural/laion-cultural-150k'
]
TARGET_LANGUAGES = [
    'arabic', 'bengali', 'bn', 'hindi', 'ja', 'hi', 'russian', 'ru', 'spanish', 'es',
    'vietnamese', 'vi', 'zh_simplified', 'ar', 'en', 'english', 'fr', 'japanese', 'french'
]
TARGET_PATHS = [
    "general/cambrian/",
    "general/ALLaVA-4V/",
    "general/allava_vflan/",
    "general/MTVQA/",
    "general/nvlr2-llava/",
    "translation/",
    "doc+chart/ChartQA/",
    "general/Viet-ShareGPT-4o-Text-VQA/",
    "doc+chart/Viet-OCR-VQA/",
    "doc+chart/Viet-Doc-VQA/",
    "doc+chart/Viet-DOC-VQA-II",
    "doc+chart/table-vqa/",
    "doc+chart/doc-vqa/",
    "text-only/NuminaMath-CoT/",
    "text-only/Openhermes-2.5/",
    "ocr/webui_multilingual_ocr/",
    "cultural/laion-cultural-150k/"
]

# === HELPER FUNCTIONS ===

def is_valid_sample(sample, tasks, target_languages):
    if not isinstance(sample, dict):
        return False
    sample_id = str(sample.get('image', '')).lower()
    sample_lang = str(sample.get('language', '')).lower()
    tasks = [t.lower() for t in tasks]
    target_languages = [l.lower() for l in target_languages]

    return any(task in sample_id for task in tasks) and sample_lang in target_languages

def download_json_file(api, files, filename):
    print(f"\nLooking for {filename} in dataset files...")
    for file in files:
        if file.startswith(filename):
            print(f"\nDownloading metadata file: {file}")
            return hf_hub_download(
                repo_id=DATASET_NAME,
                filename=file,
                repo_type="dataset",
                cache_dir=CACHE_DIR
            )
    raise FileNotFoundError(f"{filename} not found in dataset.")

def filter_json(json_path, tasks, target_languages):
    print(f"\nFiltering samples from: {json_path}")
    valid_samples = []
    with open(json_path, "r", encoding="utf-8") as f:
        items = ijson.items(f, "item")
        for item in tqdm(items, desc="Filtering JSON"):
            if is_valid_sample(item, tasks, target_languages):
                valid_samples.append(item)
    print(f"Total valid samples: {len(valid_samples)}")
    return valid_samples

def download_assets(api, files, target_paths):
    print(f"\nDownloading non-JSON files from target paths...")
    downloaded_count = 0
    for file in tqdm(files, desc="Downloading assets"):
        if any(file.startswith(path) for path in target_paths) and not file.endswith(".json"):
            try:
                hf_hub_download(
                    repo_id=DATASET_NAME,
                    filename=file,
                    repo_type="dataset",
                    cache_dir=CACHE_DIR
                )
                downloaded_count += 1
            except Exception as e:
                print(f"❌Failed to download {file}: {e}")
    print(f"Downloaded {downloaded_count} files.")

# === MAIN SCRIPT ===

def main():
    print("Starting PangeaInstruct Dataset Downloader...\n")

    api = HfApi()
    print("Fetching dataset file list...")
    files = api.list_repo_files(repo_id=DATASET_NAME, repo_type="dataset")

    # Step 1: Download JSON metadata file
    json_path = download_json_file(api, files, JSON_FILE_NAME)

    # Step 2: Filter metadata
    filtered_samples = filter_json(json_path, TARGET_TASKS, TARGET_LANGUAGES)

    # Step 3: Download assets from specified subdirectories
    download_assets(api, files, TARGET_PATHS)

    # Optionally save filtered metadata
    filtered_path = os.path.join(CACHE_DIR, "filtered_samples.json")
    with open(filtered_path, "w", encoding="utf-8") as out_f:
        json.dump(filtered_samples, out_f, ensure_ascii=False, indent=2)
    print(f"\nFiltered metadata saved to: {filtered_path}")

if __name__ == "__main__":
    main()
