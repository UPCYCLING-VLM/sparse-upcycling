#Script 2
import os
import re
import tarfile
import zipfile
from glob import glob
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR = "downloads"

# Patterns for identifying split tarballs
SPLIT_PATTERNS = [
    re.compile(r"(.+\.tar)\.part\d+$"),         # e.g., file.tar.part01
    re.compile(r"(.+\.tar\.gz)\.\d+$"),         # e.g., file.tar.gz.001
    re.compile(r"(.+\.tar)\.part[a-z]{2}$"),    # e.g., file.tar.partaa
]

def extract_tar(tar_path, extract_to):
    try:
        print(f"Extracting TAR: {tar_path}")
        with tarfile.open(tar_path, "r:*") as tar:
            tar.extractall(path=extract_to)
        print(f"✅Extracted: {tar_path}")
        return True
    except Exception as e:
        print(f"❌Failed to extract {tar_path}: {e}")
        return False

def extract_zip(zip_path, extract_to):
    try:
        print(f" Extracting ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(path=extract_to)
        print(f" Extracted: {zip_path}")
        return True
    except Exception as e:
        print(f" Failed to extract {zip_path}: {e}")
        return False

def reconstruct_and_extract_splits():
    seen = set()
    print("\n Scanning for split archives and tarballs...")

    for dirpath, _, filenames in tqdm(os.walk(ROOT_DIR), desc="Walking directories"):
        for filename in tqdm(filenames, leave=False, desc="Checking files"):
            full_path = os.path.join(dirpath, filename)

            # Handle split archives
            for pattern in SPLIT_PATTERNS:
                match = pattern.match(filename)
                if match:
                    base_name = match.group(1)
                    if (dirpath, base_name) in seen:
                        continue
                    seen.add((dirpath, base_name))
                    base_path = os.path.join(dirpath, base_name)

                    # Collect matching parts
                    parts = sorted([
                        f for f in os.listdir(dirpath)
                        if f.startswith(os.path.basename(base_name)) and "combined" not in f
                    ])
                    parts = [os.path.join(dirpath, f) for f in parts]

                    print(f"\n Reconstructing: {base_name} from {len(parts)} parts")
                    combined_path = base_path + ".combined"

                    with open(combined_path, "wb") as outfile:
                        for part in tqdm(parts, desc="Merging parts", leave=False):
                            with open(part, "rb") as infile:
                                outfile.write(infile.read())

                    if extract_tar(combined_path, dirpath):
                        os.remove(combined_path)
                        for part in parts:
                            os.remove(part)
                    break

            # Handle regular tarballs
            if filename.endswith((".tar", ".tar.gz", ".tgz")):
                if extract_tar(full_path, dirpath):
                    os.remove(full_path)

def extract_zip_files():
    print("\n Scanning for ZIP files...")
    for dirpath, _, filenames in tqdm(os.walk(ROOT_DIR), desc="Walking directories for ZIPs"):
        for filename in tqdm(filenames, leave=False, desc="Checking ZIP files"):
            if filename.endswith(".zip"):
                zip_path = os.path.join(dirpath, filename)
                if extract_zip(zip_path, dirpath):
                    os.remove(zip_path)

def main():
    print(" Archive Extraction Script Started")
    reconstruct_and_extract_splits()
    extract_zip_files()
    print("\n All done!")

if __name__ == "__main__":
    main()
