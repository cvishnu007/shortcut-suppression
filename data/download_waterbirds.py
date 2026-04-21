import os
import urllib.request
import tarfile
import sys

def download_progress(count, block_size, total_size):
    """Reports download progress to the console."""
    percent = int(count * block_size * 100 / total_size)
    sys.stdout.write(f"\rDownloading... {percent}%")
    sys.stdout.flush()

def setup_waterbirds():
    # URL for the official Waterbirds dataset (provided by Sagawa et al.)
    url = "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"
    
    # Paths based on your config.py structure
    raw_dir = os.path.join(".", "data", "raw")
    tar_path = os.path.join(raw_dir, "waterbirds.tar.gz")
    
    # Expected final path from your config.py:
    # ./data/raw/waterbird_complete95_forest2water2/waterbird_complete95_forest2water2
    nested_dir = os.path.join(raw_dir, "waterbird_complete95_forest2water2")
    
    os.makedirs(nested_dir, exist_ok=True)

    # Check if the metadata.csv already exists where config.py expects it
    expected_metadata_path = os.path.join(nested_dir, "waterbird_complete95_forest2water2", "metadata.csv")
    if os.path.exists(expected_metadata_path):
        print("✅ Waterbirds dataset is already downloaded and extracted!")
        return

    print(f"⬇️  Downloading Waterbirds dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, tar_path, reporthook=download_progress)
        print("\n✅ Download complete.")
    except Exception as e:
        print(f"\n❌ Failed to download dataset: {e}")
        return

    print("📦 Extracting archive (this might take a minute)...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            # Extracting into nested_dir to match your config.py's double-folder expectation
            tar.extractall(path=nested_dir)
        print("✅ Extraction complete.")
    except Exception as e:
        print(f"❌ Failed to extract dataset: {e}")
        return
    finally:
        # Clean up the downloaded .tar.gz file to save space
        if os.path.exists(tar_path):
            os.remove(tar_path)
            print("🧹 Cleaned up downloaded archive.")

    print(f"🎉 Dataset is ready! You can now run the waterbirds phase.")

if __name__ == "__main__":
    setup_waterbirds()