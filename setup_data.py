"""
Setup script to copy Ayana data into project structure
"""

import shutil
import os

# Source and destination paths
SOURCE = '/Users/hetalksinmaths/Downloads/filtered_hotels.csv'
DEST_DIR = './data/raw'
DEST_FILE = os.path.join(DEST_DIR, 'filtered_hotels.csv')

# Create directory if it doesn't exist
os.makedirs(DEST_DIR, exist_ok=True)

# Copy file
if os.path.exists(SOURCE):
    print(f"📂 Copying data from: {SOURCE}")
    print(f"   To: {DEST_FILE}")
    shutil.copy2(SOURCE, DEST_FILE)
    print("✅ Data copied successfully!")

    # Get file size
    size_mb = os.path.getsize(DEST_FILE) / (1024 * 1024)
    print(f"   File size: {size_mb:.2f} MB")
else:
    print(f"❌ Error: Source file not found at {SOURCE}")
    print("   Please update the SOURCE path in this script.")
