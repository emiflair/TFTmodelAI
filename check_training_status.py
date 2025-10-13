#!/usr/bin/env python3
"""
ENHANCED TFT TRAINING STATUS CHECKER
===================================
Quick verification that training is proceeding correctly.
"""
import os
import sys
import time
from pathlib import Path

print("🔍 ENHANCED TFT TRAINING STATUS CHECK")
print("=" * 50)

# Check if training artifacts are being created
artifacts_dir = Path("artifacts")
lightning_dir = Path("lightning_logs")

print(f"📂 Artifacts directory exists: {artifacts_dir.exists()}")
print(f"📂 Lightning logs directory exists: {lightning_dir.exists()}")

if lightning_dir.exists():
    versions = list(lightning_dir.glob("version_*"))
    print(f"📊 Lightning log versions found: {len(versions)}")
    for version in sorted(versions):
        print(f"   - {version.name}")
        
        # Check for recent metrics
        metrics_file = version / "metrics.csv"
        if metrics_file.exists():
            size = metrics_file.stat().st_size
            mtime = time.ctime(metrics_file.stat().st_mtime)
            print(f"     📈 metrics.csv: {size} bytes, modified: {mtime}")

# Check for checkpoint creation
checkpoints_dir = artifacts_dir / "checkpoints"
if checkpoints_dir.exists():
    checkpoints = list(checkpoints_dir.glob("*.ckpt"))
    print(f"💾 Checkpoints found: {len(checkpoints)}")
    for ckpt in sorted(checkpoints):
        size = ckpt.stat().st_size / (1024*1024)  # MB
        mtime = time.ctime(ckpt.stat().st_mtime)
        print(f"   - {ckpt.name}: {size:.1f} MB, modified: {mtime}")

# Check for any error logs
print("\n🔍 CHECKING FOR RECENT ACTIVITY...")
all_files = []
for root, dirs, files in os.walk("."):
    for file in files:
        filepath = Path(root) / file
        if filepath.suffix in ['.log', '.csv', '.ckpt', '.json']:
            try:
                mtime = filepath.stat().st_mtime
                if time.time() - mtime < 300:  # Modified in last 5 minutes
                    all_files.append((filepath, mtime))
            except:
                pass

if all_files:
    print("📝 Recent file activity (last 5 minutes):")
    for filepath, mtime in sorted(all_files, key=lambda x: x[1], reverse=True):
        print(f"   - {filepath}: {time.ctime(mtime)}")
else:
    print("⏳ No recent file activity detected...")

print("\n" + "=" * 50)
print("🚀 Enhanced TFT training status check complete!")
print("💡 If training is active, you should see new files being created.")