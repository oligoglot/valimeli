#!/usr/bin/env python3
import os
import shutil

print("====================================================================")
print("     PROJECT VALIMELI — LOCAL WORKSTATION SANITIZATION & SETUP")
print("====================================================================")

# 1. Clean up duplicate container/workspace directories
if os.path.exists("workspace") and os.path.isdir("workspace"):
    print("🧹 Detected duplicate 'workspace/' directory. Removing it to prevent path collisions...")
    try:
        shutil.rmtree("workspace")
        print("   ✅ Duplicate 'workspace/' successfully deleted.")
    except Exception as e:
        print(f"   ⚠️ Could not delete 'workspace/' folder: {e}")

# 2. Re-create the clean folder layout
folders = ["src", "data", "docs", "scratch"]
for f in folders:
    os.makedirs(f, exist_ok=True)
print(f"📁 Verified and created local layout: {', '.join(folders)}")

# 3. Smart file organizer
# Locate files downloaded from the Studio panel and route them to their proper homes
rules = {
    "valimeli-benchmark-v8.py": "src/valimeli-benchmark.py",
    "valimeli-benchmark.py": "src/valimeli-benchmark.py",
    "plot_results.py": "src/plot_results.py",
    "run_grid.sh": "run_grid.sh",
    "requirements.txt": "requirements.txt",
    "README.md": "README.md",
    "paper-skeleton-v3.md": "docs/paper-skeleton.md",
    "valimeli-pipeline-trace.md": "docs/valimeli-pipeline-trace.md"
}

# Scan both current directory and default macOS Downloads folder
downloads_path = os.path.expanduser("~/Downloads")

for source_name, target_dest in rules.items():
    found_path = None
    # Check current directory
    if os.path.exists(source_name):
        found_path = source_name
    # Check Downloads folder
    elif os.path.exists(os.path.join(downloads_path, source_name)):
        found_path = os.path.join(downloads_path, source_name)
        
    if found_path:
        print(f"🚚 Found '{source_name}'. Moving to '{target_dest}'...")
        try:
            # Copy and overwrite
            shutil.copyfile(found_path, target_dest)
            # Make run_grid.sh executable
            if target_dest == "run_grid.sh":
                os.chmod(target_dest, 0o755)
        except Exception as e:
            print(f"   ❌ Error moving {source_name}: {e}")

# 4. Check for data zips and move them to data/
zips = ["tam.zip", "mal.zip"]
for z in zips:
    found_zip = None
    if os.path.exists(z):
        found_zip = z
    elif os.path.exists(os.path.join(downloads_path, z)):
        found_zip = os.path.join(downloads_path, z)
        
    if found_zip:
        print(f"📦 Found data archive '{z}'. Organizing to 'data/{z}'...")
        try:
            shutil.copyfile(found_zip, os.path.join("data", z))
        except Exception as e:
            print(f"   ⚠️ Could not move {z}: {e}")

print("\n🎉 WORKSPACE SANITIZATION & LAYOUT SETUP COMPLETED!")
print("To start your benchmark, simply execute:")
print("  ./run_grid.sh")
print("====================================================================")
