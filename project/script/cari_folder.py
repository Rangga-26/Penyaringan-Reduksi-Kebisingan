import os

print("Current working directory:", os.getcwd())
print("\nChecking raw directory structure:")
raw_path = "raw"
if os.path.exists(raw_path):
    for root, dirs, files in os.walk(raw_path):
        level = root.replace(raw_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # Tampilkan 5 file pertama saja
            print(f"{subindent}{file}")
else:
    print(f"Folder '{raw_path}' tidak ditemukan!")
    print("Mencari folder raw di parent directory...")
    
    # Cari folder raw di parent
    parent_dir = os.path.dirname(os.getcwd())
    raw_parent = os.path.join(parent_dir, "raw")
    if os.path.exists(raw_parent):
        print(f"Found raw at: {raw_parent}")
        for root, dirs, files in os.walk(raw_parent):
            print(f"  {root}")
    else:
        print("Folder raw tidak ditemukan di current atau parent directory")