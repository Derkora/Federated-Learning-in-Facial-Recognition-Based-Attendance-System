import os

def cleanup_windows_metadata(target_dir):
    if not os.path.exists(target_dir):
        print(f"Skipping (not found): {target_dir}")
        return
    print(f"Cleaning metadata in: {target_dir}")
    count = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if ":Zone.Identifier" in file:
                file_path = os.path.join(root, file)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file}")
                    count += 1
                except Exception as e:
                    print(f"Error deleting {file}: {e}")
    print(f"Finished cleaning {os.path.basename(target_dir)}. Total deleted: {count}")

if __name__ == "__main__":
    # Script is in 'pre/'
    base_path = os.path.dirname(os.path.abspath(__file__))
    proj_path = os.path.dirname(base_path)
    
    # 1. Clean common 'datasets' folder (sibling of pre/)
    datasets_path = os.path.join(proj_path, "datasets")
    cleanup_windows_metadata(datasets_path)
    
    # 2. Clean 'centralized-learning' folder
    cl_path = os.path.join(proj_path, "centralized-learning")
    cleanup_windows_metadata(cl_path)
    
    # 3. Clean 'federated-learning' folder
    fl_path = os.path.join(proj_path, "federated-learning")
    cleanup_windows_metadata(fl_path)
    
    # 4. Clean 'pre' folder itself
    cleanup_windows_metadata(base_path)
