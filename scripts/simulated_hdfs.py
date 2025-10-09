import os
import shutil

BASE_PATH = os.path.join(os.getcwd(), "data")

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def hdfs_put(local_path, hdfs_dir="raw"):
    """Simulate uploading file to HDFS (copy to data/raw)."""
    dest_dir = os.path.join(BASE_PATH, hdfs_dir)
    ensure_dir(dest_dir)
    dest_path = os.path.join(dest_dir, os.path.basename(local_path))
    shutil.copy(local_path, dest_path)
    print(f"[HDFS PUT] {local_path} -> {dest_path}")
    return dest_path

def hdfs_list(hdfs_dir):
    """List files in simulated HDFS directory."""
    dir_path = os.path.join(BASE_PATH, hdfs_dir)
    if not os.path.exists(dir_path):
        return []
    return os.listdir(dir_path)

def hdfs_get(file_name, hdfs_dir="cleaned"):
    """Retrieve a file path from simulated HDFS."""
    path = os.path.join(BASE_PATH, hdfs_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{file_name} not found in simulated HDFS dir '{hdfs_dir}'")
    return path

def hdfs_delete(hdfs_dir="raw"):
    """Clear a simulated HDFS directory."""
    dir_path = os.path.join(BASE_PATH, hdfs_dir)
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)
    print(f"[HDFS DELETE] Cleared {hdfs_dir}")

def hdfs_move(file_name, src_dir="cleaned", dest_dir="results"):
    """Move a file between simulated HDFS directories."""
    src = os.path.join(BASE_PATH, src_dir, file_name)
    dst_dir = os.path.join(BASE_PATH, dest_dir)
    ensure_dir(dst_dir)
    dst = os.path.join(dst_dir, file_name)
    if not os.path.exists(src):
        raise FileNotFoundError(f"{src} not found")
    shutil.move(src, dst)
    print(f"[HDFS MOVE] {file_name}: {src_dir} -> {dest_dir}")
    return dst
