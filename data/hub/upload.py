import json
import os

from modelscope.hub.api import HubApi


def sync_upload_to_modelscope(namespace: str, dataset: str, repo_dir: str, local_dir: str):
    api = HubApi()
    api.login("")
    repo_id = f"{namespace}/{dataset}"

    # ── 1. 获取远端已有文件（递归） ───────────────────────────────────────
    print(f"[INFO] 拉取远端文件列表: {repo_dir}")
    try:
        resp = api.list_repo_tree(
            namespace=namespace,
            dataset_name=dataset,
            revision="master",
            root_path=repo_dir,
            recursive=True,  # ← 改为 True
            page_size=999,
        )
        files = resp.get("Data", {}).get("Files", [])
        # 用相对于 repo_dir 的路径做去重 key
        remote_paths = {entry["Path"] for entry in files if entry.get("Type") == "blob"}
    except Exception as e:
        print(f"[WARN] 获取远端列表失败，视为空: {e}")
        remote_paths = set()

    print(f"[INFO] 远端已有 {len(remote_paths)} 个文件")

    # ── 2. 递归扫描本地目录 ───────────────────────────────────────────────
    if not os.path.isdir(local_dir):
        raise ValueError(f"local_dir 不存在: {local_dir}")

    to_upload = []
    for root, _, filenames in os.walk(local_dir):  # ← os.walk 递归
        for filename in filenames:
            abs_path = os.path.join(root, filename)
            # 计算相对路径，转成 repo 路径（统一用 /）
            rel_path = os.path.relpath(abs_path, local_dir).replace("\\", "/")
            repo_path = f"{repo_dir.rstrip('/')}/{rel_path}"

            if repo_path not in remote_paths:
                to_upload.append((abs_path, repo_path))

    print(f"[INFO] 待上传 {len(to_upload)} 个文件")
    if not to_upload:
        print("[INFO] 无需上传，退出。")
        return

    # ── 3. 逐文件上传 ────────────────────────────────────────────────────
    success, failed = [], []
    for idx, (local_path, repo_path) in enumerate(to_upload, 1):
        print(f"[{idx}/{len(to_upload)}] {repo_path}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            success.append(repo_path)
        except Exception as e:
            print(f"  [ERROR] {e}")
            failed.append((repo_path, str(e)))

    print(f"\n[DONE] 成功 {len(success)} / 失败 {len(failed)}")
    if failed:
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    sync_upload_to_modelscope(
        namespace="kakarotter",
        dataset="Calix-Dataset",
        repo_dir="pretraining/tokenize/1.12B",
        local_dir="F:/transformer-decoder/pretraining/1.12B/tokenize",
    )
