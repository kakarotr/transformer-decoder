import json
import os

from modelscope.hub.api import HubApi


def sync_upload_to_modelscope(namespace: str, dataset: str, repo_dir: str, local_dir: str):
    api = HubApi()
    api.login("ms-b4f6fcf3-3f96-4fd1-946b-8700911a3d1d")
    repo_id = f"{namespace}/{dataset}"

    # ── 1. 获取远端已有文件名 ──────────────────────────────────────────────
    print(f"[INFO] 拉取远端文件列表: {dataset}/{repo_dir}")
    try:
        resp = api.list_repo_tree(
            namespace=namespace,
            dataset_name=dataset,
            revision="master",
            root_path=repo_dir,
            recursive=False,
        )
        files = resp.get("Data", {}).get("Files", [])
        remote_filenames = {entry["Name"] for entry in files if entry.get("Type") == "blob"}
    except Exception as e:
        print(f"[WARN] 获取远端列表失败，视为空（将上传全部文件）: {e}")
        remote_filenames = set()

    print(f"[INFO] 远端已有 {len(remote_filenames)} 个文件")

    # ── 2. 扫描本地目录，过滤已上传文件 ──────────────────────────────────
    if not os.path.isdir(local_dir):
        raise ValueError(f"local_dir 不存在或不是目录: {local_dir}")

    all_local_files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]
    to_upload = [f for f in all_local_files if f not in remote_filenames]

    skipped = len(all_local_files) - len(to_upload)
    print(f"[INFO] 本地共 {len(all_local_files)} 个文件，跳过 {skipped} 个，待上传 {len(to_upload)} 个")

    if not to_upload:
        print("[INFO] 无需上传，退出。")
        return

    # ── 3. 逐文件上传 ────────────────────────────────────────────────────
    success, failed = [], []
    for idx, filename in enumerate(to_upload, 1):
        local_path = os.path.join(local_dir, filename)
        repo_path = f"{repo_dir.rstrip('/')}/{filename}"

        print(f"[{idx}/{len(to_upload)}] 上传: {filename} -> {repo_path}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=repo_path,
                repo_id=repo_id,
                repo_type="dataset",
            )
            success.append(filename)
        except Exception as e:
            print(f"  [ERROR] 上传失败: {e}")
            failed.append((filename, str(e)))

    # ── 4. 汇总 ──────────────────────────────────────────────────────────
    print(f"\n[DONE] 成功 {len(success)} / 失败 {len(failed)}")
    if failed:
        print("[FAILED FILES]")
        for name, reason in failed:
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    sync_upload_to_modelscope(
        namespace="kakarotter",
        dataset="Calix-Dataset",
        repo_dir="pretraining/tokenize/1.12B/train",
        local_dir="F:/transformer-decoder/pretraining/1.12B/tokenize/train",
    )
