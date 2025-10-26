import os
import subprocess
import pandas as pd
import argparse


def safe_repo_dir(repo: str, commit: str) -> str:
    repo_name = repo.replace("/", "_")
    return f"{repo_name}_{commit}"


def clone_and_checkout(repo: str, commit: str, outdir: str):
    repo_url = f"https://github.com/{repo}.git"
    repo_dir = safe_repo_dir(repo, commit)
    repo_path = os.path.join(outdir, repo_dir)

    if os.path.exists(repo_path):
        print(f"[skip] {repo}@{commit} already exists at {repo_path}")
        return

    subprocess.run(["git", "clone", "--no-checkout", repo_url, repo_path], check=True)
    subprocess.run(["git", "checkout", commit], cwd=repo_path, check=True)

    print(f"[done] {repo}@{commit} cloned into {repo_path}")


def main(parquet_file, output_dir, limit=None):
    df = pd.read_parquet(parquet_file)
    os.makedirs(output_dir, exist_ok=True)

    for i, row in df.iterrows():
        repo = row["repo"]
        commit = row["base_commit"]

        try:
            clone_and_checkout(repo, commit, output_dir)
        except Exception as e:
            print(f"[error] {repo}@{commit}: {e}")

        if limit and i + 1 >= limit:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True, help="Path to swe-bench verified parquet")
    parser.add_argument("--output", default="/repos", help="Where to store cloned repos")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of repos to clone"
    )
    args = parser.parse_args()

    main(args.parquet, args.output, args.limit)

