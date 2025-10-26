# SWE-Bench Repo Cloner (Docker)

This setup builds a Docker image that clones repositories listed in a SWE-Bench parquet file.  
Repositories are stored in the format `<repo_author>_<repo_name>_<commit_hash>`.

---

## Build the image

From the `setup/` directory:

```bash
docker build -t swe-cloner .
```

---

## Run the container

To clone repos into a local folder:

```bash
docker run --rm \
  -v /<your_path_to_this_repo>/ReCAP-SWE/data:/data \
  -v /<your_path_to_this_repo>/ReCAP-SWE/repos:/repos \
  swe-cloner \
  python clone_repos.py \
    --parquet /data/test-00000-of-00001.parquet \
    --output /repos \
    --limit <number of repos you want to download, defaults to downloading all!!>
```

## Notes

- If you omit the `-v /Users/.../repos:/repos` volume, cloned repos will be automatically deleted when the container exits.  e.g.: 
```bash
docker run --rm \
  -v /Users/Likehan/Desktop/cs329a_hw/ReCAP-SWE/data:/data \
  swe-cloner \
  python clone_repos.py \
    --parquet /data/test-00000-of-00001.parquet \
    --output /repos \
    --limit 3
```
- Use `--limit N` to only clone the first `N` repos for testing!! 
