# SWE-bench Automation Scripts

This folder contains the command-line tooling for running the SCOPE code-tree
pipeline on SWE-bench datasets. The entry point is
`swe_bench_pipeline.py`, along with a Docker wrapper.

## Commands
-  `clone`: Download repositories for the selected SWE-bench subset/split. 
- `trees`: Build semantic trees for each instance (optionally cloning automatically and cleaning up repos afterwards). 
- `evaluate`: Generate predictions, then optionally submit to SWE-bench via `sb-cli`. 
- `pipeline` Run `trees` followed by `evaluate` in a single invocation. 

Common arguments (all commands):

- `--subset {verified,lite,full,...}` choose the SWE-bench subset. (default = verified)
- `--split {dev,test}` choose dataset split. (default = test, verified only has test)
- `--limit N` restrict processing to the first `N` instances after filtering.
- `--filter REGEX` include only matching instance IDs.

Tree-specific arguments (`trees`, `pipeline`):
Defaults all set to `True`
- `--auto-clone` automatically clone repos that are missing in `--repos-dir`.
- `--cleanup-repos` delete the cloned repo after the JSON tree is saved.
- `--skip-existing` skip tree generation if the JSON already exists.
- `--summarize` run the summarizer (requires OpenAI creds).
- `--include-source` embed source text in the JSON.

Evaluation arguments (`evaluate`, `pipeline`):
Defaults are set other than patches dir
- `--trees-dir` location of tree JSONs to feed into the codegen hook.
- `--patches-dir` directory with `<instance_id>.patch` files (optional).
- `--predictions-dir` output for individual `.pred` files. 
- `--output-dir` location for `preds.json` and sb-cli reports.
- `--run-name` label stored in the prediction entries.
- `--run-sb-cli` run `sb-cli submit` after generating predictions (requires `SB_API_KEY` or similar env var).

## Code Generation
`generate_patch_from_tree(instance, tree_file)` encapsulates the code
generation step. The current implementation simply returns the ground-truth
patch from the dataset. Any `evaluate` or `pipeline` run that does not specify `--patches-dir`
will go through this function.

## Environment Variables
include `OPENAI_API_KEY` and `SWEBENCH_API_KEY` in `SCOPE/.env` for code generation and evaluation. Swe bench key can be made following instructions [here](https://www.swebench.com/sb-cli/quick-start/)

## Docker Usage

After building the image (`docker build -t scope-swe .` at repo root), you can run:

```bash
# Build trees
docker run --rm \
  -v "$PWD/data/swe_bench:/app/data/swe_bench" \
  --env-file .env \
  scope-swe trees \
    --subset verified --split test --limit 2 \

# Generate context
docker run --rm \
  -v "$PWD/data/swe_bench:/app/data/swe_bench" \
  --env-file .env \
  scope-swe context \
    --subset verified --split test --limit 2 \

# Generate predictions and submit via sb-cli
docker run --rm \
  -v "$PWD/data/swe_bench:/app/data/swe_bench" \
  --env-file .env \
  scope-swe evaluate \
    --subset verified --split test --limit 2 \
```

Or run everything end-to-end with the pipeline command:

```bash
docker run --rm \
  -v "$PWD/data/swe_bench:/app/data/swe_bench" \
  --env-file .env \
  scope-swe pipeline \
    --subset verified --split test --limit 1 \
```
