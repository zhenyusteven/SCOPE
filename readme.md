Our main pipeline is located in the `scripts` directory. Please refer to `scripts/readme.md` for detailed instructions on running different components of the pipeline (e.g., tree generation, context generation).

The `SWE-agent` directory is adapted from the public [SWE-agent repository](https://github.com/SWE-agent/SWE-agent).  
Our primary modifications include:

- Updates to configuration files:  
  - `context_verified.yaml`  
  - `context_verified_with_context.yaml`  
  - `context_verified_with_tree.yaml`
- Code changes enabling context text or tree files to be uploaded:  
  - See approximately lines 414–440 in `SWE-agent/sweagent/run/run_batch.py`  
  - Additional logic in `SWE-agent/sweagent/run/batch_instances.py`
