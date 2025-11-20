from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Iterable, Callable, Tuple, Any
from pathlib import Path
import re
import json
import os
from code_tree import CodeSemanticTree
from ast_parser import ProjectParser



def run_example(project_path: str | Path, summarize: bool = False):

    parser = ProjectParser(str(project_path))
    parser.build_index()
    print("Index built!")

    tree = CodeSemanticTree(parser, project_name=parser.root.name)
    tree.build_from_parser_with_folder(parser)

    print("Tree overview:")
    tree.display()

    # collect context example
    ctx = tree.collect_context(query="create a code_tree tree from project source code", token_budget=1200)
    print('\nCollected context nodes:')
    for nid, text in ctx.items():
        node = tree.get(nid)
        print(f"- {node.kind} {node.name}")

    if summarize:
        print("\nGenerating summaries (this may call an external LLM if configured)...")
        try:
            tree.generate_summary()
        except Exception as e:
            print(f"[WARN] Summary generation failed: {e}")

        print("\nExample summaries:")
        shown = 0
        for nid in tree.iter_dfs("root"):
            node = tree.nodes[nid]
            if node.summary:
                print(f"- {node.kind} {node.name}: {node.summary[:200]}")
                shown += 1
            if shown >= 10:
                break

    out_json = Path("optimized_gen.json")
    tree.save_json(str(out_json), include_source=True)
    
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Example usage of CodeSemanticTree")
    p.add_argument("project", nargs="?", default="../example_repo",
                help="path to the project to index (default: ../example_repo)")
    p.add_argument("--summarize", action="store_true",
                help="Generate LLM summaries for nodes (uses OPENAI_API_KEY if set, otherwise a fallback summarizer)")
    args = p.parse_args()

    run_example(args.project, summarize=args.summarize)
