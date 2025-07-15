#!/usr/bin/env python3
"""
Populate Isaac Lab task files from an existing DeepSeek LLM output.

Usage:
    python populate_from_output.py debug_output.txt
"""

import sys
import re
import shutil
from pathlib import Path
from typing import Dict


def parse_llm_output(llm_output: str) -> Dict[str, str]:
    """Parse the LLM output to extract file contents."""
    files = {}

    task_name_match = re.search(r'TASK_NAME:\s*(\w+)', llm_output)
    if not task_name_match:
        print("Error: Could not find TASK_NAME in LLM output")
        sys.exit(1)

    task_name = task_name_match.group(1)
    print(f"Generated task name: {task_name}")

    file_pattern = r'FILE:\s*([^\n]+)\n(.*?)(?=ENDFILE|$)'
    matches = re.findall(file_pattern, llm_output, re.DOTALL)

    for file_path, content in matches:
        file_path = file_path.strip()
        content = content.strip()
        file_path = file_path.replace('task_name', task_name)
        files[file_path] = content

    return files, task_name


def create_task_directory(task_name: str, files: Dict[str, str]) -> None:
    """Write parsed files to disk."""
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    base_dir = project_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "manipulation"
    task_dir = base_dir / task_name

    # Remove if exists
    if task_dir.exists():
        print(f"Removing existing directory: {task_dir}")
        shutil.rmtree(task_dir)

    print(f"Creating directory: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)

    # Write files
    for file_path, content in files.items():
        if file_path.startswith(f'{task_name}/'):
            relative_path = file_path[len(task_name) + 1:]
            full_path = task_dir / relative_path
        else:
            full_path = base_dir / file_path

        print(f"Writing: {full_path}")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)

    print(f"✓ Task '{task_name}' created at {task_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python populate_from_output.py <output_file>")
        sys.exit(1)

    output_file = sys.argv[1]
    with open(output_file, 'r', encoding='utf-8') as f:
        llm_output = f.read()

    files, task_name = parse_llm_output(llm_output)
    if not files:
        print("Error: No files to write.")
        sys.exit(1)

    create_task_directory(task_name, files)


if __name__ == "__main__":
    main()
