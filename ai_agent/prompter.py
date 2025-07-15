#!/usr/bin/env python3
"""
Comprehensive Prompter Script for Isaac Lab Task Generation

This script uses a local DeepSeek model to generate complete Isaac Lab manipulation tasks
from high-level descriptions. It populates all template files with task-specific code.

Usage:
    python prompter.py "pick up a ball and drop it into a cup"
    python prompter.py "move a block from top drawer to bottom drawer"
"""

import sys
import os
import re
import requests
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def use_deepseek(prompt: str) -> str:
    """Call the local DeepSeek model with the given prompt."""
    payload = {
        "model": "deepseek-r1:latest",
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_ctx": 130000,  # Use full context window
            "temperature": 0.1,  # Low temperature for consistent code generation
        }
    }
    
    try:
        print("Sending request to DeepSeek model...")
        print(f"Prompt length: {len(prompt):,} characters")
        print("This may take several minutes for complex tasks...")
        
        # Add timeout and show progress
        import time
        start_time = time.time()
        
        # Make the request with a longer timeout
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=payload, 
            timeout=1800  # 30 minute timeout
        )
        
        elapsed = time.time() - start_time
        print(f"Request completed in {elapsed:.1f} seconds")
        
        response.raise_for_status()
        result = response.json()["response"]
        
        print(f"Response length: {len(result):,} characters")
        return result
        
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model may be taking too long to respond.")
        print("Try reducing the task complexity or check if the model is running properly.")
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to DeepSeek model.")
        print("Make sure Ollama is running and the model is loaded:")
        print("  ollama serve")
        print("  ollama run deepseek-r1:latest")
        sys.exit(1)
    except Exception as e:
        print(f"Error calling DeepSeek model: {e}")
        sys.exit(1)


def use_deepseek_streaming(prompt: str) -> str:
    """Alternative streaming version that shows real-time progress."""
    payload = {
        "model": "deepseek-r1:latest",
        "prompt": prompt,
        "stream": True,  # Enable streaming
        "options": {
            "num_ctx": 130000,
            "temperature": 0.1,
        }
    }
    
    try:
        print("Sending streaming request to DeepSeek model...")
        print(f"Prompt length: {len(prompt):,} characters")
        print("Generating response (streaming)...")
        print("-" * 60)
        
        response = requests.post(
            "http://localhost:11434/api/generate", 
            json=payload, 
            stream=True,
            timeout=1800
        )
        response.raise_for_status()
        
        full_response = ""
        char_count = 0
        
        for line in response.iter_lines():
            if line:
                try:
                    chunk = json.loads(line)
                    if "response" in chunk:
                        token = chunk["response"]
                        full_response += token
                        char_count += len(token)
                        
                        # Show progress every 100 characters
                        if char_count % 100 == 0:
                            print(f"Generated: {char_count:,} characters", end="\r")
                        
                        # Show some of the actual output
                        if char_count % 1000 == 0:
                            print(f"\nProgress: {char_count:,} chars - Last text: {token.strip()}")
                            
                    if chunk.get("done", False):
                        break
                        
                except json.JSONDecodeError:
                    continue
        
        print(f"\nGeneration complete! Total: {len(full_response):,} characters")
        return full_response
        
    except Exception as e:
        print(f"\nError with streaming: {e}")
        print("Falling back to non-streaming mode...")
        return use_deepseek(prompt)


def load_template_files() -> Dict[str, str]:
    """Load all template files and their contents recursively from the template directory."""
    # Get absolute path to script directory
    script_path = Path(__file__).resolve()  # Get absolute path to prompter.py
    script_dir = script_path.parent         # ai_agent/
    project_root = script_dir.parent        # DynaJump/
    template_dir = project_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "manipulation" / "template"
    
    print(f"Script file: {script_path}")
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Looking for template directory: {template_dir}")
    
    if not template_dir.exists():
        print(f"Template directory not found: {template_dir}")
        print("\nExpected directory structure:")
        print(f"  {project_root}/")
        print(f"  ├── ai_agent/")
        print(f"  │   └── prompter.py  (this script)")
        print(f"  └── source/")
        print(f"      └── isaaclab_tasks/")
        print(f"          └── isaaclab_tasks/")
        print(f"              └── manager_based/")
        print(f"                  └── manipulation/")
        print(f"                      └── template/  (needed)")
        print("\nPlease create the template directory and add your template files.")
        sys.exit(1)
    
    print(f"✓ Found template directory: {template_dir}")
    
    template_files = {}
    
    # Recursively walk through the template directory
    for file_path in template_dir.rglob("*.py"):
        # Get relative path from template directory
        relative_path = file_path.relative_to(template_dir)
        relative_path_str = str(relative_path).replace("\\", "/")  # Normalize path separators
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                template_files[relative_path_str] = content
                print(f"  ✓ Loaded: {relative_path_str} ({len(content)} chars)")
        except Exception as e:
            print(f"  ✗ Error loading {relative_path_str}: {e}")
            continue
    
    if not template_files:
        print("No .py template files found in the template directory!")
        print(f"Checked directory: {template_dir}")
        print("Make sure you have created the template files with .py extensions.")
        sys.exit(1)
        
    print(f"✓ Successfully loaded {len(template_files)} template files")
    return template_files


def create_comprehensive_prompt(task_description: str, template_files: Dict[str, str]) -> str:
    """Create a comprehensive prompt for the DeepSeek model using XML structure."""
    
    prompt = f"""<task_generation>
<role>
You are an expert robotics engineer specializing in Isaac Lab manipulation tasks. Your expertise includes:
- Robotic manipulation and control systems
- Reward function design for reinforcement learning
- Physics simulation and environment setup
- Isaac Lab framework and conventions
- Sequential task decomposition and sub-problem solving
</role>

<task_description>
{task_description}
</task_description>

<system_constraints>
<robot>
- Type: Franka Panda arm with parallel gripper
- Reach: Maximum 0.4m radius from base position
- Control: Position and force control capabilities
</robot>

<environment>
- Base: Always includes a table surface
- Objects: Must be positioned within robot reach
- Physics: Realistic object interactions and constraints
- Coordinate system: Positions given as center of mass
</environment>

<framework>
- Platform: Isaac Lab simulation environment
- Import restrictions: Use only imports provided in template files
- Naming convention: snake_case for task names
- Structure: Follow Isaac Lab task organization patterns
</framework>
</system_constraints>

<analysis_framework>
<step1_scene_analysis>
- Identify required objects and their properties
- Determine optimal object placement and orientations
- Consider environmental constraints and physics
- Plan initial scene configuration
</step1_scene_analysis>

<step2_task_decomposition>
- Break task into sequential sub-problems
- Identify key manipulation primitives needed
- Define logical progression through task phases
- Consider failure modes and recovery strategies
</step2_task_decomposition>

<step3_reward_design>
- Design dense reward signals for each sub-problem
- Create smooth reward transitions between phases
- Balance exploration vs exploitation incentives
- Include progress tracking and success metrics
</step3_reward_design>

<step4_observation_space>
- Determine critical state information needed
- Include object poses, robot state, and task progress
- Consider sensor limitations and noise
- Optimize for learning efficiency
</step4_observation_space>

<step5_termination_criteria>
- Define clear success conditions
- Set reasonable failure conditions
- Include timeout and safety terminations
- Handle edge cases and invalid states
</step5_termination_criteria>
</analysis_framework>

<examples>
<example_simple>
<description>pick up a ball</description>
<expansion>
<scene>table + spherical ball object</scene>
<sub_problems>
1. approach_ball: Move end-effector near ball
2. align_gripper: Orient gripper for optimal grasp
3. close_gripper: Execute grasp with appropriate force
4. lift_ball: Raise ball above table surface
</sub_problems>
<rewards>
- Distance-based reward for approaching ball
- Orientation reward for gripper alignment
- Grasp success reward for secure grip
- Height reward for successful lift
</rewards>
</expansion>
</example_simple>

<example_complex>
<description>move block from top drawer to bottom drawer</description>
<expansion>
<scene>cabinet with two drawers + block object in top drawer</scene>
<sub_problems>
1. open_top_drawer: Pull handle to access block
2. grasp_block: Secure grip on block
3. lift_block: Remove block from drawer
4. close_top_drawer: Return drawer to closed position
5. open_bottom_drawer: Access destination location
6. place_block: Position block in bottom drawer
7. close_bottom_drawer: Complete task
</sub_problems>
<rewards>
- Drawer opening progress rewards
- Grasp success and stability rewards
- Block positioning and placement rewards
- Task completion and efficiency rewards
</rewards>
</expansion>
</example_complex>
</examples>

<code_generation_requirements>
<completeness>
- Generate complete, executable code for ALL files
- No placeholders, comments, or partial implementations
- All template variables must be replaced with task-specific values
- Include proper error handling and edge case management
</completeness>

<quality_standards>
- Follow Isaac Lab coding conventions and style
- Use descriptive variable names and clear comments
- Implement robust reward functions with proper scaling
- Include comprehensive observation and action spaces
- Set appropriate episode lengths for task complexity
</quality_standards>

<file_specifications>
- __init__.py: Proper module imports and exports
- env_cfg.py: Complete environment configuration
- observations.py: Comprehensive state observation functions
- rewards.py: Dense, shaped reward function implementations
- terminations.py: Success, failure, and timeout conditions
- joint_pos_env_cfg.py: Robot-specific configuration
- rsl_rl_ppo_cfg.py: Training algorithm parameters
</file_specifications>
</code_generation_requirements>

<template_files>
The following template files provide the structure you must populate:

"""

    # Add all template files to the prompt with XML structure
    for file_path, content in template_files.items():
        prompt += f"""<template_file>
<path>{file_path}</path>
<content>
{content}
</content>
</template_file>

"""
    
    prompt += f"""</template_files>

<output_format>
You must generate complete code for ALL template files using this exact format:

<task_name>your_chosen_task_name</task_name>

<generated_files>
<file>
<path>__init__.py</path>
<content>
[complete file content with all imports and code]
</content>
</file>

<file>
<path>task_name/__init__.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/task_name_env_cfg.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/mdp/__init__.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/mdp/observations.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/mdp/rewards.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/mdp/terminations.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/config/__init__.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/config/franka/__init__.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/config/franka/joint_pos_env_cfg.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/config/franka/agents/__init__.py</path>
<content>
[complete file content]
</content>
</file>

<file>
<path>task_name/config/franka/agents/rsl_rl_ppo_cfg.py</path>
<content>
[complete file content]
</content>
</file>
</generated_files>

<implementation_notes>
- Replace 'task_name' with your chosen task name in all file paths
- Ensure all template placeholders are replaced with actual implementations
- Generate working Isaac Lab code that follows framework conventions
- Create reward functions that guide the robot through logical sub-problems
- Include proper termination conditions for success, failure, and timeouts
- Set reasonable episode lengths based on task complexity
- All objects must be positioned within 0.4m radius of robot base
- Use only imports present in the provided template files
</implementation_notes>
</task_generation>

Now analyze the task description and generate the complete Isaac Lab manipulation task:"""
    
    return prompt


def parse_llm_output(llm_output: str) -> Tuple[Dict[str, str], str]:
    """Parse the LLM output to extract file contents using XML structure."""
    files = {}
    
    # Extract task name using XML tags
    task_name_match = re.search(r'<task_name>(.*?)</task_name>', llm_output, re.DOTALL)
    if not task_name_match:
        print("Error: Could not find <task_name> in LLM output")
        print("Looking for alternative patterns...")
        # Fallback to original pattern
        task_name_match = re.search(r'TASK_NAME:\s*(\w+)', llm_output)
        if not task_name_match:
            print("Error: Could not find task name in any format")
            return {}, ""
    
    task_name = task_name_match.group(1).strip()
    print(f"Generated task name: {task_name}")
    
    # Extract files using XML structure
    file_pattern = r'<file>\s*<path>(.*?)</path>\s*<content>(.*?)</content>\s*</file>'
    matches = re.findall(file_pattern, llm_output, re.DOTALL)
    
    if not matches:
        print("No XML file format found, trying fallback pattern...")
        # Fallback to original pattern
        file_pattern = r'FILE:\s*([^\n]+)\n(.*?)(?=ENDFILE|$)'
        matches = re.findall(file_pattern, llm_output, re.DOTALL)
    
    for file_path, content in matches:
        file_path = file_path.strip()
        content = content.strip()
        
        # Replace task_name placeholder with actual task name
        file_path = file_path.replace('task_name', task_name)
        
        files[file_path] = content
    
    return files, task_name


def create_task_directory(task_name: str, files: Dict[str, str]) -> None:
    """Create the task directory structure and write files."""
    # Get absolute paths
    script_path = Path(__file__).resolve()  # Get absolute path to prompter.py
    script_dir = script_path.parent         # ai_agent/
    project_root = script_dir.parent        # DynaJump/
    base_dir = project_root / "source" / "isaaclab_tasks" / "isaaclab_tasks" / "manager_based" / "manipulation"
    task_dir = base_dir / task_name
    
    # Remove existing directory if it exists
    if task_dir.exists():
        print(f"Removing existing task directory: {task_dir}")
        shutil.rmtree(task_dir)
    
    # Create directory structure
    print(f"Creating task directory: {task_dir}")
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Create all necessary subdirectories
    (task_dir / "mdp").mkdir(exist_ok=True)
    (task_dir / "config").mkdir(exist_ok=True) 
    (task_dir / "config" / "franka").mkdir(exist_ok=True)
    (task_dir / "config" / "franka" / "agents").mkdir(exist_ok=True)
    
    # Write files
    for file_path, content in files.items():
        if file_path.startswith(f'{task_name}/'):
            # Remove task_name prefix for relative path within task directory
            relative_path = file_path[len(task_name)+1:]
            full_path = task_dir / relative_path
        else:
            # Root level files
            full_path = base_dir / file_path
        
        print(f"Writing file: {full_path}")
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    print(f"\nTask '{task_name}' generated successfully!")
    print(f"Location: {task_dir}")
    print("\nGenerated files:")
    for file_path in sorted(files.keys()):
        print(f"  - {file_path}")


def main():
    """Main function to orchestrate the task generation process."""
    if len(sys.argv) != 2:
        print("Usage: python prompter.py \"<task_description>\"")
        print("Example: python prompter.py \"pick up a ball and drop it into a cup\"")
        sys.exit(1)
    
    task_description = sys.argv[1]
    print(f"Generating Isaac Lab task for: {task_description}")
    
    # Load template files
    print("Loading template files...")
    template_files = load_template_files()
    
    # Create comprehensive prompt
    print("Creating XML-structured prompt for DeepSeek model...")
    prompt = create_comprehensive_prompt(task_description, template_files)
    
    # Save prompt for debugging
    with open("debug_prompt.txt", "w", encoding='utf-8') as f:
        f.write(prompt)
    print(f"Prompt saved to debug_prompt.txt (length: {len(prompt):,} chars)")
    
    # Call DeepSeek model
    print("Calling DeepSeek model (this may take a while)...")
    
    # Ask user which mode they prefer
    print("Choose generation mode:")
    print("1. Standard mode (shows elapsed time)")
    print("2. Streaming mode (shows real-time progress)")
    print("3. Test mode (generate prompt only, don't call model)")
    
    mode = input("Enter choice (1, 2, or 3), or press Enter for standard: ").strip()
    
    if mode == "3":
        print("\n" + "="*60)
        print("TEST MODE: XML-structured prompt generated successfully!")
        print("="*60)
        print(f"Prompt length: {len(prompt):,} characters")
        print(f"Prompt saved to: debug_prompt.txt")
        print("\nPrompt structure includes:")
        print("- XML-based instruction format")
        print("- Clear role definition and expertise areas")
        print("- Structured analysis framework")
        print("- Comprehensive examples and requirements")
        print("- Improved output format specification")
        print("\nTo test the actual model call, run again with mode 1 or 2.")
        print("Exiting...")
        return
    elif mode == "2":
        llm_output = use_deepseek_streaming(prompt)
    else:
        llm_output = use_deepseek(prompt)
    
    # Save raw output for debugging
    with open("debug_output.txt", "w", encoding='utf-8') as f:
        f.write(llm_output)
    print(f"Raw output saved to debug_output.txt (length: {len(llm_output):,} chars)")
    
    # Parse LLM output
    print("Parsing LLM output...")
    files, task_name = parse_llm_output(llm_output)
    
    if not files:
        print("Error: No files generated by LLM")
        print("Check debug_output.txt for the raw model response")
        sys.exit(1)
    
    # Create task directory and files
    print(f"Creating task directory structure...")
    create_task_directory(task_name, files)
    
    print("\n" + "="*60)
    print("TASK GENERATION COMPLETE!")
    print("="*60)
    print(f"Task: {task_description}")
    print(f"Generated task name: {task_name}")
    print(f"Files created: {len(files)}")
    print("\nNext steps:")
    print("1. Review the generated code for correctness")
    print("2. Test the environment configuration")
    print("3. Train the RL agent")
    print("4. Evaluate task performance")
    print("\nGenerated with improved XML-structured prompting for better results!")


if __name__ == "__main__":
    main()