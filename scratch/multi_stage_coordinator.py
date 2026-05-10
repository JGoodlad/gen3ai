import subprocess
import os
import time
import json
import sys

def run_training_stage(steps, model_path=None):
    # Optimized hyperparams for CPU training
    cmd = [
        "./deps/venv/bin/python3",
        "src/main/train_rl_agent.py",
        "--steps", str(steps),
        "--n-envs", "8",
        "--eval-battles", "200",
        "--batch-size", "512",
        "--n-epochs", "4",
        "--lr", "3e-4",
        "--n-steps", "2048"
    ]
    if model_path:
        cmd.extend(["--model", model_path])
    
    print(f"\n>>> STARTING STAGE: {steps} steps (Model: {model_path or 'NEW'})")
    start_time = time.time()
    env = os.environ.copy()
    env["PYTHONPATH"] = "src"
    
    # Use Popen to stream output
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.STDOUT, 
        text=True, 
        env=env,
        bufsize=1, # Line buffered
        universal_newlines=True
    )
    
    win_rate_random = "N/A"
    win_rate_heuristic = "N/A"
    saved_path = "Unknown"
    
    # Read output line by line
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if not line:
            continue
        print(f"  [STDOUT] {line}")
        
        if "Win rate vs Random:" in line:
            win_rate_random = line.split(":")[-1].strip()
        if "Win rate vs Heuristic:" in line:
            win_rate_heuristic = line.split(":")[-1].strip()
        if "Model saved to" in line:
            # Look for paths like models/gen3ou_ppo_new_20260510_124040/final_model
            # or models/gen3ou_ppo_continued_20260510_124040/final_model
            saved_path = line.split("to")[-1].strip()

    process.stdout.close()
    return_code = process.wait()
    duration = time.time() - start_time
    
    if return_code != 0:
        print(f"Error: Stage failed with return code {return_code}")
        return None
            
    return {
        "steps": steps,
        "duration": f"{duration:.1f}s",
        "win_rate_random": win_rate_random,
        "win_rate_heuristic": win_rate_heuristic,
        "saved_path": saved_path
    }

def main():
    # Milestones as defined by the user
    milestones = [1000, 10000, 100000, 250000]
    results = []
    current_model = None
    
    report_path = "scratch/training_progress_report.md"
    os.makedirs("scratch", exist_ok=True)
    
    with open(report_path, "w") as f:
        f.write("# Multi-Stage RL Training Progress Report\n\n")
        f.write("Generated on: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write("| Stage | Total Steps | Duration | Win Rate vs Random | Win Rate vs Heuristic | Model Path |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- |\n")

    for i, milestone in enumerate(milestones):
        if i == 0:
            step_increment = milestones[0]
        else:
            step_increment = milestones[i] - milestones[i-1]
            
        stage_result = run_training_stage(step_increment, current_model)
        if not stage_result:
            print(f"FAILED AT STAGE {milestone}")
            break
            
        results.append(stage_result)
        current_model = stage_result["saved_path"]
        
        with open(report_path, "a") as f:
            f.write(f"| {i+1} | {milestone} | {stage_result['duration']} | {stage_result['win_rate_random']} | {stage_result['win_rate_heuristic']} | `{stage_result['saved_path']}` |\n")
        
        print(f"\n✅ STAGE {i+1} COMPLETE ({milestone} total steps)")
        print(f"   Win Rate vs Random: {stage_result['win_rate_random']}")
        print(f"   Win Rate vs Heuristic: {stage_result['win_rate_heuristic']}")
        print(f"----------------------------------------\n")

    print(f"\nAll stages complete. Final report available at {report_path}")

if __name__ == "__main__":
    main()
