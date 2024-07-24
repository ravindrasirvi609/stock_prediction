import subprocess
from pathlib import Path

def run_script(script_path):
    result = subprocess.run(['python', script_path], check=True)
    if result.returncode != 0:
        raise Exception(f"Script {script_path} failed with return code {result.returncode}")

if __name__ == "__main__":
    src_dir = Path(__file__).resolve().parents[1] / 'src' / 'data'
    
    print("Running data collection...")
    run_script(src_dir / 'collection.py')
    
    print("Running data processing...")
    run_script(src_dir / 'processing.py')
    
    print("Data pipeline completed successfully!")