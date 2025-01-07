import subprocess
import argparse
from utils.config import get_run_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor', action='store_true', help='Start dashboard monitoring')
    args = parser.parse_args()

    run_id = get_run_id()
    
    # Start training
    training_process = subprocess.Popen(['python', 'VIZDOOM-STAGE1.py', '--run_id', run_id])
    
    if args.monitor:
        # Start dashboard in monitoring mode
        dashboard_process = subprocess.Popen(['python', 'DASHBOARD1.py', '--mode', 'monitor', '--run_id', run_id])
        
        training_process.wait()
        dashboard_process.wait()
    else:
        training_process.wait()

if __name__ == "__main__":
    main()
