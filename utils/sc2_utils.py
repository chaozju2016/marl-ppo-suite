"""
Utility functions for handling StarCraft II processes.
"""
import os
import signal
import subprocess
import time
import platform


def kill_sc2_processes():
    """
    Kill all running StarCraft II processes.
    
    This is useful when SC2 processes don't shut down properly and prevent
    new environments from connecting.
    
    Returns:
        int: Number of processes killed
    """
    system = platform.system()
    killed_count = 0
    
    try:
        if system == "Darwin":  # macOS
            # Get all SC2 process IDs
            output = subprocess.check_output(
                "ps -A | grep 'SC2' | grep -v grep | awk '{print $1}'", 
                shell=True, 
                text=True
            ).strip()
            
            if output:
                pids = output.split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        killed_count += 1
                        print(f"Killed SC2 process with PID {pid}")
                    except (ProcessLookupError, ValueError):
                        pass
                
                # Give processes time to terminate
                time.sleep(1)
                
        elif system == "Windows":
            # Windows implementation
            output = subprocess.check_output(
                "tasklist /FI \"IMAGENAME eq SC2*\" /NH", 
                shell=True, 
                text=True
            )
            
            if "SC2" in output:
                subprocess.call("taskkill /F /IM SC2*", shell=True)
                killed_count = output.count("SC2")
                print(f"Killed {killed_count} SC2 processes")
                time.sleep(1)
                
        elif system == "Linux":
            # Linux implementation
            output = subprocess.check_output(
                "ps -A | grep 'SC2' | grep -v grep | awk '{print $1}'", 
                shell=True, 
                text=True
            ).strip()
            
            if output:
                pids = output.split('\n')
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        killed_count += 1
                        print(f"Killed SC2 process with PID {pid}")
                    except (ProcessLookupError, ValueError):
                        pass
                
                # Give processes time to terminate
                time.sleep(1)
    
    except Exception as e:
        print(f"Error while trying to kill SC2 processes: {e}")
    
    return killed_count
