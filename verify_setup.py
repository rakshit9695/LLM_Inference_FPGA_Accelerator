import sys
import subprocess
import importlib

def check_python_packages():
    packages = [
        'torch', 'transformers', 'sentence_transformers', 
        'faiss', 'numpy', 'pandas', 'matplotlib', 'jupyter'
    ]
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package} installed successfully")
        except ImportError:
            print(f"✗ {package} not found")

def check_system_tools():
    tools = ['verilator', 'yosys', 'vvp']
    
    for tool in tools:
        try:
            result = subprocess.run([tool, '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✓ {tool} available")
            else:
                print(f"✗ {tool} not working properly")
        except FileNotFoundError:
            print(f"✗ {tool} not found")

if __name__ == "__main__":
    print("Checking Python packages...")
    check_python_packages()
    print("\nChecking system tools...")
    check_system_tools()
    print(f"\nPython version: {sys.version}")