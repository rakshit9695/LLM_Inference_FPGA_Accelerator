#!/usr/bin/env python3
"""
Debug script to check Verilator executable location and paths
Run this to diagnose the path issue with your bridge
"""

import os
from pathlib import Path

def check_paths():
    print("=== VERILATOR BRIDGE PATH DEBUGGING ===")
    print()
    
    # Current working directory
    print(f"Current working directory: {os.getcwd()}")
    
    # Script location
    script_dir = Path(__file__).parent
    print(f"Script directory: {script_dir.resolve()}")
    
    # Check all possible executable locations
    possible_paths = [
        # From scripts/python/ directory
        script_dir / "../../sim/obj_dir/Vgemm_accelerator",
        script_dir / "../../sim/build/obj_dir/Vgemm_accelerator", 
        # From project root
        Path("sim/obj_dir/Vgemm_accelerator"),
        Path("sim/build/obj_dir/Vgemm_accelerator"),
        # Current directory
        Path("./obj_dir/Vgemm_accelerator"),
        Path("obj_dir/Vgemm_accelerator"),
    ]
    
    print("\n=== CHECKING POSSIBLE EXECUTABLE LOCATIONS ===")
    found_executable = None
    
    for i, path in enumerate(possible_paths, 1):
        exists = path.exists()
        status = "✅ EXISTS" if exists else "❌ NOT FOUND"
        resolved_path = path.resolve() if exists else "N/A"
        print(f"{i}. {path}")
        print(f"   Status: {status}")
        print(f"   Resolved: {resolved_path}")
        
        if exists and found_executable is None:
            found_executable = resolved_path
        print()
    
    # Check sim directory structure
    print("=== SIM DIRECTORY STRUCTURE ===")
    sim_paths = [
        Path("sim"),
        script_dir / "../../sim",
    ]
    
    for sim_path in sim_paths:
        if sim_path.exists():
            print(f"📁 {sim_path.resolve()} - EXISTS")
            try:
                contents = list(sim_path.iterdir())
                for item in sorted(contents):
                    if item.is_dir():
                        print(f"  📁 {item.name}/")
                        if item.name in ['obj_dir', 'build']:
                            try:
                                sub_contents = list(item.iterdir())
                                for sub_item in sorted(sub_contents):
                                    marker = "🚀" if sub_item.name == "Vgemm_accelerator" else "📄"
                                    print(f"    {marker} {sub_item.name}")
                            except PermissionError:
                                print(f"    ❌ Permission denied")
                    else:
                        print(f"  📄 {item.name}")
            except PermissionError:
                print(f"  ❌ Permission denied to read directory")
        else:
            print(f"📁 {sim_path} - NOT FOUND")
        print()
    
    # Recommendations
    print("=== RECOMMENDATIONS ===")
    if found_executable:
        print(f"✅ Found executable: {found_executable}")
        print("You can:")
        print("1. Use the auto-detection in verilator_bridge_fixed.py")
        print(f"2. Or specify the path explicitly: VerilatorGEMM('{found_executable}')")
    else:
        print("❌ No executable found. You need to:")
        print("1. Build the simulation first:")
        print("   cd sim")
        print("   make")
        print("2. Ensure the build completed successfully")
        print("3. Check that obj_dir/Vgemm_accelerator was created")
    
    print()
    print("=== DIRECTORY NAVIGATION TIPS ===")
    print("Current script location suggests you should either:")
    print("1. Run from project root:")
    print("   cd ../../  # Go to project root")
    print("   python scripts/python/verilator_bridge_fixed.py")
    print()
    print("2. Or use the auto-detection in the fixed script")
    
    return found_executable

if __name__ == "__main__":
    check_paths()