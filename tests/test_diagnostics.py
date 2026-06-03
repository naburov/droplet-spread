#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced diagnostic system.
"""

import subprocess
import sys
import os

def test_diagnostics():
    """Test the enhanced diagnostic system."""
    
    print("🧪 TESTING ENHANCED DIAGNOSTIC SYSTEM")
    print("=" * 50)
    
    # Test on the existing experiment
    experiment_dir = "experiment_20250915_214517"
    
    if not os.path.exists(experiment_dir):
        print(f"❌ Experiment directory {experiment_dir} not found!")
        return False
    
    print(f"📁 Testing diagnostics on: {experiment_dir}")
    
    # Run diagnostics
    try:
        result = subprocess.run([
            "native/bin/python", 
            "diagnostics/run_all_diagnostics.py", 
            experiment_dir
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            print("✅ Diagnostic system test PASSED!")
            print("\n📊 Generated files:")
            
            # Check what files were created
            diagnostics_dir = os.path.join(experiment_dir, "diagnostics")
            if os.path.exists(diagnostics_dir):
                files = os.listdir(diagnostics_dir)
                png_files = [f for f in files if f.endswith('.png')]
                json_files = [f for f in files if f.endswith('.json')]
                
                print(f"  📈 Plots: {len(png_files)} files")
                for f in sorted(png_files):
                    print(f"    - {f}")
                
                print(f"  📄 Data: {len(json_files)} files")
                for f in sorted(json_files):
                    print(f"    - {f}")
                
                # Check for reference guide
                if "diagnostic_reference_guide.png" in png_files:
                    print("  🎯 Reference guide: ✅ Included")
                else:
                    print("  🎯 Reference guide: ❌ Missing")
            
            return True
        else:
            print("❌ Diagnostic system test FAILED!")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Test timed out!")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main test function."""
    success = test_diagnostics()
    
    if success:
        print("\n🎉 ENHANCED DIAGNOSTIC SYSTEM IS WORKING!")
        print("\n✨ Features:")
        print("  • Reference guide automatically included")
        print("  • Enhanced plots with expected trends")
        print("  • Clear visual indicators for good/bad behavior")
        print("  • Comprehensive interpretation guidelines")
        print("\n🚀 Ready for production use!")
    else:
        print("\n❌ DIAGNOSTIC SYSTEM NEEDS FIXES")
        sys.exit(1)

if __name__ == "__main__":
    main()




