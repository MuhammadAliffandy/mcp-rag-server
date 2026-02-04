#!/usr/bin/env python3
"""
Auto Dependency Installer
Automatically installs missing Python packages when ImportError occurs
"""

import subprocess
import sys
import os

def auto_install_dependency(package_name):
    """
    Automatically install a missing package using pip.
    
    Args:
        package_name: Name of the package to install
        
    Returns:
        bool: True if installation successful, False otherwise
    """
    try:
        # Get the python executable from current environment
        python_exe = sys.executable
        
        # Run pip install
        print(f"🔧 Auto-installing missing package: {package_name}")
        result = subprocess.run(
            [python_exe, "-m", "pip", "install", package_name],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes max
        )
        
        if result.returncode == 0:
            print(f"✅ Successfully installed {package_name}")
            return True
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during installation: {e}")
        return False

def try_import_with_auto_install(package_name, import_name=None):
    """
    Try to import a package, auto-install if missing.
    
    Args:
        package_name: Package name for pip (e.g., 'scikit-learn')
        import_name: Import name if different from package name (e.g., 'sklearn')
        
    Returns:
        module: The imported module if successful, None otherwise
    """
    if import_name is None:
        import_name = package_name
        
    try:
        # Try importing
        module = __import__(import_name)
        return module
    except ImportError:
        # Auto-install if missing
        print(f"⚠️ Package '{import_name}' not found. Installing '{package_name}'...")
        
        if auto_install_dependency(package_name):
            try:
                # Try importing again after installation
                module = __import__(import_name)
                print(f"✅ Successfully imported {import_name} after installation")
                return module
            except ImportError as e:
                print(f"❌ Still cannot import {import_name} after installation: {e}")
                return None
        else:
            return None

# Common package mappings (pip name -> import name)
PACKAGE_MAPPINGS = {
    'scikit-learn': 'sklearn',
    'opencv-python': 'cv2',
    'python-docx': 'docx',
    'pillow': 'PIL',
    'statsmodels': 'statsmodels',
    'shap': 'shap',
    'catboost': 'catboost',
    'optuna': 'optuna',
}

if __name__ == "__main__":
    # Test the auto-installer
    print("Testing auto-dependency installer...")
    
    # Test with a common package
    result = try_import_with_auto_install('statsmodels')
    if result:
        print("✅ Test passed!")
    else:
        print("❌ Test failed!")
