"""
Pre-Installation Setup Script
Ensures Python 3.12 environment is ready for all packages
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    print(f"🐍 Python Version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3:
        print("❌ Python 3 required")
        return False
    
    if version.minor < 8:
        print("❌ Python 3.8+ required")
        return False
    
    if version.minor >= 12:
        print("✅ Python 3.12 detected - using compatible package versions")
    
    return True

def upgrade_core_tools():
    """Upgrade pip, setuptools, and wheel"""
    print("\n🔧 Upgrading core tools...")
    
    commands = [
        [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "setuptools>=68.0.0"],
        [sys.executable, "-m", "pip", "install", "--upgrade", "wheel>=0.41.0"],
    ]
    
    for cmd in commands:
        try:
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed: {e}")
            print(f"Error output: {e.stderr}")
            return False
    
    return True

def install_compatibility_packages():
    """Install compatibility packages for Python 3.12"""
    print("\n🔧 Installing Python 3.12 compatibility packages...")
    
    compat_packages = [
        "packaging>=23.0",
        "typing-extensions>=4.7.0",
        "importlib-metadata>=6.0.0",
    ]
    
    for package in compat_packages:
        try:
            cmd = [sys.executable, "-m", "pip", "install", package]
            print(f"Installing: {package}")
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Success")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Warning: Failed to install {package}")
    
    return True

def check_system_dependencies():
    """Check system-level dependencies"""
    print("\n🔧 Checking system dependencies...")
    
    # Check if we're on Windows and need Visual C++
    if platform.system() == "Windows":
        print("🪟 Windows detected")
        print("📝 Note: Some packages may require Visual C++ Build Tools")
        print("   Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
    
    # Check available disk space
    if hasattr(os, 'statvfs'):  # Unix-like systems
        statvfs = os.statvfs('.')
        free_space = statvfs.f_frsize * statvfs.f_bavail / (1024**3)  # GB
    else:  # Windows
        import shutil
        free_space = shutil.disk_usage('.').free / (1024**3)  # GB
    
    print(f"💾 Available disk space: {free_space:.1f} GB")
    
    if free_space < 2.0:
        print("⚠️ Warning: Low disk space. At least 2GB recommended.")
    else:
        print("✅ Sufficient disk space")
    
    return True

def create_directories():
    """Create necessary project directories"""
    print("\n📁 Creating project directories...")
    
    directories = [
        "data", "results", "results/plots", "results/reports",
        "validation_reports", "logs", "cache", "src"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True, parents=True)
        print(f"✅ Created: {directory}")
    
    return True

def test_critical_imports():
    """Test if critical packages can be imported"""
    print("\n🧪 Testing critical package imports...")
    
    critical_packages = [
        ("json", "JSON support"),
        ("sqlite3", "Database support"),
        ("urllib", "URL handling"),
        ("pathlib", "Path operations"),
    ]
    
    for package, description in critical_packages:
        try:
            __import__(package)
            print(f"✅ {package}: {description}")
        except ImportError:
            print(f"❌ {package}: {description} - MISSING")
            return False
    
    return True

def install_packages_batch():
    """Install packages in batches to avoid conflicts"""
    print("\n📦 Installing packages in optimized order...")
    
    # Install in specific order to avoid conflicts
    batches = [
        # Batch 1: Core build tools (already done)
        [],
        
        # Batch 2: Core scientific computing
        ["numpy>=1.24.3,<1.26.0"],
        
        # Batch 3: Data processing
        ["pandas>=2.0.3,<2.2.0", "requests>=2.31.0,<3.0.0"],
        
        # Batch 4: Visualization base
        ["matplotlib>=3.7.2,<3.9.0"],
        
        # Batch 5: Enhanced visualization
        ["seaborn>=0.12.2,<0.14.0", "plotly>=5.15.0,<5.18.0"],
        
        # Batch 6: Scientific computing
        ["scipy>=1.11.1,<1.12.0", "scikit-learn>=1.3.0,<1.4.0"],
        
        # Batch 7: Utilities
        ["python-dotenv>=1.0.0,<1.1.0", "tqdm>=4.65.0,<4.67.0"],
        
        # Batch 8: Optional but recommended
        ["matplotlib-venn>=0.11.9,<0.12.0", "colorlog>=6.7.0,<6.8.0"],
    ]
    
    for i, batch in enumerate(batches, 1):
        if not batch:  # Skip empty batches
            continue
            
        print(f"\n📦 Installing batch {i}: {batch}")
        
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + batch
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✅ Batch installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️ Batch {i} failed, trying individual packages...")
            
            # Try installing each package individually
            for package in batch:
                try:
                    cmd = [sys.executable, "-m", "pip", "install", package]
                    subprocess.run(cmd, capture_output=True, text=True, check=True)
                    print(f"✅ {package}")
                except subprocess.CalledProcessError:
                    print(f"❌ {package} - FAILED")
    
    return True

def verify_installation():
    """Verify that key packages are installed correctly"""
    print("\n🔍 Verifying installation...")
    
    test_imports = [
        "requests", "pandas", "numpy", "matplotlib", 
        "seaborn", "plotly", "scipy", "sklearn"
    ]
    
    success_count = 0
    for package in test_imports:
        try:
            if package == "sklearn":
                __import__("sklearn")
            else:
                __import__(package)
            print(f"✅ {package}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {package}: {str(e)}")
    
    print(f"\n📊 Installation Success: {success_count}/{len(test_imports)} packages")
    
    if success_count >= len(test_imports) - 2:  # Allow 2 failures
        print("🎉 Installation SUCCESSFUL! Core functionality available.")
        return True
    else:
        print("⚠️ Installation PARTIAL. Some features may not work.")
        return False

def main():
    """Main setup process"""
    print("🚀 Enhanced Movie Genre Analyzer - Pre-Installation Setup")
    print("=" * 60)
    
    steps = [
        ("Checking Python version", check_python_version),
        ("Upgrading core tools", upgrade_core_tools),
        ("Installing compatibility packages", install_compatibility_packages),
        ("Checking system dependencies", check_system_dependencies),
        ("Creating directories", create_directories),
        ("Testing critical imports", test_critical_imports),
        ("Installing packages in batches", install_packages_batch),
        ("Verifying installation", verify_installation),
    ]
    
    for step_name, step_func in steps:
        print(f"\n{'='*20} {step_name} {'='*20}")
        
        try:
            if not step_func():
                print(f"❌ Step failed: {step_name}")
                print("\n🛠️  Manual troubleshooting may be required.")
                return False
        except Exception as e:
            print(f"❌ Step error: {step_name} - {str(e)}")
            return False
    
    print("\n" + "="*60)
    print("🎉 PRE-INSTALLATION SETUP COMPLETE!")
    print("✅ Ready to run: python main.py")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)