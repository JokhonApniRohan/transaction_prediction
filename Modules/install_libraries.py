import subprocess
import sys

# List of required libraries
required_libraries = [
    "numpy",
    "pandas",
    "matplotlib",
    "lightgbm",
    "xgboost",
    "catboost",
    "scikit-learn",  # Required by preprocessing modules
]

def install_libraries():
    """Install all required libraries using pip"""
    print("🔧 Installing required libraries...\n")
    
    for library in required_libraries:
        try:
            print(f"📦 Installing {library}...", end=" ")
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print("✅ Done\n")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {library}\n")
    
    print("✅ All libraries installed successfully!")

if __name__ == "__main__":
    install_libraries()