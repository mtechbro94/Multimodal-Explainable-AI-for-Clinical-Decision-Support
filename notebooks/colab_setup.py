import os
import sys

def setup_colab(project_path='/content/drive/MyDrive/DeeniTalks'):
    """Sets up Colab environment."""
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("Google Drive mounted.")
        
        # Add project path to sys.path
        src_path = os.path.join(project_path, 'research', 'src')
        if src_path not in sys.path:
            sys.path.append(src_path)
            print(f"Added {src_path} to sys.path")
            
        return True
    except ImportError:
        print("Not running in Google Colab.")
        return False

def check_gpu():
    import torch
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("GPU not available. Using CPU.")
        return False

if __name__ == "__main__":
    setup_colab()
    check_gpu()
