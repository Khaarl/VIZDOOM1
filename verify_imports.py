import os
import sys

def verify_imports():
    try:
        from src.utils.resource_manager import ResourceManager, ResourceContext
        from src.training.trainer import Trainer, TrainerConfig, TrainingMetrics
        print("✓ Successfully imported all required modules")
        return True
    except Exception as e:
        print(f"Error importing modules: {e}")
        return False

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    if project_root not in sys.path:
        sys.path.append(project_root)
    
    if verify_imports():
        print("All modules are properly configured and accessible")
    else:
        print("Please check your project structure and imports")
