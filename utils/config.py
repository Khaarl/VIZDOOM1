import os
import logging
from datetime import datetime

# Shared paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")
METRICS_DIR = os.path.join(PROJECT_ROOT, "metrics")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Ensure directories exist
for directory in [LOGS_DIR, METRICS_DIR, MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

def get_run_id():
    """Generate unique run ID using timestamp"""
    return datetime.now().strftime("%m%d%H%M")

def setup_logging(name, run_id=None):
    """Setup logging with consistent format"""
    if run_id is None:
        run_id = get_run_id()
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler
    fh = logging.FileHandler(os.path.join(LOGS_DIR, f"{run_id}_{name}.log"))
    fh.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger, run_id
