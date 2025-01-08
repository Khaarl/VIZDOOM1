import gc
import torch
import psutil
import logging
from contextlib import contextmanager

class ResourceManager:
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_available = torch.cuda.is_available()
        self.initial_gpu_memory = self.get_gpu_memory() if self.gpu_available else 0
        
    def get_gpu_memory(self):
        """Get current GPU memory usage in MB"""
        return torch.cuda.memory_allocated() / 1024 / 1024
        
    def get_ram_usage(self):
        """Get current RAM usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
        
    @contextmanager
    def monitor_resources(self):
        """Context manager for monitoring resource usage"""
        try:
            initial_ram = self.get_ram_usage()
            initial_gpu = self.get_gpu_memory() if self.gpu_available else 0
            yield
        finally:
            final_ram = self.get_ram_usage()
            final_gpu = self.get_gpu_memory() if self.gpu_available else 0
            
            self.logger.info(f"RAM Usage (MB): {final_ram - initial_ram:.2f}")
            if self.gpu_available:
                self.logger.info(f"GPU Memory Usage (MB): {final_gpu - initial_gpu:.2f}")

    def cleanup(self):
        """Explicit cleanup of resources"""
        try:
            gc.collect()
            if self.gpu_available:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            return True
        except Exception as e:
            self.logger.error(f"Error during resource cleanup: {e}")
            return False

class ResourceContext:
    def __init__(self, resource_manager, video_writer=None, tensorboard_writer=None):
        self.resource_manager = resource_manager
        self.video_writer = video_writer
        self.tensorboard_writer = tensorboard_writer
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.video_writer:
                try:
                    self.video_writer.close()
                except Exception as e:
                    self.resource_manager.logger.error(f"Error closing video writer: {e}")
                    
            if self.tensorboard_writer:
                try:
                    self.tensorboard_writer.close()
                except Exception as e:
                    self.resource_manager.logger.error(f"Error closing tensorboard writer: {e}")
                    
            self.resource_manager.cleanup()
            
        except Exception as e:
            self.resource_manager.logger.error(f"Error in resource cleanup: {e}")
            return False
        return True
