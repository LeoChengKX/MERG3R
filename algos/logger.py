import psutil
import torch
import os


class MemLogger:
    """
    Log the time and memory usage for a process.
    """
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.logs = ""
    
    def log_start(self):
        torch.cuda.reset_peak_memory_stats()

    def log(self, tag=""):
        ram = self.process.memory_info().rss / 1024 ** 2
        gpu = torch.cuda.memory_allocated() / 1024 ** 2
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2

        self.logs += f"[MEMLOG] {tag} | RAM: {ram:.2f}MB | GPU: {gpu:.2f} MB | Peak GPU: {peak:.2f}MB\n"

    def get_logs(self):
        return self.logs
    