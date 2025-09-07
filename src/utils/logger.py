import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(
        self,
        name: str = "transformer",
        log_level: str = "INFO",
        log_file: Optional[str] = None,
        tensorboard_dir: Optional[str] = None
    ):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # TensorBoard writer
        self.tb_writer = None
        if tensorboard_dir:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            tb_path = Path(tensorboard_dir) / f"run_{timestamp}"
            self.tb_writer = SummaryWriter(str(tb_path))
            self.logger.info(f"TensorBoard logging to: {tb_path}")
    
    def info(self, message: str):
        self.logger.info(message)
    
    def debug(self, message: str):
        self.logger.debug(message)
    
    def warning(self, message: str):
        self.logger.warning(message)
    
    def error(self, message: str):
        self.logger.error(message)
    
    def critical(self, message: str):
        self.logger.critical(message)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value to TensorBoard"""
        if self.tb_writer:
            self.tb_writer.add_scalar(tag, value, step)
    
    def log_scalars(self, tag: str, value_dict: dict, step: int):
        """Log multiple scalar values to TensorBoard"""
        if self.tb_writer:
            self.tb_writer.add_scalars(tag, value_dict, step)
    
    def log_histogram(self, tag: str, values, step: int):
        """Log histogram to TensorBoard"""
        if self.tb_writer:
            self.tb_writer.add_histogram(tag, values, step)
    
    def log_model_graph(self, model, input_tensor):
        """Log model graph to TensorBoard"""
        if self.tb_writer:
            self.tb_writer.add_graph(model, input_tensor)
    
    def close(self):
        """Close TensorBoard writer"""
        if self.tb_writer:
            self.tb_writer.close()


def setup_logger(
    name: str = "transformer",
    config: Optional[dict] = None,
    log_file: Optional[str] = None
) -> Logger:
    """
    Setup logger with configuration
    """
    if config is None:
        config = {}
    
    log_level = config.get("log_level", "INFO")
    tensorboard_dir = config.get("tensorboard_dir", None)
    
    # Create logs directory if logging to file
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create tensorboard directory
    if tensorboard_dir:
        Path(tensorboard_dir).mkdir(parents=True, exist_ok=True)
    
    return Logger(
        name=name,
        log_level=log_level,
        log_file=log_file,
        tensorboard_dir=tensorboard_dir
    )