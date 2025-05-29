import logging
import os

# 封装 logger 的设置
def setup_logger(log_dir, logger_name="train_log", log_level=logging.INFO):
    """
    初始化并返回 logger 和 console_handler.
    
    :param log_dir: 日志文件保存的目录
    :param logger_name: 日志的名称
    :param log_level: 日志的记录级别
    :return: logger 对象和控制台处理器对象
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)

    # 文件处理器 (日志保存到文件)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    file_handler = logging.FileHandler(os.path.join(log_dir, f"{logger_name}.log"))
    file_handler.setLevel(log_level)

    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)

    # 将文件处理器添加到 logger
    logger.addHandler(file_handler)

    # 创建控制台处理器（默认不添加）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    return logger, console_handler

def enable_console_logging(logger, console_handler, enable=True):
    """
    动态启用或禁用控制台日志输出.
    
    :param logger: 需要控制的 logger 对象
    :param console_handler: 控制台处理器
    :param enable: True 表示启用，False 表示禁用
    """
    if enable:
        if console_handler not in logger.handlers:
            logger.addHandler(console_handler)  # 添加控制台处理器
    else:
        if console_handler in logger.handlers:
            logger.removeHandler(console_handler)  # 移除控制台处理器
