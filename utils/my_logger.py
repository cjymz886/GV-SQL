import logging
import sys

def setup_logging(log_level=logging.INFO,log_file='app.log'):
    """日志配置函数"""

    #获取跟 logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)


    #创建一个Formatter
    log_format = logging.Formatter(
        '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s'
    )

    #1.配置控制台输出(StreamHandler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_format)
    console_handler.setLevel(logging.INFO)

    #2.配置文件输出(FileHandler)
    file_handler = logging.FileHandler(log_file,mode='a',encoding='utf-8')
    file_handler.setFormatter(log_format)
    file_handler.setLevel(logging.DEBUG)

    #将handlers 添加到根logger
    #防止重复添加handler
    if not root_logger.handlers:
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

    

