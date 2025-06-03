import sys
from pathlib import Path
import logging
from datetime import datetime


log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_filepath = log_dir / log_file


log_format = "[%(asctime)s] [%(lineno)d] [%(module)s] - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,
    format=log_format,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("chestCancerClassifierLogger")

if __name__ == '__main__':
    logger.info("Welcome to the Project")