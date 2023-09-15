from logging.config import dictConfig
from logging import Logger
import logging
import yaml

def get_logger():
  with open('conf/logging.yaml', 'r') as f:
      config = yaml.safe_load(f.read())
      dictConfig(config)

  logger = logging.getLogger(__name__)
  
  return logger

if __name__ == '__main__':
  logger = get_logger()
  logger.info('logging service working as expected.')