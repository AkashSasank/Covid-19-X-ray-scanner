import logging
from datetime import date
import os

class Logger:
    @staticmethod
    def get_logger():
        if not os.path.exists('./logging'):
            os.mkdir('./logging')
        filename = './logging' + '/error_' + date.today().\
            strftime("%Y%m%d") + '.log'
        error_logging = './logging'
        format = "%(asctime)s - %(pathname)s:%(funcName)s:%(lineno)d " \
                 "\n%(message)s"
        if not error_logging:
            logging.disable(logging.CRITICAL)

        logging.basicConfig(filename=filename, format=format)

        return logging
