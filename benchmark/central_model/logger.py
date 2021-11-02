import logging

server_logger = logging.getLogger('Server Log')
server_logger.setLevel(logging.INFO)

fh = logging.FileHandler('server.log')

formatter = logging.Formatter('%(message)s')

fh.setFormatter(formatter)

server_logger.addHandler(fh)