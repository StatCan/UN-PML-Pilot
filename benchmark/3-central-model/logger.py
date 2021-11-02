import logging

#Log accuracy and execution times separately
accuracy_logger = logging.getLogger('Accuracy Log')
time_logger = logging.getLogger('Time Log')
accuracy_logger.setLevel(logging.INFO)
time_logger.setLevel(logging.INFO)

acc_fh = logging.FileHandler('accuracy.log')
acc_fh.setLevel(logging.INFO)

time_fh = logging.FileHandler('time.log')
time_fh.setLevel(logging.INFO)

formatter = logging.Formatter('%(message)s')
acc_fh.setFormatter(formatter)
time_fh.setFormatter(formatter)

accuracy_logger.addHandler(acc_fh)
time_logger.addHandler(time_fh)