import pandas as pd
import numpy as np
import csv
from read_data import get_time_dict,Enrollment, Truth, Date, Object, Log



if __name__ == '__main__':
    time_dict = get_time_dict()
    enrollment = Enrollment('data/train/enrollment_train.csv')
    truth = Truth('data/train/truth_train.csv')
    date = Date('data/date.csv')
    objects = Object('data/object.csv')
    log = Log('data/train/log_train.csv')


    print(enrollment.enrollment_info.get("4")) #get the particular item from the class
    print(truth.truth_info.get("5")) #get the particular item from the class
