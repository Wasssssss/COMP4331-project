import pandas as pd
import numpy as np
import csv

def get_time_dict():
    rng = pd.date_range('2013-10-27', '2014-08-01')
    print('number of dates:', len(rng))
    time_dict = pd.Series(np.arange(len(rng)), index=rng)
    print(time_dict['2013/10/30'])
    return time_dict




class Enrollment():
    def __init__(self, filename):
        fin = open(filename)
        fin.next()

        self.enrollment_ids = []
        self.enrollment_info = {}
        self.user_info = {}
        self.user_enrollment_id = {}
        self.course_info = {}

        for line in fin:
            enrollment_id, username, course_id = line.strip().split(',')
            if enrollment_id == 'enrollment_id':        # ignore the first row
                continue
            self.enrollment_ids.append(enrollment_id)
            self.enrollment_info[enrollment_id] = [username, course_id]

            if username not in self.user_info:
                self.user_info[username] = [course_id]
                self.user_enrollment_id[username] = [enrollment_id]
            else:
                self.user_info[username].append(course_id)
                self.user_enrollment_id[username].append(enrollment_id)

            if course_id not in self.course_info:
                self.course_info[course_id] = [username]
            else:
                self.course_info[course_id].append(username)
        print("load enrollment info over!")
        print("number of courses:", len(self.course_info))
        print("number of enrollments:", len(self.enrollment_info))
        print("information of enrollment_id=1:", self.enrollment_info.get("1"))

time_dict = get_time_dict()
enrollment = Enrollment('data/train/enrollment_train.csv')

class Truth():          
    def _init_(self, filename):
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            self.truth_ids= []
            self.truth_info = {} 
            
            for line in reader:
                self.truth_ids.append(line[0])
                self.truth_info[line[0]] = [line[0],line[1]]

if __name__ == '__main__':
    time_dict = get_time_dict()
    enrollment = Enrollment('train/enrollment_train.csv')
    truth = Truth('train/truth_train.csv')
    print(enrollment.enrollment_info.get("1")) #get the particular item from the class
    print(truth.truth_info.get("5")) #get the particular item from the class
    
    
