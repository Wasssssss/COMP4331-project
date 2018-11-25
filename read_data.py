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
        # fin.next()

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
#        print("load enrollment info over!")
#        
#        print("number of courses:", len(self.course_info))
#        print("number of enrollments:", len(self.enrollment_info))
#        print("information of enrollment_id=4:", self.enrollment_info.get("4"))


class Truth():          
    def __init__(self, filename):
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            self.truth_ids= []
            self.truth_info = {} 
            
            for line in reader:
                self.truth_ids.append(line[0])
                self.truth_info[line[0]] = [line[0],line[1]]
                
class Log():
    def __init__(self, filename):
        fin = open(filename)
        #fin.next()

        self.enrollment_ids = []
        self.enrollment_info = {}
        self.dates = {}
        self.events = {}
        #self.source = {}
        #self.objects = {}

        for line in fin:
            enrollment_id, time, source, event, objects = line.strip().split(',')
            if enrollment_id == 'enrollment_id':        # ignore the first row
                continue
            self.enrollment_ids.append(enrollment_id)
            
            if enrollment_id not in self.enrollment_info:
                # self.enrollment_info[enrollment_id] = [time, source, event, objects]
                self.enrollment_info[enrollment_id] = [enrollment_id] # print(log.enrollment_info.get("4")[0]) --> enrollment id
                self.enrollment_info[enrollment_id].append([enrollment_id, time, source, event, objects]) # print(log.enrollment_info.get("4")[1]) --> ['4', 2014-06-15T01:44:10', 'server', 'navigate', 'Oj6eQgzrdqBMlaCtaq1IkY6zruSrb71b']
                self.dates[enrollment_id] = [time[:10]]
                self.events[enrollment_id] = [event]
            else:
                self.enrollment_info[enrollment_id].append([enrollment_id, time, source, event, objects])
                if time[:10] not in self.dates[enrollment_id]:
                    self.dates[enrollment_id].append(time[:10])
                self.events[enrollment_id].append(event)

class Date():
    def __init__(self, filename):
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)
            self.course_ids = []
            self.course_info = {}
            i = 0

                        
            for line in reader:
                if line[0] == 'course_id':        # ignore the first row             
                    continue
                if line[0] != "":     
                    self.course_info[i] = [line[0], line[1], line[2]]
                    i = i + 1

class Object():
    def __init__(self, filename):
        with open(filename, 'r') as fin:
            reader = csv.reader(fin)  
            self.module_ids = []
            self.object_info = {}
            i = 0
            
            for line in reader:
                if line[0] == 'course_id':        # ignore the first row             
                    continue
                if line[0] != "":
                    self.object_info[i] = [line[0], line[1], line[2],line[3], line[4]]
                    i = i + 1

                
if __name__ == '__main__':
    time_dict = get_time_dict()
    enrollment = Enrollment('data/train/enrollment_train.csv')
    truth = Truth('data/train/truth_train.csv')
    date = Date('data/date.csv')
    objects = Object('data/object.csv')
    log = Log('data/train/log_train.csv')


    # print(enrollment.enrollment_info.get("4")) #get the particular item from the class
    # print(truth.truth_info.get("5")) #get the particular item from the class
    
    #handling/testing print of "Date('data/date.csv')"
    #print a tuple when the get content of column[0] = "bWdj2GDclj5ofokWjzoa5jAwMkxCykd6"
    # for i in date.course_info:
        # if date.course_info.get(i)[0] == "bWdj2GDclj5ofokWjzoa5jAwMkxCykd6":
            # print(date.course_info.get(i))
    
    #handling/testing print of "Object('data/object.csv')"
    #print all the tuples when the get content of column[0] = "SpATywNh6bZuzm8s1ceuBUnMUAeoAHHw"     
    # for i in objects.object_info:
        # if (objects.object_info.get(i)[0] == "SpATywNh6bZuzm8s1ceuBUnMUAeoAHHw"):
            # print(objects.object_info.get(i))
    
#    print(log.enrollment_info.get("4"))

#    print(log.enrollment_info.get("4")[0])
#    print("/n")
#    print(log.enrollment_info.get("4")[1])
#    print("/n")
#    print(log.enrollment_info.get("4")[2])
#    print("/n")
#    print(log.enrollment_info.get("4")[3])
#    print("/n")
#    print(log.enrollment_info.get("4")[4][0])
#    print("/n")
#    print(log.enrollment_info.get("4")[5])
#    print("/n")
    
    # print(log.dates.get("1")) #get the particular item from the class
    # print(log.events.get("1")) #get the particular item from the class
