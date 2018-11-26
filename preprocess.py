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

#    
#    # enrollment_ids_list
#    # enrollment.enrollment_ids[i]
#    courseid_date_To= {}
#    i = 0
#    for i in log.enrollment_info,date.course_info:
#        if log.enrollment_info.get(str(enrollment.enrollment_ids[i])[3]== date.course_info[i][0]:
#            courseid_date_To =  [date.course_info[i][0], date.course_info[i][2]]
#    i = 0
#    for i in log.enrollment_info,date.course_info:
#        print(log.enrollment_info.get(str(enrollment.enrollment_ids[i]))[4][3])
##    print(log.enrollment_info.get(str(enrollment.enrollment_ids[1])

#    i = 0
##    enrollment.enrollment_ids[i] #enrollment id list
#    if enrollment.enrollment_ids[i] == log.log_info[str(enrollment.enrollment_ids[i])]:
#        haha = [log.log_info.get(i)[0], log.log_info.get(str(enrollment.enrollment_ids[i]))[1]]
#    print(haha)


#    array = {}
#    for i in objects.object_info,enrollment.enrollment_ids:
#        if (log.log_info.get(i)[0] == str(enrollment.enrollment_ids[i])):
#            array[i] = [enrollment.enrollment_info.get(str(enrollment.enrollment_ids[i])), log.log_info.get(i)[1]]
#            print(array[i])

#    handling/testing print of "Date('data/date.csv')"
#    print a tuple when the get content of column[0] = "bWdj2GDclj5ofokWjzoa5jAwMkxCykd6"




#<-------these are ok ------>
#    append course_date_To and objects
    i = 0
    j = 0
    k = 0
    for i in date.course_info:
        for j in objects.object_info: 
#            for k in enrollment.enrollment_info:
            if date.course_info.get(i)[0] == objects.object_info.get(j)[0]:          
#                    if enrollment.enrollment_info(k)[2] == objects.object_info.get(j)[0]:
                print(objects.object_info.get(i),date.course_info.get(i)[2])

    # for k in log.enrollment_ids:
        # print(k)
        # print(log.enrollment_info.get(k)[1])

    print(log.enrollment_info.get("4")[0]) # 4
    print(log.enrollment_info.get("4")[1][1][:10]) # 2014-06-15


# ------------- meaningful data, grouped by each user ------------- #

    # count no. of different events by each user
    events = {}
    for key in enrollment.enrollment_ids: # Log_Data.events.keys():
        countproblem = 0
        countvideo = 0 
        countaccess = 0
        countwiki = 0 
        countdiscussion = 0
        countnavigate = 0
        countpage_close = 0
        for event in log.events.get(key): # Log_Data.events.get(key):
            if event == "problem":
                countproblem += 1
            elif event == "video":
                countvideo += 1
            elif event == "access":
                countaccess += 1
            elif event == "wiki":
                countwiki += 1
            elif event == "discussion":
                countdiscussion += 1
            elif event == "navigate":
                countnavigate += 1
            elif event == "page_close":
                countpage_close += 1
            else:
                print("Error")
        if key not in events:
            events[key] = [countproblem]
        events[key].append(countvideo)
        events[key].append(countaccess)
        events[key].append(countwiki)
        events[key].append(countdiscussion)
        events[key].append(countnavigate)
        events[key].append(countpage_close)            
    #print(events)

    # count no. of access per day by each user
    period = {}
    for key in enrollment.enrollment_ids:
        period[key] =len(log.dates.get(key))
    #print(period)

    # find the latest access by each user
    latest_access = {}
    for key in enrollment.enrollment_ids:
        latest_access[key] = max(log.dates.get(key))
    #print(latest_access)

# ------------- www ------------- #
# >>> a = np.array([[1, 2], [3, 4]])
# >>> b = np.array([[5, 6]])
# >>> np.concatenate((a, b), axis=0)

# arr = np.array([])

# Gather all info tgt#
    features = []
    for EnrollID in enrollment.enrollment_ids:
        #features: Enrollment ID, UserID, CourseID, # of problem, # of video, # of access, # of wiki, # of discussion, # of navigate, # of page_close, # of dates enrolled to this course, # latest_access
        featuresarr = np.array([EnrollID, enrollment.enrollment_info[EnrollID][0], enrollment.enrollment_info[EnrollID][1], *events[EnrollID], period[EnrollID], latest_access[EnrollID]],dtype=object)
        features.append(featuresarr)
