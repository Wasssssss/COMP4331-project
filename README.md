# COMP4331-project

# Load data
pandas

# Data pre-processing
1. handle duplicate values
2. handle missing values
3. handle impossible data combinations

# Data pre-processing techniques
1. feature selection: selecting a subset of relevant features for use in model construction
https://en.wikipedia.org/wiki/Feature_selection
2. 

# Requirement
no activity @log_train.csv in past 10 days --> may drop the course

# Data to classifiers
date.csv
- course id
- to: 10 days after the date have no related records in log_train => this is a dropout
* each line contains the timespan of each course in our log data (both train and test data). The timespans of each course for calculating dropouts is 10 days after the last day of that course, i.e., course C is from 2014.4.1 to 2014.4.30 in the given data, a user enrolled the course C will be treated as a dropout if he/she leaves no record from 2014.5.1 to 2014.5.10.

object.csv
- course id
- module id <--> log_train: object
- category <--> log_train: event
- children (separate from one record [pre-processing])

enrollment_train.csv
- enrollment id
- username
- course id <--> object: course id

log_train.csv
- enrollment id
- time: only keep recent 10 days activities
- event (problem, video, discussion) <--> object: category
- object <--> object: module id, children
* an user access an event but he/she may not be enrolled into that particular course
* need to further check if he/she enrolled in to that course (enrollment_train)

true_train.csv
- enrollment id
- ground truth

# Classifiers
- KNN
- SVM
- random forest
