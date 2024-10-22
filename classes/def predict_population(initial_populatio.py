def student_analysis(python_students, data_science_students):
    # Students enrolled in both Python and Data Science classes
    both_courses = python_students.intersection(data_science_students)
    
    # Students enrolled in either Python or Data Science but not both
    either_but_not_both = python_students.symmetric_difference(data_science_students)
    
    return both_courses, either_but_not_both
python_students = {101, 102, 103, 104}
data_science_students = {103, 104, 105, 106}
both, either = student_analysis(python_students, data_science_students)
print(both)
print(either)