import csv
import os

def readFromCSV(filePath):
    file = open(filePath, "rt")
    reader = csv.reader(file)
    images_names = []
    labels = []
    for row in reader:
        values = " ".join(row)
        images_names.append(values.split(";")[0])
        labels.append(values.split(";")[7])
    file.close()
    images_names.remove("Filename")
    labels.remove("ClassId")
    return images_names, labels
images_test, labels_test = readFromCSV('/home/albatros/Workspace/MachineLearning/german_traffic_signs/GTSRB/Final_Test/Images/GT-final_test.csv')

print (len(images_test))
print (len(labels_test))
print (len(os.listdir('/home/albatros/Workspace/MachineLearning/german_traffic_signs/GTSRB/Final_Test/Images')))
