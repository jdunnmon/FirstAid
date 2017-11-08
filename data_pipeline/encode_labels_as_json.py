import json
import csv
name_to_label = {}
with open('test/mass_case_description_test_set.csv', 'rU') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        image_name = row['image file path']
        image_name = image_name.split("/",1)[0]
        image_name = image_name.split("_",1)[1]
        image_name = image_name + ".tif"
        if row['pathology'] == 'MALIGNANT':
            name_to_label[image_name] = 1
        else:
            name_to_label[image_name] = 0
        #print image_name, name_to_label[image_name]

with open('train/mass_case_description_train_set.csv', 'rU') as csvfile2:
    reader2 = csv.DictReader(csvfile2)
    for row in reader2:
        image_name = row['image file path']
        image_name = image_name.split("/",1)[0]
        image_name = image_name.split("_",1)[1]
        image_name = image_name + ".tif"
        if image_name in name_to_label:
            print "train has same named file as test: ", image_name
        if row['pathology'] == 'MALIGNANT':
            name_to_label[image_name] = 1
        else:
            name_to_label[image_name] = 0
with open('mass_to_label.json', 'w') as outfile:
    json.dump(name_to_label, outfile)