import csv
import os
import shutil
from os import path

project_path = ""
data_csv_path = os.path.join("Data", "Data_Entry_2017.csv")
bbox_csv_path = os.path.join("Data", "BBox_List_2017.csv")
source = path.join("..", "AI_Whole_Images", "images")
destination = path.join("Data", "Classified")
keep_classes = ['Effusion', 'Mass', 'Nodule', 'Pneumothorax']


def org_images(csv_path):
    with open(csv_path)as csv_file:
        read_csv = csv.DictReader(csv_file)
        label_field = next(x for x in read_csv.fieldnames if 'Finding Label' in x)
        item = 1
        for row in read_csv:
            img = row['Image Index']
            class_name = row[label_field]
            if class_name not in keep_classes:
                continue
            if '|' not in class_name:
                source_path = path.join(source, img)
                if os.path.exists(source_path):
                    dest_class = path.join(destination, class_name)
                    dest_path = path.join(dest_class, img)
                    if not os.path.exists(dest_class):
                        os.makedirs(dest_class)
                    shutil.copyfile(source_path, dest_path)
                    print('Processed item #', item)
                    item += 1


def count_files():
    class_names = set()
    with open(data_csv_path)as csv_file:
        read_csv = csv.DictReader(csv_file)
        for row in read_csv:
            class_name = row['Finding Labels']
            class_names.add(class_name)

    print('Read csv')
    for class_name in class_names:
        class_dir = os.path.join(project_path, class_name)
        if os.path.exists(class_dir):
            list_results = os.listdir(class_dir)
            print(class_name, ':', len(list_results))


org_images(data_csv_path)
