import os
import tkinter
from abc import ABC
from glob import glob
from itertools import chain
from tkinter import Tk
from tkinter.filedialog import askopenfilenames

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras_preprocessing.image import ImageDataGenerator
from matplotlib import patches
from tensorflow.python.keras.models import model_from_json, load_model

IMAGE_SIZE = 128
USER_UPLOAD = "user_upload"
root = Tk()
bbox_tester = None
class_tester = None
pd.options.mode.chained_assignment = None
class_img_paths = os.path.join('.', 'Data', 'Samples', 'Classification', '*.png')
bbox_img_paths = os.path.join('.', 'Data', 'Samples', 'BBOX', '*.png')


class ModelTester(ABC):
    model = None
    df = None
    plot = True
    save_plot = False

    def load_model(self):
        pass

    def read_data(self):
        pass

    def predict(self, file_paths):
        pass


def read_img(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    return img


class BBoxTester(ModelTester):

    def load_model(self):
        # load json and create model
        json_file = open(os.path.join('Models', 'bbox_detection_model.json'), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        # load weights into new model
        self.model.load_weights(os.path.join('Models', 'bbox_detection_weight.h5'))
        print("Loaded model from disk")

    def read_data(self):
        all_xray_df = pd.read_csv(os.path.join('.', 'Data', 'Data_Entry_2017.csv'))
        all_xray_annotations = pd.read_csv(os.path.join('.', 'Data', 'BBox_List_2017.csv'))
        all_xray_annotations.drop(['Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8'], axis=1, inplace=True)
        self.df = pd.merge(all_xray_annotations, all_xray_df, on='Image Index')
        file_paths = [os.path.abspath(x) for x in glob(bbox_img_paths)]
        all_image_paths = {os.path.basename(x): x for x in file_paths}
        self.df['path'] = self.df['Image Index'].map(all_image_paths.get)
        self.df['x0'] = self.df['Bbox [x'] * IMAGE_SIZE / 1024  # self.df['OriginalImage[Width']
        self.df['y0'] = self.df['y'] * IMAGE_SIZE / 1024  # self.df['Height]']
        self.df['w0'] = self.df['w'] * IMAGE_SIZE / 1024  # self.df['OriginalImage[Width']
        self.df['h0'] = self.df['h]'] * IMAGE_SIZE / 1024  # self.df['Height]']
        print("Loaded the data for bbox detection")

    def predict(self, file_paths):
        if self.model is None:
            self.load_model()

        if self.df is None:
            self.read_data()

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            img = read_img(file_path)
            x = np.array(img, np.float32) / 255
            image = cv2.resize(x, (IMAGE_SIZE, IMAGE_SIZE))
            file_df = self.df.loc[self.df['path'] == file_path]
            if file_df.empty:
                print("Path", file_path, "not in the data")
                continue
            finding_label = ','.join(file_df['Finding Label'].values)
            org_x0 = file_df['x0'].values
            org_y0 = file_df['y0'].values
            org_h0 = file_df['h0'].values
            org_w0 = file_df['w0'].values
            ###########################################
            pred_bbox = self.model.predict(x=np.array([image]))[0]
            pred_x0 = pred_bbox[0]
            pred_y0 = pred_bbox[1]
            pred_h0 = pred_bbox[2]
            pred_w0 = pred_bbox[3]
            #####################################################
            # Display the image
            fig, ax = plt.subplots(1)
            fig.canvas.set_window_title('BBOX detection: ' + file_name)
            ax.set_title(finding_label)
            ax.imshow(img)
            # Create a Rectangle patch
            # x1-prex_x0 is the width of the bounding box
            # y1-pred_y0 is the height of the bounding box
            rect_pred = patches.Rectangle((pred_x0, pred_y0), pred_w0, pred_h0, linewidth=2, edgecolor='r',
                                          facecolor='none')
            rect_org = patches.Rectangle((org_x0, org_y0), org_w0, org_h0, linewidth=2, edgecolor='b', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect_pred)
            ax.add_patch(rect_org)
            # Image coordinates - top-left of the image is (0,0)
            ax.plot(pred_x0, pred_y0, 'o', color='b')  # top-left of the bounding box
            ax.plot(pred_x0 + pred_w0, pred_y0 + pred_h0, '*', color='c')  # bottom-right of the bounding-box
        plt.show()


class ClassTester(ModelTester):
    core_idg = None
    all_labels = []

    def load_model(self):
        self.model = load_model(os.path.join('Models', 'classifier_model.h5'))
        self.model.load_weights(os.path.join('Models', 'classifier_model_weight.hdf5'))
        print("Loaded model from disk")

    def read_data(self):
        self.df = pd.read_csv(os.path.join('.', 'Data', 'Data_Entry_2017.csv'))
        # For display purposes - actual vs predicted values. If not found, actual will be 'Unknown'
        file_paths = [os.path.abspath(x) for x in glob(class_img_paths)]
        all_image_paths = {os.path.basename(x): x for x in file_paths}
        self.df['path'] = self.df['Image Index'].map(all_image_paths.get)
        # Preserve only the following classes
        keep_classes = ['Effusion', 'Mass', 'Nodule', 'Pneumothorax']
        self.df = self.df[self.df['Finding Labels'].isin(keep_classes)]
        self.df['Patient Age'] = self.df['Patient Age'].map(lambda x: int(x))
        self.all_labels = np.unique(list(chain(*self.df['Finding Labels'].map(lambda x: x.split('|')).tolist())))
        self.all_labels = [x for x in self.all_labels if len(x) > 0]
        self.all_labels.append('Unknown')
        self.set_finding_flags(self.df)
        self.df['newLabel'] = self.df.apply(lambda x: x['Finding Labels'].split('|'), axis=1)
        print("Loaded the data for classification")

    def set_finding_flags(self, df):
        for c_label in self.all_labels:
            if len(c_label) > 1:  # leave out empty labels
                df[c_label] = df['Finding Labels'].map(lambda finding: 1.0 if c_label in finding else 0)

    def predict(self, file_paths):
        plot_path = os.path.join(USER_UPLOAD, "classification_plot_{}.png".format(str(np.random.randint(100, 999))))
        if self.model is None:
            self.load_model()

        if self.df is None:
            self.read_data()

        valid_df = pd.DataFrame()
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            found_df = self.df.loc[self.df['Image Index'] == file_name]
            if found_df.empty:
                print("Path", file_name, "is not found. Mimicing properties.")
                found_df = self.df.head(1).copy(deep=True)
                found_df['Image Index'] = file_name
                found_df['Finding Labels'] = 'Unknown'
                found_df['newLabel'] = ['Unknown']
                self.set_finding_flags(found_df)
            found_df['path'] = file_path
            valid_df = valid_df.append(found_df)

        if valid_df.empty:
            print("Path(s) ", ','.join(file_paths), " is not found")
            return

        if self.core_idg is None:
            self.core_idg = ImageDataGenerator(samplewise_center=True,
                                               samplewise_std_normalization=True,
                                               horizontal_flip=True,
                                               vertical_flip=False,
                                               height_shift_range=0.05,
                                               width_shift_range=0.1,
                                               rotation_range=5,
                                               shear_range=0.1,
                                               fill_mode='reflect',
                                               zoom_range=0.15)

        test_x, test_y = next(self.core_idg.flow_from_dataframe(dataframe=valid_df,
                                                                directory=None,
                                                                x_col='path',
                                                                y_col='newLabel',
                                                                class_mode='categorical',
                                                                classes=self.all_labels,
                                                                target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                                color_mode='grayscale',
                                                                batch_size=1024))
        pred_y = self.model.predict(test_x, batch_size=32)

        # Image_name Actual_classes Predicted_classes
        sickest_idx = np.argsort(np.sum(test_y, 1) < 1)
        fig, m_axs = plt.subplots(len(sickest_idx), 2, figsize=(12, 4))
        fig.canvas.set_window_title('Class prediction: ' + ','.join([os.path.basename(p) for p in file_paths]))
        fig.tight_layout(pad=3.0)
        axes = m_axs.flatten()
        act_infos, pred_infos = self.get_pred_string(test_y, pred_y)

        for idx in sickest_idx:
            act_info = act_infos[idx]
            pred_info = pred_infos[idx]
            pred_str = ['%s:%2.0f%%' % (n_class, p_score) for n_class, p_score in pred_info]

            if self.plot or self.save_plot:
                ax_idx = idx * 2
                ax_img = axes[ax_idx]
                ax_plt = axes[ax_idx + 1]
                # Image display
                ax_img.imshow(test_x[idx, :, :, 0], cmap='bone', vmin=-1.5, vmax=1.5)
                ax_img.set_title('Actual: ' + ', '.join(act_info) + '\nPredicted: ' + ', '.join(pred_str))
                ax_img.axis('off')

                # Result display
                ax_plt.set_xlim([0, 100])
                ax_plt.barh([val[0] for val in pred_info], [val[1] for val in pred_info], align='center')
                ax_plt.set_title('Predicted results')
            else:
                if act_info[0] == pred_info[0][0]:
                    print(file_paths[idx], 'match', act_info[0])
                else:
                    print(file_paths[idx], 'non match', act_info[0], pred_info[0][0])

        if self.save_plot:
            if not os.path.exists(USER_UPLOAD):
                os.makedirs(USER_UPLOAD)
            fig.savefig(plot_path)
        if self.plot:
            plt.show()
        return plot_path

    def get_pred_string(self, test_y, pred_y):
        act_infos = []
        pred_infos = []
        for idx in range(0, len(test_y)):
            act_info = [n_class for n_class, n_score in zip(self.all_labels, test_y[idx]) if n_score > 0.5]
            pred_info = [[n_class, p_score * 100] for n_class, n_score, p_score in zip(self.all_labels,
                                                                                       test_y[idx], pred_y[idx])
                         if (n_score > 0.5) or (p_score > 0)]
            pred_info.sort(key=lambda x: x[1], reverse=True)
            # Get top 4 classes only
            pred_info = pred_info[:4]
            act_infos.append(act_info)
            pred_infos.append(pred_info)
        return act_infos, pred_infos


def _quit():
    root.quit()  # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent


def bbox_test():
    global bbox_tester
    if bbox_tester is None:
        bbox_tester = BBoxTester()
    filez = askopenfilenames(parent=root, title='Choose file(s) to predict bboxes')
    bbox_tester.predict([os.path.abspath(p) for p in root.tk.splitlist(filez)])


def classify_test():
    global class_tester
    if class_tester is None:
        class_tester = ClassTester()
    filez = askopenfilenames(parent=root, title='Choose file(s) to classify')
    class_tester.plot = True
    class_tester.force_class = True
    class_tester.save_plot = False
    class_tester.predict([os.path.abspath(p) for p in root.tk.splitlist(filez)])


def test_function():
    quit_button = tkinter.Button(master=root, text="Quit", command=_quit)
    bbox = tkinter.Button(master=root, text="BBOX", command=bbox_test)
    classify = tkinter.Button(master=root, text="Classify", command=classify_test)
    classify.pack()
    bbox.pack()
    quit_button.pack()
    root.mainloop()


def main():
    # Test here
    test_function()


if __name__ == "__main__":
    main()
