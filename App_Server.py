import matplotlib
from PIL import Image
from flask import Flask, request, render_template

from TestModel import *

matplotlib.use('agg')
app = Flask(__name__, static_url_path='/user_upload', static_folder='user_upload')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global class_tester
    try:
        if request.files.get('filename') is not None:
            original_fname = request.files['filename'].filename
            img_upload = Image.open(request.files['filename'].stream).convert('RGB')
            open_cv_image = np.array(img_upload)
            dir_path = os.path.dirname(os.path.realpath(__file__))
            user_upload_dir = os.path.join(dir_path, USER_UPLOAD)
            t_file_path = os.path.join(user_upload_dir, original_fname)
            if not os.path.exists(user_upload_dir):
                os.makedirs(user_upload_dir)
            cv2.imwrite(t_file_path, open_cv_image[:, :, ::-1])

            if class_tester is None:
                class_tester = ClassTester()
                class_tester.plot = False
                class_tester.force_class = True
                class_tester.save_plot = True

            plot_path = class_tester.predict([t_file_path])
            return render_template('output.html', result=os.path.basename(plot_path))

    except Exception as e:
        print("\n\n\n", str(e), "\n\n\n")
        pass

    return render_template('upload(1).html')


if __name__ == "__main__":
    app.run()
