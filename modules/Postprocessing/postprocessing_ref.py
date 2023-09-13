import glob, re, tensorflow as tf
import random
import yaml
from keras_preprocessing.image.utils import array_to_img
from matplotlib import cm
import os
import cv2
import numpy as np
from modules.Utility import gradcamplusplus, extract_yaw_from_pathstring, PrettySafeLoader
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import psycopg2 #psycopg2_binary

class Sim_data_db:
    def __init__(self, db_name):
        self.conn = psycopg2.connect(dbname=db_name, user="postgres", password="1234", host="localhost")
        self.c = self.conn.cursor()
        self.c.execute('CREATE TABLE IF NOT EXISTS data (image_path TEXT UNIQUE, gradcam_mean_path TEXT UNIQUE, model_path TEXT UNIQUE, yaw REAL UNIQUE, acc REAL, rec_scalar REAL)')

    def add(self, **kwargs):
        conflict_target = ""
        if "yaw" in kwargs:
            conflict_target = "(yaw)"
            self.c.execute("SELECT * FROM data WHERE yaw=%s", (kwargs["yaw"],))
        elif "model_path" in kwargs:
            conflict_target = "(model_path)"
            self.c.execute("SELECT * FROM data WHERE model_path=%s", (kwargs["model_path"],))

        result = self.c.fetchone()
        if result:
            # If a row with this key exists, convert it to a dictionary
            col_names = [desc[0] for desc in self.c.description]
            existing_row = dict(zip(col_names, result))

            # Update the existing dictionary with new values
            existing_row.update(kwargs)
            kwargs = existing_row

        columns = ', '.join(kwargs.keys())
        placeholders = ', '.join(['%s'] * len(kwargs))
        updates = ', '.join(f"{col} = EXCLUDED.{col}" for col in kwargs.keys())

        upsert_sql = f"""
            INSERT INTO data ({columns})
            VALUES ({placeholders})
            ON CONFLICT {conflict_target}
            DO UPDATE SET {updates}
        """

        self.c.execute(upsert_sql, list(kwargs.values()))
        self.conn.commit()

    def get(self, value_column, key_column, key_value):
        key_column = f"CAST({key_column} AS TEXT)" if isinstance(key_value, float) else key_column
        self.c.execute(f"SELECT {value_column} FROM data WHERE {key_column} = %s",
                       (str(key_value) if isinstance(key_value, float) else key_value,))
        result = self.c.fetchone()
        return result[0] if result else None

    def delete(self, key_variable, value):
        self.c.execute(f'DELETE FROM data WHERE {key_variable}=%s', (value,))
        self.conn.commit()

    def get_asList(self, column_name):
        self.c.execute(f'SELECT {column_name} FROM data')
        return [result[0] for result in self.c.fetchall()]

    def reset(self):
        self.c.execute('DROP TABLE IF EXISTS data')
        self.conn.commit()
        self.c.execute(
            'CREATE TABLE IF NOT EXISTS data (image_path TEXT UNIQUE, gradcam_mean_path TEXT UNIQUE, model_path TEXT UNIQUE, yaw REAL UNIQUE, acc REAL, rec_scalar REAL)')
        self.conn.commit()

    def max_distinct_count(self):
        return max((lambda column: self.c.execute(f'SELECT COUNT(DISTINCT {column}) FROM data') or self.c.fetchone()[0] if self.c.fetchone() else 0)(column) for column in ['image_path', 'gradcam_mean_path', 'model_path', 'yaw', 'acc', 'rec_scalar'])

    def close(self):
        self.c.close()
        self.conn.close()

class Postprocessor:
    def __init__(self):
        self.rp = "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg"
        self.root_path_to_preprocessed_images_yaw = f"{self.rp}/preprocessed_sorded_by_yaw"
        with open('../params.yaml', 'r') as stream:
            self.params = yaml.load(stream, Loader=PrettySafeLoader.PrettySafeLoader)
        self.root_path_to_bulktraining_output = self.params["root_path_to_bulktraining_output"]
        self.db = Sim_data_db('postgres')
        self.db.__init__(db_name='postgres')
    def run(self):
        #self.populate_db_with_yaw_model_path()
        #self.populate_db_with_yaw_images_path()
        #self.populate_db_with_yaw_acc()
        self.bulk_compute_gradcamPP_mean_for_each_yaw()
    def populate_db_with_yaw_model_path(self):
        for ytf in os.listdir(self.root_path_to_bulktraining_output):
            ytfp = os.path.join(self.root_path_to_bulktraining_output, ytf)
            for tomp in os.listdir(ytfp):
                tompp = os.path.join(ytfp, tomp)
                if not os.path.isdir(tompp) and ".h5" in tomp.lower():
                    self.db.add(yaw=str(extract_yaw_from_pathstring.extract_value(tompp)), model_path=tompp)

    def populate_db_with_yaw_images_path(self):
        for pi in os.listdir(self.root_path_to_preprocessed_images_yaw):
            p = os.path.join(self.root_path_to_preprocessed_images_yaw, pi)
            self.db.add(yaw=str(extract_yaw_from_pathstring.extract_value(p)), image_path=p)

    def populate_db_with_yaw_acc(self):
        p = re.compile(r"cnn_acc(?P<accuracy>\d+\.\d+)_auc\d+\.\d+\.h5")
        for y in self.db.get_asList("yaw"):
            mp = self.db.get("model_path", "yaw", y)
            if mp is not None:
                m = p.search(os.path.basename(mp))
                if m: self.db.add(yaw=y, acc=float(m.group("accuracy")) / 100)

    def bulk_compute_gradcamPP_mean_for_each_yaw(self):
        for mp in self.db.get_asList("model_path"):
            self.compute_gradcamPP_mean_and_populate_gradcam_path(mp, self.db.get("image_path","model_path",mp))

    def compute_gradcamPP_mean_and_populate_gradcam_path(self, model_path, yaw_dir):
        m = tf.keras.models.load_model(model_path); sum = np.zeros((28, 28)); count = 0
        for pip in self.getImgPaths(yaw_dir):
            img = cv2.imread(pip); gray_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=-1)
            sum += cv2.resize(gradcamplusplus.get_heatmap(gray_img, m, 'conv2d_81'), (28, 28)); count += 1
           # print(f"sum: {sum} count: {count}")
        try:
            mean = sum / count
            cmap = cm.get_cmap("cool")(np.arange(256))[:, :3]
            cmap[0:0] = 0
            f"mean: {mean}"
            heatmap_disp_Img = array_to_img(cmap[(mean * 255).astype(int)])
            td = os.path.dirname(model_path)
            if not os.path.exists(td): os.makedirs(td)
            gradcam_path=os.path.join(td, f"gradcamPP_mean_Img_yaw{self.db.get('yaw', 'model_path', model_path)}.png")
            heatmap_disp_Img.save(gradcam_path)
            self.db.add(gradcam_mean_path=gradcam_path, model_path=model_path)
        except:
            print(f'ERROR110:  iamge path: ## model_Path {model_path}')
    def compute_rec_scalar_from_gradcam_mean_for_each_yaw(self):
        yaw_list=self.db.get_asList("yaw")
        for y in yaw_list:
            image_path = self.db.get("gradcam_mean_path", "yaw", y)
            if image_path is None:
                image_path=self.db.get("gradcam_mean_path", "yaw", round(y))
            if image_path is not None and "45.0.png" in image_path:
                img =cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                rec_scalar_x=self.compute_rec_scalar(img)
                self.db.add(yaw=y, rec_scalar=rec_scalar_x)

    def getImgPaths(self, dir_path):
        exts = ['png', 'jpg', 'jpeg', 'bmp']
        paths = [f for ext in exts for f in glob.glob(f"{dir_path}/*.{ext}")]
        random.shuffle(paths)
        return paths

    def test_gcpp_mean(self):
        self.compute_gradcamPP_mean_and_populate_gradcam_path(
            "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/Bulktraining_Outputs/Einzeltraining/yaw_6.4285712242126465/cnn_acc86.0_auc0.9.h5",
            "/home/lwecke/Datensätze/Datensatz_v1_50p_3reg/preprocessed_sorded_by_yaw/yaw_6.4285712242126465"
        )

    def compute_rec_scalar(self, img):
        UPSCALE_FACTOR = 100  # antialising supersampling
        upscaled_img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR),
                                  interpolation=cv2.INTER_NEAREST)

        grayscale_img = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2GRAY)

        min_val, max_val = np.min(grayscale_img), np.max(grayscale_img)
        normalized_grayscale_img = (grayscale_img - min_val) / (max_val - min_val)

        height, width = normalized_grayscale_img.shape
        total_weight = np.sum(normalized_grayscale_img)
        if total_weight == 0:
            print(f'ERROR147: total_weight == 0')
            return None
        avg_x = 0
        avg_y = 0
        for y in range(height):
            for x in range(width):
                weight = normalized_grayscale_img[y, x]
                avg_x += (width - x - 1) * weight
                avg_y += (height - y - 1) * weight  # Flipping the y-axis

        avg_x /= total_weight
        avg_y /= total_weight
        center_x, center_y = (width - 1) / 2, (height - 1) / 2

        # Calculate center_of_gravity_x based on center_x
        center_of_gravity_x = (avg_x - center_x)

        return center_of_gravity_x

def main():
    postprocessor = Postprocessor()
    #postprocessor.db.reset()
    #postprocessor.run()
    #postprocessor.populate_db_with_yaw_model_path()
    #postprocessor.populate_db_with_yaw_images_path()
    #postprocessor.populate_db_with_yaw_acc()
    #postprocessor.bulk_compute_gradcamPP_mean_for_each_yaw()
    postprocessor.compute_rec_scalar_from_gradcam_mean_for_each_yaw()
    #postprocessor.test_gcpp_mean()

if __name__ == '__main__':
    main()
