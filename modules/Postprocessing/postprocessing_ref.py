import glob, re, tensorflow as tf
import random
from keras_preprocessing.image.utils import array_to_img
from matplotlib import cm
import os
from modules.Utility import gradcamplusplus, extract_yaw_from_pathstring, PrettySafeLoader
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import psycopg2 #psycopg2_binary
from psycopg2 import sql
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import numpy as np
import cv2

class Sim_data_db:
    def __init__(self, db_name):
        try:
            self.conn = psycopg2.connect(dbname=db_name, user="postgres", password="1234", host="localhost")
        except psycopg2.OperationalError as e:
            # If database doesn't exist, then create one and reconnect
            print(f"Database {db_name} does not exist. Creating...")
            conn_temp = psycopg2.connect(dbname="postgres", user="postgres", password="1234", host="localhost")
            conn_temp.autocommit = True
            cursor_temp = conn_temp.cursor()

            # Use psycopg2.sql to safely create the database
            cursor_temp.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name)))

            cursor_temp.close()
            conn_temp.close()

            # Now try connecting again
            self.conn = psycopg2.connect(dbname=db_name, user="postgres", password="1234", host="localhost")

        self.c = self.conn.cursor()
        self.c.execute(
            'CREATE TABLE IF NOT EXISTS data (image_path TEXT UNIQUE, gradcam_mean_path TEXT UNIQUE, model_path TEXT UNIQUE, yaw REAL UNIQUE, acc REAL, rec_scalar REAL)')
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

    def getYawAccMapping(self):
        yaw_list = self.get_asList("yaw")  # Assuming 'yaw' is the column name for yaw values
        acc_mapping = {}

        for yaw in yaw_list:
            acc_value = self.get("acc", "yaw", yaw)  # Assuming 'acc' is the column name for acceleration values
            if acc_value is not None:
                acc_mapping[yaw] = acc_value

        return acc_mapping
    def getYawRecMapping(self):
        yaw_list = self.get_asList("yaw")  # Assuming 'yaw' is the column name for yaw values
        acc_mapping = {}

        for yaw in yaw_list:
            acc_value = self.get("rec_scalar", "yaw", yaw)  # Assuming 'acc' is the column name for acceleration values
            if acc_value is not None:
                acc_mapping[yaw] = acc_value

        return acc_mapping
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
    def __init__(self,datapath):
        self.rp = "/home/lwecke/Datensätze/Test_datensatz_output"
        self.root_path_to_preprocessed_images_yaw = f"{self.rp}/preprocessed_sorded_by_yaw" #sort by yaw first

        self.root_path_to_bulktraining_output = datapath
        self.filename=os.path.basename(datapath)
        print(f"Database Name and Filename: {self.filename}")
        self.db = Sim_data_db(os.path.basename(datapath))
        self.datapath=datapath
    def run(self):
        self.populate_db_with_yaw_model_path()
        self.populate_db_with_yaw_images_path()
        self.populate_db_with_yaw_acc()
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
        p = re.compile(r"cnn_acc(?P<accuracy>\d+\.\d+)(_auc\d+\.\d+)?\.h5")

        for y in self.db.get_asList("yaw"):
            # Format the yaw to match the database storage format
            formatted_yaw = str(int(y)) if y.is_integer() else str(y)

            mp = self.db.get("model_path", "yaw", formatted_yaw)
            if mp is not None:
                m = p.search(os.path.basename(mp))
                if m: self.db.add(yaw=formatted_yaw, acc=float(m.group("accuracy")) / 100)
        return self.db.getYawAccMapping()

    def bulk_compute_gradcamPP_mean_for_each_yaw(self):
        for mp in self.db.get_asList("model_path"):
            if mp is None:
                continue
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
            if image_path is not None:
                img =cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                rec_scalar_x=self.compute_rec_scalar(img)
                self.db.add(yaw=y, rec_scalar=rec_scalar_x)
        return self.db.getYawRecMapping()

    def plot_rec_scalar_from_gradcam_mean_path(self):
        imagePath_gravity_dict = {}
        yaw_list = self.db.get_asList("yaw")

        for idx, y in enumerate(yaw_list):
            image_path = self.db.get("gradcam_mean_path", "yaw", y)
            img = mpimg.imread(image_path)

            if img is None:
                print("Image not found at path:", image_path)
                imagePath_gravity_dict[image_path] = (-1, -1)
                continue

            center_of_gravity_x, center_of_gravity_y = self.compute_rec_scalar(img)
            UPSCALE_FACTOR = 100 # antialiasing supersampling
            subplot_scale= 100

            img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR),
                                      interpolation=cv2.INTER_NEAREST)
            print(img.shape[0])
            print(img.shape[1])
            print(f"x {center_of_gravity_x} y{center_of_gravity_y}")
            fig, ax2 = plt.subplots(1, 1, figsize=(10,10))
            vector_length = 4
            center_of_gravity_x*=vector_length
            center_of_gravity_y*=vector_length
            ax2.imshow(img)
            Rectangle_scale=0.1
            scale=10



            center_x = round(img.shape[1] / 2)
            center_y = round(img.shape[0] / 2)

            ax2.axis('off')

            # Draw a red square at the center of the image
            width=10*scale
            rect_image = patches.Rectangle((center_x-width/2, center_y-width/2), width, width, linewidth=3, edgecolor='blue',
                                           facecolor='none')
            ax2.add_patch(rect_image)

            # Compute the vector from the center of the image to the center of gravity

            vector_x =  center_of_gravity_x
            vector_y =  center_of_gravity_y
             #y7574002675
            vector_y = vector_y
            vector_x= vector_x
            rect_gravity = patches.Rectangle(((center_of_gravity_x+center_x)-width/2 , (center_of_gravity_y +center_y)-width/2), 100, 100,   linewidth=3, edgecolor='r', facecolor='none')
            ax2.add_patch(rect_gravity)
            print("vector",vector_y)
            print("vector",vector_x)
            # Draw a blue line from the center of the image to the center of gravity
            ax2.plot([center_x, center_x + vector_x], [center_y, center_y + vector_y], color='green',linewidth=2)

            #ax2.set_title(f"x: {center_of_gravity_x}, y: {center_of_gravity_y}")

            save_path = f"saved_image_{idx}.png"
            plt.savefig(save_path)
            plt.close(fig)
            print(save_path)





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
        UPSCALE_FACTOR = 100  # antialiasing supersampling
        upscaled_img = cv2.resize(img, (img.shape[1] * UPSCALE_FACTOR, img.shape[0] * UPSCALE_FACTOR),
                                  interpolation=cv2.INTER_NEAREST)

        grayscale_img = cv2.cvtColor(upscaled_img, cv2.COLOR_RGB2GRAY)
        min_val, max_val = np.min(grayscale_img), np.max(grayscale_img)
        normalized_grayscale_img = (grayscale_img - min_val) / (max_val - min_val)

        height, width = normalized_grayscale_img.shape

        x_coords = np.arange(width)
        y_coords = np.arange(height)

        x_grid, y_grid = np.meshgrid(x_coords, y_coords)

        flipped_x_grid = width - x_grid - 1
        flipped_y_grid = height - y_grid - 1

        total_weight = np.sum(normalized_grayscale_img)

        avg_x = np.sum(flipped_x_grid * normalized_grayscale_img) / total_weight
        avg_y = np.sum(flipped_y_grid * normalized_grayscale_img) / total_weight

        center_x, center_y = (width - 1) / 2, (height - 1) / 2

        center_of_gravity_x = (avg_x - center_x)
        center_of_gravity_y = (avg_y - center_y)

        return center_of_gravity_x, center_of_gravity_y

def yawacc(datapath, acc=True, rec=False):
    postprocessor = Postprocessor(datapath)
    postprocessor.populate_db_with_yaw_model_path()
    postprocessor.populate_db_with_yaw_images_path()
    yawacc = postprocessor.populate_db_with_yaw_acc()
    return yawacc

def yawrec(datapath):
    postprocessor = Postprocessor(datapath)
    #postprocessor.bulk_compute_gradcamPP_mean_for_each_yaw()
    yawrec = postprocessor.compute_rec_scalar_from_gradcam_mean_for_each_yaw()
    return yawrec

def plot_recscalar(datapath):
    Postprocessor(datapath).plot_rec_scalar_from_gradcam_mean_path()

if __name__ == '__main__':
   print(f'Results:\n{plot_recscalar("/home/lwecke/Datensätze/Test_datensatz_output/evaluated/n_repeats/Bulktraining_n_repeat3")}')


