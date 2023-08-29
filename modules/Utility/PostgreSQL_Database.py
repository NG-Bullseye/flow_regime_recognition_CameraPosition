import os, glob, cv2, re, imutils, numpy as np, matplotlib.pyplot as plt, matplotlib.patches as patches, matplotlib.image as mpimg, tensorflow as tf
import random

import yaml
from keras_preprocessing.image.utils import load_img, img_to_array, array_to_img
from matplotlib import cm

import modules.Utility.PrettySafeLoader
from modules.Utility import grad_cam, gradcamplusplus, extract_yaw_from_pathstring, PrettySafeLoader
from silence_tensorflow import silence_tensorflow
silence_tensorflow()
import psycopg2

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