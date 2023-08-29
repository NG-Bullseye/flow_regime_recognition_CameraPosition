import psycopg2

class Sim_data_db:
    def __init__(self, db_name):
        self.conn = psycopg2.connect(dbname=db_name, user="postgres", password="1234", host="localhost")
        self.c = self.conn.cursor()
        self.reset()
        self.c.execute('CREATE TABLE IF NOT EXISTS data (image_path TEXT UNIQUE, gradcam_mean_path TEXT UNIQUE, model_path TEXT UNIQUE, yaw REAL UNIQUE, acc REAL, rec_scalar REAL)')

    def put(self, **kwargs):
       pass

    def get(self, value_column, key_column, key_value):
        self.c.execute(f"SELECT {value_column} FROM data WHERE {'CAST({key_column} AS TEXT)' if isinstance(key_value, float) else key_column} = %s", (str(key_value) if isinstance(key_value, float) else key_value,))
        result = self.c.fetchone()
        return result[0] if result else None

    def delete(self, key_variable, value):
        self.c.execute(f'DELETE FROM data WHERE {key_variable}=%s', (value,))
        self.conn.commit()

    def get_asList(self, column_name):
        self.c.execute(f'SELECT {column_name} FROM data')
        return [result[0] for result in self.c.fetchall()]

    def reset(self):
        self.c.execute('DELETE FROM data')
        self.conn.commit()

    def max_distinct_count(self):
        return max((lambda column: self.c.execute(f'SELECT COUNT(DISTINCT {column}) FROM data') or self.c.fetchone()[0] if self.c.fetchone() else 0)(column) for column in ['image_path', 'gradcam_mean_path', 'model_path', 'yaw', 'acc', 'rec_scalar'])

    def close(self):
        self.c.close()
        self.conn.close()
