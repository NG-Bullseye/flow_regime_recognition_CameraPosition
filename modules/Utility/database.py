import os

import psycopg2

class Sim_data_db:
    def __init__(self, db_name):
        self.conn = psycopg2.connect(dbname=db_name, user="postgres", password="1234", host="localhost")
        self.c = self.conn.cursor()

        # Create table
        self.c.execute('''
            CREATE TABLE IF NOT EXISTS data 
            (image_path TEXT UNIQUE, gradcam_mean_path TEXT UNIQUE, model_path TEXT UNIQUE, yaw REAL UNIQUE, acc REAL, rec_scalar REAL)
        ''')

    def put(self, **kwargs):
        # Insert or update a row depending on whether it already exists or not
        columns = ', '.join(kwargs.keys())
        placeholders = ', '.join(['%s'] * len(kwargs))
        updates = ', '.join(f"{col} = EXCLUDED.{col}" for col in kwargs.keys())

        upsert_sql = f"""
            INSERT INTO data ({columns})
            VALUES ({placeholders})
            ON CONFLICT (yaw)
            DO UPDATE SET {updates}
        """

        self.c.execute(upsert_sql, list(kwargs.values()))
        self.conn.commit()


    def get(self, value_column, key_column, key_value):
        """Retrieve a value from the database based on a key.

        This method fetches a single value from the database table 'data' by using the given 'key_value'
        to look up the 'value_column' and 'key_column'.

        Parameters:
            value_column (str): The name of the column that contains the value to be retrieved.
            key_column (str): The name of the column to use as the lookup key.
            key_value (str or float): The value of the 'key_column' to search for.

                If 'key_value' is a float, it will be treated as a string, as keys in the database
                are assumed to be of string type. Otherwise, 'key_value' is used directly.

        Returns:
            str or None: The retrieved value from the 'value_column' corresponding to the given 'key_value'.
                If the 'key_value' is not found in the database, the function returns None.

        Raises:
            None

        Example:
            Assuming the 'data' table in the database contains the following data:

            | yaw        | image_path   |
            |------------|--------------|
            |    "1"     |    "Value1"  |
            |    "2"     |    "Value2"  |
            |    "3"     |    "Value3"  |

            You can use this method as follows:

            db = YourDatabase()  # Assuming YourDatabase is the class that defines this 'get' method
            value = db.get("image_path", "yaw", "1")
            print(value)  # Output: "Value1"

            value = db.get("value_column", "key_column", "C")
            print(value)  # Output: None
        """
        # Treat key_value as a string if it's a float
        if isinstance(key_value, float):
            sql = f'SELECT {value_column} FROM data WHERE CAST({key_column} AS TEXT) = %s'
            key_value = str(key_value)
        else:
            sql = f'SELECT {value_column} FROM data WHERE {key_column} = %s'
        self.c.execute(sql, (key_value,))
        result = self.c.fetchone()
        return result[0] if result else None

    def delete(self, key_variable, value):
        # Construct the SQL query
        sql = f'DELETE FROM data WHERE {key_variable}=%s'

        # Execute the query
        self.c.execute(sql, (value,))

        # Save (commit) the changes
        self.conn.commit()

    def get_asList(self, column_name):
        # Construct the SQL query
        sql = f'SELECT {column_name} FROM data'

        # Execute the query
        self.c.execute(sql)

        # Fetch all results
        results = self.c.fetchall()

        # Return a list containing the first element of each result tuple
        return [result[0] for result in results]

    def reset(self):
        # Delete all rows from the table
        self.c.execute('DELETE FROM data')

        # Save (commit) the changes
        self.conn.commit()

    def max_distinct_count(self):
        # List of columns
        columns = ['image_path', 'gradcam_mean_path', 'model_path', 'yaw', 'acc', 'rec_scalar']

        max_count = 0
        for column in columns:
            # Query to count distinct values in each column
            self.c.execute(f'SELECT COUNT(DISTINCT {column}) FROM data')

            # Fetch the result
            result = self.c.fetchone()
            if result is not None:
                count = result[0]
                if count > max_count:
                    max_count = count

        return max_count

    def close(self):
        self.c.close()
        self.conn.close()
