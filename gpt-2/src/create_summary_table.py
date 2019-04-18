"""
Creates a summary table in Amazon RDS instance
"""

import pymysql
import os


# database connection params
HOST = os.environ.get("HOST")
PORT = int(os.environ.get("PORT"))
DBNAME = os.environ.get("DBNAME")
USER = os.environ.get("USER")
PASSWORD = os.environ.get("PASSWORD")


def create_table():
    """
    Creates Summary table. Drops Summary table if it previously existed
    """
    conn = pymysql.connect(HOST, user=USER, port=PORT,
                           passwd=PASSWORD, db=DBNAME)

    # cursor to database
    curs = conn.cursor()

    # drop if table already exists
    drop_query = 'DROP TABLE IF EXISTS Summary'
    curs.execute(drop_query)
    print('Table Dropped!')

    # create Summary table
    create_query = """CREATE TABLE Summary(id int NOT NULL AUTO_INCREMENT, 
                    Date varchar(10), 
                    Commentary varchar(3000),
                    PRIMARY KEY (id))"""
    curs.execute(create_query)
    print('Summary Table Created!')

    # close cursor and commit
    curs.close()
    conn.commit()
    conn.close()

if __name__ == '__main__':
    create_table()