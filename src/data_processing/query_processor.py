import pandas as pd
from sqlalchemy import create_engine
import urllib

connection_string = "DRIVER={SQL Server Native Client 11.0};SERVER=LAPTOP-C3LFVOFI;DATABASE=student_life;UID=student_sense;PWD=abhinav123"


# Create a connection with SQL server to get data.
def exec_sql_query(query, param=None):
    """
    @param query: The query to be executed.
    @param param: If executing a stored procedure, pass the list of parameters in params.
    @return: DataFrame of the result set from the query.
    """
    # Create Database Connection.

    params = urllib.parse.quote_plus(connection_string)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params)
    connection = engine.raw_connection()

    try:
        cursor = connection.cursor()
        if (param):
            cursor.execute(query, param)
        else:
            cursor.execute(query)

        results = cursor.fetchall()
        columns = [column[0] for column in cursor.description]
        df = pd.DataFrame.from_records(results, columns=columns)
        cursor.close()
        connection.commit()
    finally:
        connection.close()

    del engine
    return df
