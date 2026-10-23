import mysql.connector

try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="foodscanner"
    )

    print("Database connected successfully!")

    connection.close()

except mysql.connector.Error as error:

    print("Database connection failed!")
    print(error)