# test_get_data.py
import psycopg2
from get_data import get_data

# Create connection (you can reuse this elsewhere)
conn = psycopg2.connect(
    dbname="testing",
    user="postgres",
    password="Pranav2004",
    host="localhost",
    port="5522"
)

# Example 1: Get result as list of dicts
data = get_data(conn, "SELECT * FROM test_users WHERE age > ?", [26], "normal")
print(data)

# Example 2: Get result as DataFrame
df = get_data(conn, "SELECT id, name FROM test_users WHERE name = ?", ["Bob"], "dataframe")
print(df)
