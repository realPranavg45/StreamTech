# test_delete.py

import psycopg2
from delete_data_child import delete_record_safe

# Update these with your actual PostgreSQL credentials
DB_CONFIG = {
    'host': 'localhost',
    'dbname': 'number',
    'user': 'postgres',
    'password': 'Pranav2004',
    'port': 5522
}

def main():
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print("🚫 Attempting to delete from tution1 where id = 1...")
    result = delete_record_safe(
        cur,
        "DELETE FROM tution1 WHERE id = %s",
        [1],  # Replace with the actual id you want to test
        output_format='text'
    )
    print(result)

    conn.commit()
    cur.close()
    conn.close()

if __name__ == "__main__":
    main()
