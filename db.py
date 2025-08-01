from sqlite3 import connect

db = connect('game.db')
cursor = db.cursor()

def init_db(): 
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id                       TEXT,
            generation                   INTEGER,
            best_fitness                 REAL,
            best_solution                TEXT,
            fitness                      TEXT,
            avg_fitness                  REAL,
            std_fitness                  REAL,
            avg_loudness                 REAL,
            std_loudness                 REAL,
            avg_pulse_rate               REAL,
            std_pulse_rate               REAL,
            avg_distance_between_bats    REAL,
            std_distance_between_bats    REAL,
            avg_movement                 REAL,
            std_movement                 REAL
        )
    ''')

    db.commit()


    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp                    TEXT,
            run_id                       TEXT,
            population_size              INTEGER,
            dimensions                   INTEGER,
            max_gen                      INTEGER,
            min_freq                     REAL,
            max_freq                     REAL,
            lower_bound                  REAL,
            upper_bound                  REAL,
            min_A                        REAL,
            max_A                        REAL,
            min_pulse                    REAL,
            max_pulse                    REAL,
            alpha                        REAL,
            gamma                        REAL,
            elitism                      BOOLEAN
        )
    ''')

    db.commit()
    

def add_to_table(table_name, data):
    """Add data to a specific table in the database."""

    columns = ', '.join(data.keys())
    placeholders = ', '.join(['?'] * len(data))
    sql = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    
    # print(f"Executing SQL: {sql} with data: {data}")
    cursor.execute(sql, tuple(data.values()))
    db.commit()

def close_connection():
    """Close the database connection."""
    cursor.close()
    db.close()