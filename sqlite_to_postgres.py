import sqlite3
import pandas as pd
import psycopg2

columns = ['character_id', 'name', 'level', 'exp', 'hp', 'strength', 'intelligence', 'dexterity', 'wisdom']
conn = sqlite3.connect('rpg_db.sqlite3')
curs = conn.cursor()

pg_conn = psycopg2.connect(dbname='mlrgffyq', user='mlrgffyq', password='O6SgFeUVSIUk1q0cvOhV4IM6vsyrSRmS',
                           host='queenie.db.elephantsql.com')

pg_curs = pg_conn.cursor()

character_query = '''SELECT * FROM charactercreator_character'''

results = curs.execute(character_query).fetchall()

sqlite_df = pd.DataFrame(results)

sqlite_df.columns = columns

for row in sqlite_df.iterrows():
    characters_insert_query = """
INSERT INTO character_sqlite ({})
VALUES ({})
""".format(", ".join(columns), "'" + "', '".join([str(x) for x in row[1]]) + "'")
    pg_curs.execute(characters_insert_query)
    pg_conn.commit()

conn.close()
