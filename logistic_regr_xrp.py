from database_writer import CsvReader

csv = CsvReader()

conn = csv.connect_db(csv.db_path)
cur = conn.cursor()

data = csv.read_db(cur, 'xrp')

for i in range(0, 10):
    print data[i]
