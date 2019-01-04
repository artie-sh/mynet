from database_writer import CsvReader

csv = CsvReader()

conn = csv.connect_db(csv.db_path)
cur = conn.cursor()

data = csv.read_db(cur, 'xrp')

def avg(timerow):
    res = [line[1] for line in timerow]
    return sum(res)/max(len(timerow), 1)

def get_timerow(data, rowname='close'):
    rows = {'open': 1, 'high': 2, 'low': 3, 'close': 4}
    return [(line[0], line[rows[rowname]]) for line in data[:10]]

def get_ma(timerow, periods):
    result = []
    for i in range(len(timerow) - periods):
        result.append(timerow[i + periods])

    return result

    return [(timerow[i + periods][0], avg([timerow[j][1] for j in range(periods)])) for i in range(len(timerow) - periods)]


print get_ma(get_timerow(data, 'close'), 3)

#row = get_timerow(data)

#av = avg(row)
#print av