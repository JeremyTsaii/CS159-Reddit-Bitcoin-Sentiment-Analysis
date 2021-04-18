import pandas
import json

by_year = dict()
by_month = dict()

csv = pandas.read_csv(
    "bitcoin_reddit_all.csv",
    usecols=['date', 'body'],
    dtype={
        "date": "string",
        'body': 'string'
    },
)

for _, comment in csv.iterrows():
    date = comment['date']
    body = comment['body']

    if not pandas.isna(date) and not pandas.isna(body):
        date = comment['date'].split('-')
    
        year = date[0]
        year_month = '-'.join(date[:2])

        if by_year.get(year, []):
            by_year[year].append(body)
        else:
            by_year[year] = [body]

        if by_month.get(year_month, []):
            by_month[year_month].append(body)
        else:
            by_month[year_month] = [body]

with open("by_year.json", 'w') as file:
    json.dump(by_year, file)

with open("by_month.json", 'w') as file:
    json.dump(by_month, file)