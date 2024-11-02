import requests
import csv
from datetime import date
from bs4 import BeautifulSoup
import pathlib
from datetime import datetime

DATA_DIR = (pathlib.Path(__file__).parent.parent.parent / "data").resolve()

header = None

csv_data = []

scrape_year = 1996
while scrape_year <= datetime.now().year:
    print(f"scraping year: {scrape_year}", end="\r")

    if scrape_year < 1999:
        url = (
            f"https://www.rochesterastronomy.org/snimages/snredshift{scrape_year}.html"
        )
    elif scrape_year == 1999:
        url = f"https://www.rochesterastronomy.org/snimages/sn{scrape_year}/snredshift.html"
    else:
        url = f"https://www.rochesterastronomy.org/sn{scrape_year}/snredshift.html"

    print(url)
    response = requests.get(url)

    if not response.status_code == 200:
        raise Exception("Something, somewhere, went wrong.")

    soup = BeautifulSoup(response.content, features="html.parser")

    sn_table = soup.find_all("table")[1]
    rows = iter(sn_table.find_all("tr"))

    header_row = next(rows)
    if header is None:
        header = [col.text.strip() for col in header_row.find_all("th")]
        csv_data.append(header)

    for row in rows:
        cols = [col.text.strip() for col in row.find_all("td")]
        csv_data.append(cols)

    scrape_year += 1

print()

with open(DATA_DIR / "rochester-data.csv", "w") as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(csv_data)
