import urllib.request

from bs4 import BeautifulSoup

url = "https://tac.nist.gov/publications/2017/results.html"


def get_soup(url):
    with urllib.request.urlopen(url) as req:
        contents = req.read()
        return BeautifulSoup(contents, 'html.parser')


def tr_string(soup):
    return ','.join(f'"{td.string}"' if td.string else "" for td in soup.find_all('td'))


soup = get_soup(url).find('table')
for a in soup.find_all('a'):
    if 'entity-discovery-and-linking' in a['href']:
        if 'TEDL' in a['href']:
            print('https://tac.nist.gov/publications/2017/' + a['href'])
