import re, sys, shutil
import os, os.path
import urllib.request

from bs4 import BeautifulSoup


# check commandline arguments
if len(sys.argv) != 2:
    print('Usage: %s <dest dir>' % sys.argv[0])
    sys.exit(1)

# config
_base_dir = sys.argv[1]
_geo_base = 'http://www.ncbi.nlm.nih.gov'

# halp
def read_page(url):
    page = urllib.request.urlopen(url)
    return page.read()

def download_file(url, fname):
    """ Via FTP (cough)
    """
    os.system('wget -O "%s" "%s"' % (fname, url)) # cough cough
    os.system('gunzip "%s"' % fname) # cough cough cough

    newfname = fname[:-3]
    shutil.move(newfname, os.path.join(_base_dir, newfname))

# load webpage
_geo_search = '%s/gds?term=escherichia%%20coli%%5BOrganism%%5D' % _geo_base
content = read_page(_geo_search)

# cook soup
soup = BeautifulSoup(content)

# eat soup
urls = []
for ele in soup.findAll('div', {'class': 'rprt'}):
    title = ele.find('p', {'class': 'title'})
    href = title.find('a')['href']

    urls.append('%s/%s' % (_geo_base, href[1:]))

# now to the main dish
for url in urls:
    soup = BeautifulSoup(read_page(url))

    href = soup.find('a', {'class': 'dataset-soft-full'}).get('href')
    name = re.search(r'soft/(GDS.*)_full\.soft\.gz', href).groups()[0]

    print('Downloading', name)
    download_file(href, '%s.soft.gz' % name)
