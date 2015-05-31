import itertools
import subprocess

import os
import os.path

import urllib.request
import urllib.parse

import xml.etree.ElementTree as ET
from progressbar import ProgressBar


UID_CHUNK_SIZE = 5000
SRR_CHUNK_SIZE = 200
FASTQ_DUMP_CMD = '/home/kpj/Downloads/sratoolkit.2.4.5-2-ubuntu64/bin/fastq-dump'

def curl_xml(url):
    content = urllib.request.urlopen(url).read().decode('utf-8')
    return ET.fromstring(content)

def retrieve_uids(term, fname):
    """ Save UIDs associated with given query under specified filename
    """
    def get_url(start_index, result_length):
        _url_query = urllib.parse.urlencode({
            'db': 'sra',
            'term': term,
            'retstart': start_index,
            'retmax': result_length
        })
        return 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?%s' % _url_query

    # find overall number of entries
    entry_num = int(curl_xml(get_url(0, 0)).find('Count').text)
    pbar = ProgressBar(maxval=entry_num)

    # retrieve all UIDs
    pbar.start()
    with open(fname, 'w') as fd:
        i = 0
        while True:
            _url = get_url(i, UID_CHUNK_SIZE)
            _xml = curl_xml(_url)

            query_size = int(_xml.find('RetMax').text)
            if query_size == 0: break

            for uid_ele in _xml.find('IdList').findall('Id'):
                uid = uid_ele.text
                fd.write(uid + '\n')

            i += query_size
            pbar.update(i)
    pbar.finish()

def retrieve_srr(uid_fname, srr_fname):
    """ Associate given UIDs with SRR
    """
    with open(uid_fname, 'r') as uid_fd, open(srr_fname, 'w') as srr_fd:
        uid_lines = uid_fd.read().split('\n')
        pbar = ProgressBar(maxval=len(uid_lines))

        pbar.start()
        current_query = []
        for i, line in enumerate(uid_lines):
            if len(line) == 0: continue

            if len(current_query) < SRR_CHUNK_SIZE:
                current_query.append(line)
            else:
                queries = 'OR'.join(current_query)

                _url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=%s' % queries
                _xml = curl_xml(_url)

                if not _xml is None:
                    for exp_pkg in _xml.findall('EXPERIMENT_PACKAGE'):
                        try:
                            identifiers = exp_pkg.find('RUN_SET').find('RUN').find('IDENTIFIERS')
                            srr = identifiers.find('PRIMARY_ID').text

                            srr_fd.write(srr + '\n')
                        except AttributeError:
                            pass

                current_query = []

            pbar.update(i)
        pbar.finish()

def retrieve_fastq(fname, dst_dir):
    """ Load all specified SRR files into given directory
    """
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)

    with open(fname, 'r') as fd:
        srr_lines = fd.read().split('\n')

        pbar = ProgressBar(maxval=len(srr_lines))
        pbar.start()
        for i, srr in enumerate(srr_lines):
            proc = subprocess.Popen([
                FASTQ_DUMP_CMD, srr,
                '-O', dst_dir
            ])
            stdout, stderr = proc.communicate()
            pbar.update(i)
        pbar.finish()

def main():
    retrieve_uids('escherichia coli[organism]', 'uid_list.txt')
    retrieve_srr('uid_list.txt', 'srr_list.txt')
    retrieve_fastq('srr_list.txt', 'fastq')

if __name__ == '__main__':
    main()
