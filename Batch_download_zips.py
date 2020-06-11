# Download the 56 zip files in Images_png in batches
import multiprocessing
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from urllib import request

# URLs for the zip files
links = [
    'https://nihcc.box.com/shared/static/vfk49d74nhbxq3nqjg0900w5nvkorp5c.gz',
    'https://nihcc.box.com/shared/static/i28rlmbvmfjbl8p2n3ril0pptcmcu9d1.gz',
    'https://nihcc.box.com/shared/static/f1t00wrtdk94satdfb9olcolqx20z2jp.gz',
    'https://nihcc.box.com/shared/static/0aowwzs5lhjrceb3qp67ahp0rd1l1etg.gz',
    'https://nihcc.box.com/shared/static/v5e3goj22zr6h8tzualxfsqlqaygfbsn.gz',
    'https://nihcc.box.com/shared/static/asi7ikud9jwnkrnkj99jnpfkjdes7l6l.gz',
    'https://nihcc.box.com/shared/static/jn1b4mw4n6lnh74ovmcjb8y48h8xj07n.gz',
    'https://nihcc.box.com/shared/static/tvpxmn7qyrgl0w8wfh9kqfjskv6nmm1j.gz',
    'https://nihcc.box.com/shared/static/upyy3ml7qdumlgk2rfcvlb9k6gvqq2pj.gz',
    'https://nihcc.box.com/shared/static/l6nilvfa9cg3s28tqv1qc1olm3gnz54p.gz',
    'https://nihcc.box.com/shared/static/hhq8fkdgvcari67vfhs7ppg2w6ni4jze.gz',
    'https://nihcc.box.com/shared/static/ioqwiy20ihqwyr8pf4c24eazhh281pbu.gz'
]


def download_async(item_no, link):
    zip_file = 'images_{}.tar.gz'.format(item_no)
    print('downloading item', item_no, ':', zip_file)
    try:
        request.urlretrieve(link, zip_file)
        print('downloaded item', item_no, ':', zip_file)
        # print('unzipping', zip_file)
        # shutil.unpack_archive(zip_file)
        # print('unzipped', zip_file)
        # os.remove(zip_file)
    except Exception as e:
        print(e)


executor = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
for (item_no, link) in enumerate(links, start=1):
    executor.submit(download_async, item_no, link)
