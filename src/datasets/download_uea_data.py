import requests, zipfile, io
import os 
import argparse


def download_and_extract_dataset(url, data_dir, database_name, pwd=None):
    """ Download dataset from url, only if not previously downloaded.
    """
    #Concatenate data directory with database name:
    database_dir = os.path.join(data_dir, database_name)
    if not os.path.exists(database_dir): # check if database directory exists, if not proceed
        if not os.path.exists(data_dir):
            print('Creating data diretory..')
            os.makedirs(data_dir)
        print('Data directory available..')
        print('Downloading the data..')
        r = requests.get(url)
        print(f'request ok: {r.ok}')
        print('Unzipping data files')
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(data_dir, pwd=pwd)
    else: 
        print('data base directory already exists')
    return 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', 
                        default='data',
                        help='Directory to data sets')
    parser.add_argument('--database', 
                        default='UEA',
                        help='Which database to download: [UEA, UCR]')
    parser.add_argument('--used_format', 
                        default='ts',
                        help='Which format to use, sktime ts format, or textfile tsv: [ts, tsv]')
    args = parser.parse_args()
    database_name = args.database
    data_dir = args.data_dir    
    used_format = args.used_format
    
    #Check and Set conditions (ulr, password, formats)
    pwd = None #in most cases, no pwd is required for unzipping..
    if database_name == 'UEA':
        if used_format == 'tsv':
            raise ValueError('Currently, only ts format implemented for UEA database..')
        else: 
            url = 'https://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip'        
    elif database_name == 'UCR':
        if used_format == 'tsv':
            print('Using tsv format, need to set password for unzipping..')
            pwd = b'someone'
            url = 'https://www.cs.ucr.edu/~eamonn/time_series_data_2018/UCRArchive_2018.zip' 
        else:
            url = 'https://www.timeseriesclassification.com/Downloads/Archives/Univariate2018_ts.zip'
    
    print(f'Trying to download {database_name} database in {used_format} format...')
    #Perform data download and extraction:
    download_and_extract_dataset(url, data_dir, database_name, pwd=pwd) 

if __name__ == '__main__':
    main()         

