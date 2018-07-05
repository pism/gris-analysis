#!/usr/bin/env python

import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import gdal
from glob import glob
import multiprocessing as mp
import tarfile
import wget
from os.path import join, realpath, dirname, exists, splitext, isfile
from os import mkdir
script_path = dirname(realpath(__file__))

def dem_files(members):
    for tarinfo in members:
        if splitext(tarinfo.name)[1] == ".tif":
            yield tarinfo
            
            
def extract_tar(file, extracted_dir):
    '''
    Extract all files from archive
    '''
    print("Extracting DEM from file {}".format(file))
    tar = tarfile.open(file)
    tar.extractall(path=extracted_dir, members=dem_files(tar))
    tar.close()

def wget_file(tasks, downloaded_files, process_name, tar_dir, dry):
    '''
    Download file using wget
    '''
    while True:
        url = tasks.get()
        if not isinstance(url, str):
            print('[%s] evaluation routine quits' % process_name)
            
            # Indicate finished
            downloaded_files.put(0)
            break
        else:           
            print('Processing file {}'.format(url))
            if dry==False:
                out_file = join(tar_dir, wget.filename_from_url(url))
                if not isfile(out_file):
                    out_file = wget.download(url, out=tar_dir)
            else:
                out_file = wget.filename_from_url(url)

            downloaded_files.put(out_file)
    return


def get_fileurls(file):
    '''
    Get URLs of files
    '''
    with open(file) as f:
        reader = csv.DictReader(f)
        data = [row for row in reader]
    fileurls = [x['fileurl'] for x in data]
    return fileurls


def download_files(fileurls, num_processes, tar_dir='.', dry=False):

    '''
    Download requested files
    '''
    
    manager = mp.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    downloaded_files = mp.Queue()

    pool = mp.Pool(processes=num_processes)  
    processes = []
    
    for i in range(num_processes):

        # Set process name
        process_name = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = mp.Process(target=wget_file, args=(tasks, downloaded_files, process_name, tar_dir, dry))

        # Add new process to the list of processes
        processes.append(new_process)

        # Start the process
        new_process.start()

    # Fill task queue
    task_list = fileurls
    for single_task in task_list:  
        tasks.put(single_task)

    for i in range(num_processes):  
        tasks.put(0)

    # Read calculation results
    num_finished_processes = 0
    all_downloaded_files = []
    k = 0
    while True:  
        # Read result
        new_result = downloaded_files.get()
        # Have a look at the results
        if new_result == 0:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                break
        else:
            # Output result
            all_downloaded_files.append(new_result)
            k += 1
    return all_downloaded_files


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.description = "Generate tables for the paper"
    parser.add_argument("--num_procs", dest="num_processes",
                        help="Number of simultaneous downloads. Default=4", type=int,
                        default=4)
    parser.add_argument("--tar_dir", dest="tar_dir",
                        help="Directory to store the tar files. Default='tar_files'",
                        default='tar_files')
    parser.add_argument("--csv_file", dest="csv_file",
                        help="CSV file that containes tiles information. Default='gris-tiles.csv'",
                        default=join(script_path, 'gris-tiles.csv'))
    options = parser.parse_args()
    csv_file = options.csv_file
    num_processes = options.num_processes
    tar_dir = options.tar_dir
    dry_download = False
    if not exists(tar_dir):
        mkdir(tar_dir)
            
    fileurls = get_fileurls(csv_file)
    all_downloaded_files = download_files(fileurls, num_processes, tar_dir=tar_dir, dry=dry_download)
    extracted_dir = 'extracted_files'
    # for m_file in all_downloaded_files:
    #     extract_tar(m_file, extracted_dir=extracted_dir)

    resolution = 5
    destName = 'gris-dem-{}m.vrt'.format(resolution)
    srcDSOrSrcDSTab = glob(join(extracted_dir, '*dem.tif'))
    options = gdal.BuildVRTOptions(resolution='user', xRes=resolution, yRes=resolution, resampleAlg=gdal.GRA_Average)
    print("Building VRT {}".format(destName))
    # gdal.BuildVRT(destName, srcDSOrSrcDSTab, options=options)
    gdal.BuildVRT(destName, srcDSOrSrcDSTab)
    img = gdal.Open(destName, 0)  # 0 = read-only, 1 = read-write.
    print("Building pyramids for {}".format(destName))
    gdal.SetConfigOption('BIGTIFF', 'YES')
    gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'YES')
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'DEFLATE')
    img.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64] )
    del img
    
