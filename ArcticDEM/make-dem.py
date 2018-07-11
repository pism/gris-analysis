#!/usr/bin/env python
# (c) 2018, Andy Aschwanden


import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import csv
import gdal
from glob import glob
import multiprocessing as mp
import tarfile
import wget
from os.path import basename, join, realpath, dirname, exists, split, splitext, isfile
from os import mkdir
script_path = dirname(realpath(__file__))


def dem_files(members):
    for tarinfo in members:
        if tarinfo.name.find('reg_dem.tif') != -1:
            yield tarinfo
            
            
def extract_tar(file, dem_dir=None):
    '''
    Extract DEM files from archive
    '''
    print("Extracting DEM from file {}".format(file))
    tar = tarfile.open(file)
    tar.extractall(path=dem_dir, members=dem_files(tar))
    tar.close()


def process_file(tasks, dem_files, process_name, tar_dir, dem_dir):
    '''
    Download file using wget, extract dem from tar archive, and calculate stats
    '''
    while True:
        url = tasks.get()
        if not isinstance(url, str):
            print('[%s] evaluation routine quits' % process_name)
            
            # Indicate finished
            dem_files.put(0)
            break
        else:           
            print('Processing file {}'.format(url))
            out_file = join(tar_dir, wget.filename_from_url(url))
            if not isfile(out_file):
                out_file = wget.download(url, out=tar_dir)
            #extract_tar(out_file, dem_dir=dem_dir)
            m_file = basename(out_file)
            root, ext = splitext(m_file)
            if ext == '.gz':
                root, ext = splitext(root)
            m_file =  join(dem_dir, root + '_reg_dem.tif')
            #calc_stats_and_overviews(m_file)
            dem_files.put(m_file)
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


def collect_files_mp(fileurls, num_processes, tar_dir='.', dem_dir='.'):

    '''
    Collect and process requested files
    '''
    
    manager = mp.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    dem_files = mp.Queue()

    pool = mp.Pool(processes=num_processes)  
    processes = []
    
    for i in range(num_processes):

        # Set process name
        process_name = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = mp.Process(target=process_file, args=(tasks, dem_files, process_name, tar_dir, dem_dir))

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
    all_dem_files = []
    k = 0
    while True:  
        # Read result
        new_result = dem_files.get()
        # Have a look at the results
        if new_result == 0:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                break
        else:
            # Output result
            all_dem_files.append(new_result)
            k += 1
    return all_dem_files


def calc_stats_and_overviews(destName):
    '''
    Calculate statistics and build overviews for tile
    '''
    
    ds = gdal.OpenEx(destName, 0)  # 0 = read-only, 1 = read-write.
    print('Building overviews and calculating stats for {}'.format(destName))
    ds.GetRasterBand(1).GetStatistics(0, 1)
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'PACKBITS')
    ds.BuildOverviews("NEAREST", [2, 4, 8, 16, 32, 64])
    del ds


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.description = "Generate tables for the paper"
    parser.add_argument("-l",  "--levels", dest="levels",
                        help="Comma seperated list of preview levels. Default: 16,32,64,128,256,512,1024", 
                        default='16,32,64,128,256,512,1024')
    parser.add_argument("--num_procs", dest="num_processes",
                        help="Number of simultaneous downloads. Default=4", type=int,
                        default=4)
    parser.add_argument("-o", "--outname_prefix", dest="outname_prefix",
                        help="Prefix of the output Virtual Raster file {outname}-{resolution}m.vrt. Default='gris-dem'",
                        default='gris-dem')
    parser.add_argument("-r", "--resolution", dest="resolution", type=int,
                        help="Resolution in meters. Default='5'",
                        default=5)
    parser.add_argument("--tar_dir", dest="tar_dir",
                        help="Directory to store the tar files. Default='tar_files'",
                        default='tar_files')
    parser.add_argument("--dem_dir", dest="dem_dir",
                        help="Directory to store the dem files. Default='dem_files'",
                        default='dem_files')
    parser.add_argument("--csv_file", dest="csv_file",
                        help="CSV file that containes tiles information. Default='gris-tiles.csv'",
                        default=join(script_path, 'gris-tiles.csv'))
    options = parser.parse_args()
    csv_file = options.csv_file
    pyramid_levels = [int(x) for x in options.levels.split(',')]
    num_processes = options.num_processes
    outname_prefix = options.outname_prefix
    resolution = options.resolution
    tar_dir = options.tar_dir
    dem_dir = options.dem_dir
    
    if not exists(tar_dir):
        mkdir(tar_dir)
    if not exists(dem_dir):
        mkdir(dem_dir)
            
    fileurls = get_fileurls(csv_file)
    all_dem_files = collect_files_mp(fileurls, num_processes, tar_dir=tar_dir, dem_dir=dem_dir)

    destName = '{prefix}-{resolution}m.vrt'.format(prefix=outname_prefix, resolution=resolution)
    print("Building VRT {}".format(destName))
    gdal.BuildVRT(destName, all_dem_files)
    ds = gdal.OpenEx(destName, 0)  # 0 = read-only, 1 = read-write.
    print("Building pyramids for {}".format(destName))
    gdal.SetConfigOption('BIGTIFF', 'YES')
    gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'YES')
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'PACKBITS')
    ds.GetRasterBand(1).GetStatistics(0, 1)
    ds.BuildOverviews("NEAREST", pyramid_levels)
    del ds
    
