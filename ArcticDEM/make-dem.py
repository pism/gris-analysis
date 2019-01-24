#!/usr/bin/env python
# (c) 2018-19 Andy Aschwanden


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
        if tarinfo.name.find('_dem.tif') != -1:
            yield tarinfo


def extract_tar(file, dem_dir=None):
    '''
    Extract DEM files from archive
    '''
    print("Extracting DEM from file {}".format(file))
    tar = tarfile.open(file)
    tar.extractall(path=dem_dir, members=dem_files(tar))
    tar.close()


def process_file(tasks, dem_files, dem_hs_files, process_name, options_dict, zf, multiDirectional, tile_pyramid_levels, tar_dir, dem_dir):
    '''
    Download file using wget, extract dem from tar archive, and calculate stats
    '''
    while True:
        url = tasks.get()
        if not isinstance(url, str):
            print('[%s] evaluation routine quits' % process_name)
            # Indicate finished
            dem_files.put(0)
            dem_hs_files.put(0)
            break
        else:
            out_file = join(tar_dir, wget.filename_from_url(url))
            if options_dict['download']:
                if not exists(out_file):
                    print('Processing file {}'.format(url))
                    out_file = wget.download(url, out=tar_dir)
            if options_dict['extract']:
                extract_tar(out_file, dem_dir=dem_dir)
            m_file = basename(out_file)
            root, ext = splitext(m_file)
            if ext == '.gz':
                root, ext = splitext(root)
            m_file = join(dem_dir, root + '_dem.tif')
            m_ovr_file = join(m_file, ".ovr")
            m_hs_file = join(dem_dir, root + '_dem_hs.tif')
            m_hs_ovr_file = join(m_hs_file, ".ovr")
            if options_dict['build_tile_overviews']:
                if not exists(m_ovr_file):
                    calc_stats_and_overviews(m_file, tile_pyramid_levels)
            if options_dict['build_tile_hillshade']:
                if not exists(m_hs_file):
                    create_hillshade(m_file, m_hs_file, zf, multiDirectional)
            if options_dict['build_tile_hillshade_overviews']:
                if not exists(m_hs_ovr_file):
                    calc_stats_and_overviews(m_hs_file, tile_pyramid_levels)
            dem_files.put(m_file)
            dem_hs_files.put(m_hs_file)
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


def collect_files_mp(fileurls, num_processes, zf, multiDirectional, tile_pyramid_levels, options_dict, tar_dir='.', dem_dir='.'):
    '''
    Collect and process requested files
    '''

    manager = mp.Manager()

    # Define a list (queue) for tasks and computation results
    tasks = manager.Queue()
    dem_files = mp.Queue()
    dem_hs_files = mp.Queue()

    pool = mp.Pool(processes=num_processes)
    processes = []

    for i in range(num_processes):

        # Set process name
        process_name = 'P%i' % i

        # Create the process, and connect it to the worker function
        new_process = mp.Process(target=process_file, args=(tasks, dem_files, dem_hs_files,
                                                            process_name, options_dict, zf, multiDirectional, tile_pyramid_levels, tar_dir, dem_dir))

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
    all_dem_hs_files = []
    k = 0
    while True:
         # Read result
        dem_result = dem_files.get()
        hs_result = dem_hs_files.get()
        # Have a look at the results
        if dem_result == 0:
            # Process has finished
            num_finished_processes += 1

            if num_finished_processes == num_processes:
                break
        else:
            # Output result
            all_dem_files.append(dem_result)
            all_dem_hs_files.append(hs_result)
            k += 1

    return all_dem_files, all_dem_hs_files


def calc_stats_and_overviews(destName, pyramid_levels):
    '''
    Calculate statistics and build overviews for tile
    '''

    ds = gdal.OpenEx(destName, 0)  # 0 = read-only (create external .ovr file)
    print('Building overviews and calculating stats for {}'.format(destName))
    ds.GetRasterBand(1).GetStatistics(0, 1)
    gdal.SetConfigOption('COMPRESS_OVERVIEW', 'PACKBITS')
    ds.BuildOverviews("NEAREST", pyramid_levels)
    del ds


def create_hillshade(srcDS, destName, zf, multiDirectional):
    '''
    Calculate hillshade for tile
    '''

    print('Creating hillshade for {}'.format(destName))
    gdal.DEMProcessingOptions(zFactor=zf, multiDirectional=multiDirectional)
    gdal.DEMProcessing(destName, srcDS, 'hillshade')


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.description = "Create Virtual Raster DEM from ArcticDEM tiles or stripes."
    parser.add_argument("-l",  "--levels", dest="vrt_levels",
                        help="Comma seperated list of overview levels used for the Virtual Raster. Default: 16,32,64,128,256,512,1024",
                        default='16,32,64,128,256,512,1024')
    parser.add_argument("-p",  "--levels_tiles", dest="tile_levels",
                        help="Comma seperated list of overview levels used for the individual tiles. Default: 2,4,8,16,32,64",
                        default='2,4,8,16,32,64')
    parser.add_argument("--num_procs", dest="num_processes",
                        help="Number of simultaneous downloads. Default=4", type=int,
                        default=4)
    parser.add_argument("--zf", dest="zf",
                        help="Number of simultaneous downloads. Default=1",
                        default=1.0)
    parser.add_argument("--multi_directional", dest="multiDirectional", action="store_true",
                        help="Create multi directional hillshade. Default=False",
                        default=False)
    parser.add_argument("--options", dest="process_options",
                        help="Default='all'",
                        default='all',
                        choices=['all',
                                 'download',
                                 'extract',
                                 'build_tile_overviews',
                                 'build_tile_hillshade',
                                 'build_tile_hillshade_overviews',
                                 'build_vrt_raster',
                                 'build_vrt_overviews',
                                 'build_vrt_hillshade',
                                 'build_vrt_hillshade_overviews',
                                 'none'])
    parser.add_argument("-o", "--outname_prefix", dest="outname_prefix",
                        help="Prefix of the output Virtual Raster file {outname}-{resolution}m.vrt. Default='gris-dem'",
                        default='gris-dem')
    parser.add_argument("--tar_dir", dest="tar_dir",
                        help="Directory to store the tar files. Default='tar_files'",
                        default='tar_files')
    parser.add_argument("--dem_dir", dest="dem_dir",
                        help="Directory to store the dem files. Default='dem_files'",
                        default='dem_files')
    parser.add_argument("--csv_file", dest="csv_file",
                        help="CSV file that containes tiles information. Default='gris-tiles.csv'",
                        default=join(script_path, 'test-tiles.csv'))

    options_dict = {'download': False,
                    'extract': False,
                    'build_tile_hillshade': False,
                    'build_tile_hillshade_overviews': False,
                    'build_vrt_hillshade': False,
                    'build_vrt_hillshade_overviews': False,
                    'build_vrt_raster': False,
                    'build_vrt_overviews': False,
                    'build_tile_overviews': False}

    options = parser.parse_args()
    csv_file = options.csv_file
    vrt_pyramid_levels = [int(x) for x in options.vrt_levels.split(',')]
    tile_pyramid_levels = [int(x) for x in options.tile_levels.split(',')]
    num_processes = options.num_processes
    outname_prefix = options.outname_prefix
    tar_dir = options.tar_dir
    dem_dir = options.dem_dir
    zf = options.zf
    multiDirectional = options.multiDirectional
    process_options = options.process_options

    if process_options == 'all':
        for k in options_dict:
            options_dict[k] = True
    elif process_options == 'download':
        options_dict['download'] = True
    elif process_options == 'extract':
        options_dict['extract'] = True
    elif process_options == 'build_tile_overviews':
        options_dict['build_tile_overviews'] = True
    elif process_options == 'build_tile_hillshade_overviews':
        options_dict['build_tile_hillshade_overviews'] = True
    elif process_options == 'build_tile_hillshade':
        options_dict['build_tile_hillshade'] = True
    elif process_options == 'build_vrt_raster':
        options_dict['build_vrt_raster'] = True
    elif process_options == 'build_vrt_overviews':
        options_dict['build_vrt_overviews'] = True
    elif process_options == 'build_vrt_hillshade':
        options_dict['build_vrt_hillshade'] = True
    elif process_options == 'build_vrt_hillshade_overviews':
        options_dict['build_vrt_hillshade_overviews'] = True
    else:
        pass

    if not exists(tar_dir):
        mkdir(tar_dir)
    if not exists(dem_dir):
        mkdir(dem_dir)

    # Extract URLs from a CSV file generated from the SHP Tiles File
    fileurls = get_fileurls(csv_file)
    # Collect and process all DEM files using multiprocessing
    all_dem_files, all_dem_hs_files = collect_files_mp(
        fileurls, num_processes, zf, multiDirectional, tile_pyramid_levels, options_dict, tar_dir=tar_dir, dem_dir=dem_dir)

    destName = '{prefix}.vrt'.format(prefix=outname_prefix)
    if options_dict['build_vrt_raster']:
        print("Building VRT {}".format(destName))
        gdal.BuildVRT(destName, all_dem_files)
    if options_dict['build_vrt_overviews']:
        ds = gdal.OpenEx(destName, 0)  # 0 = read-only (create external .ovr file)
        print("Building pyramids for {}".format(destName))
        gdal.SetConfigOption('BIGTIFF', 'YES')
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'YES')
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'PACKBITS')
        ds.GetRasterBand(1).GetStatistics(0, 1)
        ds.BuildOverviews("NEAREST", vrt_pyramid_levels)
        del ds

    destName = '{prefix}_hs.vrt'.format(prefix=outname_prefix)
    if options_dict['build_vrt_hillshade']:
        print("Building VRT {}".format(destName))
        gdal.BuildVRT(destName, all_dem_hs_files)
    if options_dict['build_vrt_hillshade_overviews']:
        ds = gdal.OpenEx(destName, 0)  # 0 = read-only (create external .ovr file)
        print("Building pyramids for {}".format(destName))
        gdal.SetConfigOption('BIGTIFF', 'YES')
        gdal.SetConfigOption('BIGTIFF_OVERVIEW', 'YES')
        gdal.SetConfigOption('COMPRESS_OVERVIEW', 'PACKBITS')
        ds.GetRasterBand(1).GetStatistics(0, 1)
        ds.BuildOverviews("NEAREST", vrt_pyramid_levels)
        del ds
