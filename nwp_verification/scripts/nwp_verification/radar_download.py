import os
from datetime import datetime, timedelta
import subprocess
import shutil
from netCDF4 import Dataset

def run_get_ref(work_dir, analysis_time_start, analysis_time_end, 
                fcst_length, reflectivity_thres=-15.0, 
                processor_number=2, interval_hr=1, delete_raw_files=False):

    output = os.path.join(work_dir, 'radar')
    
    start_radar_datetime = analysis_time_start
    end_radar_datetime = analysis_time_end + timedelta(
        seconds=3600*int(fcst_length))
    
    cur_radar_datetime = start_radar_datetime
    
    
    while cur_radar_datetime <= end_radar_datetime:
        rad_cmd = ('get_radar_ref.py -t {} -d {} '
                   '--reflectivity_thres {} --cappi_z_range_min 0 --cappi_z_range_max 500 '
                   '--cappi_z_levels 1 ').format(
                        cur_radar_datetime.strftime('%Y%m%d%H%M'), output,
                        reflectivity_thres)

        rad_cmd += ' --switch_on_qc'
        rad_cmd += ' --use_multiprocessor ' + \
                '--processor_number {}'.format(processor_number)
        rad_cmd += ' --download-data'
        rad_cmd += ' --plot_radar'
        
        if os.path.exists(os.path.join(
                output, 'cappi_reflectivity_{}.nc'.format(
                    cur_radar_datetime.strftime('%Y%m%d%H%M')))):
            cur_radar_datetime = cur_radar_datetime + timedelta(seconds=3600*interval_hr)
            continue

        print '{}: {}'.format(cur_radar_datetime, rad_cmd)
        p1 = subprocess.Popen(rad_cmd, cwd=os.getcwd(), shell=True)
        p1.wait()
        if delete_raw_files:
            shutil.rmtree(
                os.path.join(output, cur_radar_datetime.strftime('%Y%m%d%H%M')))
        cur_radar_datetime = cur_radar_datetime + timedelta(seconds=3600*interval_hr)


def extract_radar_grids(work_dir, valid_time):
    radar_dir = os.path.join(work_dir, 'radar')
    radar_data_filename = os.path.join(
        radar_dir, 'cappi_reflectivity_{}.nc'.format(valid_time.strftime('%Y%m%d%H%M')))
    nc_fid = Dataset(radar_data_filename, 'r')
    lats = nc_fid.variables['point_latitude'][0, :]
    lons = nc_fid.variables['point_longitude'][0, :]
    nc_fid.close()
    return lats, lons
    
    
    
     
        
        
    
        

        
