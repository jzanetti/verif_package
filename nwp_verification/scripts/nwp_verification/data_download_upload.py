import pandas
import os
from AMPSAws.utils import load_file_from_s3
try:
    from urllib.parse import urlparse
except ImportError:
    from urlparse import urlparse
from AMPSAws import resources
from uuid import uuid4
from netCDF4 import Dataset
from pyresample import geometry, image

from datetime import datetime, timedelta
from radar_download import extract_radar_grids
from radar_obs import download_and_write_radar
import numpy
import pygrib
import scipy
from wrf import to_np, getvar

REMAP_RADIUS_OF_INFLUENCE = 50000

def append_item_to_dict(target_dict, target_key, item_to_append):
    try:
        target_dict[target_key].append(item_to_append)
    except KeyError:
        target_dict[target_key] = []
        target_dict[target_key].append(item_to_append)
    return target_dict

def download_model(args, model_type=None):
    analysis_time_start = args.analysis_time_start
    analysis_time_end = args.analysis_time_end
    analysis_hour_interval = int(args.analysis_hour_interval)
    fcst_length = int(args.fcst_length)
    
    cur_analysis_time = analysis_time_start
    
    data_info = {}
    # ifs_info = {}
    # gfs_info = {}
    
    while cur_analysis_time <= analysis_time_end:
        if model_type == 'wrf':
            wrf_download_dir = os.path.join(
                args.work_dir, 'model', 'wrf', args.model_name)
            if not os.path.exists(wrf_download_dir):
                os.makedirs(wrf_download_dir)
            success, data_info_out = download_wrf(
                data_info,
                wrf_download_dir, cur_analysis_time, 
                fcst_length, args.wrf_cycle_flag_dir,
                args.model_name, args.status, args.region,
                args.role_name)        
        if model_type == 'gfs':
            gfs_download_dir = os.path.join(
                args.work_dir, 'model', 'gfs', 
                cur_analysis_time.strftime('%Y%m%d%H'))
            if not os.path.exists(gfs_download_dir):
                os.makedirs(gfs_download_dir)
            
            success, data_info_out = download_gfs(
                data_info, gfs_download_dir, 
                cur_analysis_time, fcst_length,
                args.gfs_dir_on_s3)

        if model_type == 'ifs':
            ifs_download_dir = os.path.join(
                args.work_dir, 'model', 'ifs',
                cur_analysis_time.strftime('%Y%m%d%H'))
            if not os.path.exists(ifs_download_dir):
                os.makedirs(ifs_download_dir)
            success, data_info_out = download_ifs(
                data_info, ifs_download_dir, 
                 cur_analysis_time, fcst_length,
                 args.ifs_dir_on_s3)
        
        if not success:
            print '{} data download failed for {}'.format(
                model_type, cur_analysis_time)
        
        cur_analysis_time += timedelta(seconds=3600*analysis_hour_interval)
    
    
    return data_info_out


def download_ifs(ifs_info, ifs_download_dir, 
                 cur_analysis_time, fcst_length,
                 ifs_on_s3):
    ifs_dir_s3 = os.path.join(ifs_on_s3, cur_analysis_time.strftime('%Y/%m/%d/%H'))
    unique_id = str(uuid4())
    ifs_info[unique_id] = {}
    for fcst in range(0, fcst_length+3, 3):
        valid_time = cur_analysis_time + timedelta(seconds=3600*fcst)
        if fcst == 0:
            ifs_end_index = '11'
        else:
            ifs_end_index = '01'
        ifs_filename = 'N1D{analysis_hour_MMDDHH}00{valid_hour_MMDDHH}0{ifs_end_index}.bz2'.format(
            analysis_hour_MMDDHH=cur_analysis_time.strftime('%m%d%H'),
            valid_hour_MMDDHH=valid_time.strftime('%m%d%H'),
            ifs_end_index=ifs_end_index)
        ifs_path_on_s3 = os.path.join(ifs_dir_s3, ifs_filename)
        ifs_path_local = os.path.join(ifs_download_dir, ifs_filename)
        
        try:
            if not os.path.exists(ifs_path_local[:-4]):
                resources.copy(ifs_path_on_s3, ifs_path_local)
                os.system('bzip2 -d {}'.format(ifs_path_local))
            append_item_to_dict(ifs_info[unique_id], 'datetime_analysis', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(ifs_info[unique_id], 'datetime_cold', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(ifs_info[unique_id], 'datetime_lbc', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(ifs_info[unique_id], 'local_path', ifs_path_local[:-4])
            append_item_to_dict(ifs_info[unique_id], 'datetime_valid', valid_time.strftime('%Y%m%d%H'))
            append_item_to_dict(ifs_info[unique_id], 'model_name', 'ifs')
        except:
            pass
    
    return True, ifs_info


def download_gfs(gfs_info, gfs_download_dir, 
                 cur_analysis_time, fcst_length,
                 gfs_on_s3):
    gfs_dir_s3 = os.path.join(gfs_on_s3, cur_analysis_time.strftime('%Y/%m/%d/%H'))
    unique_id = str(uuid4())
    gfs_info[unique_id] = {}
    for fcst in range(0, fcst_length+1):
        valid_time = cur_analysis_time + timedelta(seconds=3600*fcst)
        gfs_filename = 'gfs.t{analysis_hour}z.pgrb2.0p25.f{fcst_length}'.format(
            analysis_hour='%02d'%(cur_analysis_time.hour),
            fcst_length='%03d'%(fcst))
        gfs_path_on_s3 = os.path.join(gfs_dir_s3, gfs_filename)
        gfs_path_local = os.path.join(gfs_download_dir, gfs_filename)
        
        try:
            if not os.path.exists(gfs_path_local):
                resources.copy(gfs_path_on_s3, gfs_path_local)
            append_item_to_dict(gfs_info[unique_id], 'datetime_analysis', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(gfs_info[unique_id], 'datetime_cold', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(gfs_info[unique_id], 'datetime_lbc', cur_analysis_time.strftime('%Y%m%d%H'))
            append_item_to_dict(gfs_info[unique_id], 'local_path', gfs_path_local)
            append_item_to_dict(gfs_info[unique_id], 'datetime_valid', valid_time.strftime('%Y%m%d%H'))
            append_item_to_dict(gfs_info[unique_id], 'model_name', 'gfs')
        except:
            pass
    
    return True, gfs_info

def download_wrf(wrf_info, wrf_download_dir, cur_analysis_time, fcst_length,
                 wrf_cycle_flag_dir, model_name,
                 status, region, role_name):
    
    def _get_wrfout_path(wrf_info, wrf_cycle_flag_dir, model_name,
                         cur_analysis_time,  status, region,
                         role_name):
        wrf_cycle_flag_dir2 = os.path.join(
                wrf_cycle_flag_dir, model_name, 
                cur_analysis_time.strftime('%Y%m%d%H'))
        cycle_flags = resources.list_files_in_directory(wrf_cycle_flag_dir2)
        for cycle_flag in cycle_flags:
            cycle_flag = os.path.join(wrf_cycle_flag_dir2, cycle_flag)
            cfg = load_file_from_s3(
                status, region, urlparse(cycle_flag).netloc,
                urlparse(cycle_flag).path[1:], markup='yaml',
                role_name=role_name)
            if not cfg['catch_up_run']:
                unique_name = cfg['uuid']
                wrf_info[unique_name] = {}
                wrf_info[unique_name]['datetime_analysis'] = str(cfg['datetime'])
                wrf_info[unique_name]['datetime_cold'] = str(cfg['datetime_cold'])
                wrf_info[unique_name]['datetime_lbc'] = str(cfg['lbc_datetime'])
                wrf_info[unique_name]['s3_path'] = str(cfg['wrfout_dir'])
                return unique_name, wrf_info
        
        return None, wrf_info
    
    # note: only one cycle will be used to match the real situation
    unique_name, wrf_info = _get_wrfout_path(
        wrf_info,
        wrf_cycle_flag_dir, model_name,
        cur_analysis_time,  status, region,
        role_name)

    if not unique_name:
        return False, wrf_info
    
    wrf_download_dir2 = os.path.join(
        wrf_download_dir, wrf_info[unique_name]['datetime_cold'],
        wrf_info[unique_name]['datetime_analysis'])
    
    wrf_info_out = {}
    wrf_info_out[unique_name] = {}
    for fcst in range(0, fcst_length+1):
        wrf_valid_time = datetime.strptime(wrf_info[unique_name]['datetime_analysis'], '%Y%m%d%H') + timedelta(seconds=3600*fcst)
        wrf_filename = 'wrf_hourly_{model_name}_d02_{valid_time}'.format(
            model_name=model_name,
            valid_time=wrf_valid_time.strftime('%Y-%m-%d_%H:%M:%S'))
        wrf_path_on_s3 = os.path.join(
            wrf_info[unique_name]['s3_path'], wrf_filename)
        
        wrf_path_local = os.path.join(
            wrf_download_dir2, wrf_filename)
        
        try:
            if not os.path.exists(wrf_path_local):
                resources.copy(wrf_path_on_s3, wrf_path_local)
            append_item_to_dict(wrf_info_out[unique_name], 'datetime_analysis', wrf_info[unique_name]['datetime_analysis'])
            append_item_to_dict(wrf_info_out[unique_name], 'datetime_cold', wrf_info[unique_name]['datetime_cold'])
            append_item_to_dict(wrf_info_out[unique_name], 'datetime_lbc', wrf_info[unique_name]['datetime_lbc'])
            append_item_to_dict(wrf_info_out[unique_name], 'local_path', wrf_path_local)
            append_item_to_dict(wrf_info_out[unique_name], 'datetime_valid', wrf_valid_time.strftime('%Y%m%d%H'))
            append_item_to_dict(wrf_info_out[unique_name], 'model_name', model_name)
        except:
            pass
    
    return True, wrf_info_out
        
        
def convert_unique_dict_to_pandas(work_dir, data_info, model_type, model_name=None):
    decoded_data_items = ['model_name', 
                          'datetime_analysis', 'datetime_cold', 'datetime_lbc',
                          'datetime_valid', 'local_path']
    decoded_data = {}
    for decoded_data_item in decoded_data_items:
        decoded_data[decoded_data_item] = []
        for data_key in data_info:
            # if decoded_data_item == 'unique_id':
            #    decoded_data[decoded_data_item].append(data_key)
            #else:
            decoded_data[decoded_data_item].extend(data_info[data_key][decoded_data_item])

    pandas_dataset = pandas.DataFrame.from_dict(decoded_data)
    
    csv_out = 'model_out_{}'.format(model_type)
    if model_name:
        csv_out = csv_out + '_' + model_name
    csv_out += '.csv'
    csv_out_path = os.path.join(work_dir, 'model', csv_out)
    pandas_dataset.to_csv(
        csv_out_path, header=True, columns=decoded_data_items)
                
        
def read_wrf(wrf_path, wrf_var, wrf_time):
    nc_fid = Dataset(wrf_path, 'r')
    times_list = nc_fid.variables['Times'][:]
    times_list = [''.join(i.tolist()) for i in times_list] 
    
    if wrf_time:
        times_index = times_list.index(
            wrf_time.strftime('%Y-%m-%d_%H:%M:%S'))
    else:
        times_index = 0
    
    if wrf_var == '10 metre speed':
        var_out = getvar(nc_fid, 'uvmet10_wspd_wdir')[0].values
    else:
        if wrf_var == 'td2':
            var_out = getvar(nc_fid, wrf_var, units='K').values
        else:
            var_out = getvar(nc_fid, wrf_var).values
    lats = nc_fid.variables['XLAT'][times_index, :]
    lons = nc_fid.variables['XLONG'][times_index, :]
    nc_fid.close()

    return lats, lons, var_out


def get_var_based_on_nx_ny(grbs, var, nx, ny):
    fid_var_list = grbs.select(name=var)
    if nx and ny:
        for fid_var in fid_var_list:
            if fid_var.Nx == nx and fid_var.Ny == ny:
                fid_lat, fid_lon = fid_var.latlons()
                break
    else:
        fid_var = fid_var_list[0]
        fid_lat, fid_lon = fid_var.latlons()

    return fid_var.values, fid_lat, fid_lon


def read_gfs(gfs_path, gfs_var, gfs_time, nx=None, ny=None):
    grbs = pygrib.open(gfs_path)
    try:
        if gfs_var == '10 metre speed':
            udata, clat, clon = get_var_based_on_nx_ny(
                        grbs, '10 metre U wind component', nx, ny)
            vdata, _, _ = get_var_based_on_nx_ny(
                        grbs, '10 metre V wind component', nx, ny)
            cdata = numpy.sqrt(udata**2 + vdata**2)
        else:
            cdata, clat, clon = get_var_based_on_nx_ny(
                          grbs, gfs_var, nx, ny)
        grbs.close()
    except ValueError:
        if gfs_var == 'Total Precipitation':
            fid_var = grbs.select(name='Visibility')[0]
            clat, clon = fid_var.latlons()
            cdata = numpy.zeros(fid_var.values.shape)
        else:
            raise Exception('the required field {} for '
                            'gfs is not there'.format(gfs_var))
    return clat, clon, cdata


def read_ifs(ifs_path, ifs_var, ifs_time, nx=None, ny=None):
    grbs = pygrib.open(ifs_path)
    try:
        if ifs_var == '10 metre speed':
            udata, clat, clon = get_var_based_on_nx_ny(
                        grbs, '10 metre U wind component', nx, ny)
            vdata, _, _ = get_var_based_on_nx_ny(
                        grbs, '10 metre V wind component', nx, ny)
            cdata = numpy.sqrt(udata**2 + vdata**2)
        else:
            cdata, clat, clon = get_var_based_on_nx_ny(
                          grbs, ifs_var, nx, ny)
        grbs.close()
    except ValueError:
        if ifs_var == 'Total precipitation':
            fid_var = grbs.select(name='Sea surface temperature')[0]
            clat, clon = fid_var.latlons()
            cdata = numpy.zeros(fid_var.values.shape)
        else:
            raise Exception('the required field {} for '
                            'ufs is not there'.format(ifs_var))
    return clat, clon, cdata


def obtain_model_rainfall_on_radar_grids(
        model_lats, model_lons, model_rainfall,
        radar_lats, radar_lons):

    def _resample_data(src_geo_def, target_geo_def):
        return src_geo_def.resample(target_geo_def)
    
    model_lons = model_lons % 360
    radar_lons = radar_lons % 360
    radar_grid_def = geometry.GridDefinition(lons=radar_lons, lats=radar_lats)
    
    model_grid_def = geometry.GridDefinition(
        lons=model_lons, lats=model_lats)
    model_img_container = image.ImageContainerNearest(model_rainfall, model_grid_def, radius_of_influence=100000, fill_value=None)
    model_rainfall_on_radar_grids = model_img_container.resample(radar_grid_def)
    
    model_rainfall_3d = numpy.zeros(
        (1, model_rainfall_on_radar_grids.shape[0], 
         model_rainfall_on_radar_grids.shape[1]))
    model_rainfall_3d[0, :, :] = model_rainfall_on_radar_grids.image_data
    
    return model_rainfall_3d


def write_model_rainfall_on_radar_grids(
        work_dir, model_type, accumulation_length=1, model_name=None):

    def _looking_for_the_previous_fcst(
            model_outs,
            cur_model_name, cur_datetime_analysis,
            cur_datetime_cold, cur_datetime_lbc,
            cur_datetime_valid, accumulation_length):
        pre_datetime_valid = datetime.strptime(
            cur_datetime_valid, '%Y%m%d%H') - timedelta(seconds=3600*accumulation_length)  
        pre_model_out = model_outs.loc[
            (model_outs['model_name'] == cur_model_name) & 
            (model_outs['datetime_analysis'] == int(cur_datetime_analysis)) &
            (model_outs['datetime_cold'] == int(cur_datetime_cold)) &
            (model_outs['datetime_lbc'] == int(cur_datetime_lbc)) &
            (model_outs['datetime_valid'] == int(pre_datetime_valid.strftime('%Y%m%d%H')))]
        if len(pre_model_out) == 0:
            return False, None
        elif len(pre_model_out) == 1:
            return pre_model_out['local_path'][pre_model_out.index[0]], pre_datetime_valid.strftime('%Y%m%d%H')
        else:
            raise Exception('found multiple pre_model_out for '
                            '{} at {}'.format(cur_model_name, cur_datetime_valid))
    
    
    def _get_wrf_rainfall(cur_datetime_local_path, pre_datetime_local_path,
                          cur_datetime_valid, pre_datetime_valid):
        lats, lons, cur_rainnc = read_wrf(
                cur_datetime_local_path, 'RAINNC',
                datetime.strptime(cur_datetime_valid, '%Y%m%d%H'))
        _, _, cur_rainc = read_wrf(
                cur_datetime_local_path, 'RAINC',
                datetime.strptime(cur_datetime_valid, '%Y%m%d%H'))
        _, _, pre_rainnc = read_wrf(
                pre_datetime_local_path, 'RAINNC',
                datetime.strptime(pre_datetime_valid, '%Y%m%d%H'))
        _, _, pre_rainc = read_wrf(
                pre_datetime_local_path, 'RAINC',
                datetime.strptime(pre_datetime_valid, '%Y%m%d%H'))
        wrf_rainfall = cur_rainnc + cur_rainc - pre_rainnc - pre_rainc
        
        return lats, lons, wrf_rainfall
    
    def _get_gfs_rainfall(cur_datetime_local_path, pre_datetime_local_path,
                          cur_datetime_valid, pre_datetime_valid):
        lats, lons, cur_rainfall = read_gfs(
                cur_datetime_local_path, 'Total Precipitation',
                datetime.strptime(cur_datetime_valid, '%Y%m%d%H'))
        _, _, pre_rainfall = read_gfs(
                pre_datetime_local_path, 'Total Precipitation',
                datetime.strptime(pre_datetime_valid, '%Y%m%d%H'))
        gfs_rainfall = cur_rainfall - pre_rainfall
        
        return lats, lons, gfs_rainfall

    def _get_ifs_rainfall(cur_datetime_local_path, pre_datetime_local_path,
                          cur_datetime_valid, pre_datetime_valid):
        lats, lons, cur_rainfall = read_ifs(
                cur_datetime_local_path, 'Total precipitation',
                datetime.strptime(cur_datetime_valid, '%Y%m%d%H'))
        _, _, pre_rainfall = read_ifs(
                pre_datetime_local_path, 'Total precipitation',
                datetime.strptime(pre_datetime_valid, '%Y%m%d%H'))
        gfs_rainfall = cur_rainfall - pre_rainfall
        
        return lats, lons, gfs_rainfall*3600.0


    def _assign_model_map():
        model_map = {}
        for ele in ['dx', 'dy', 'cen_lat', 'cen_lon', 
                    'truelat1', 'truelat2', 
                    'moad_cen_lat', 'standard_lon']:
            model_map[ele] = -999.0
        return model_map
    
    
    # ------------------------------
    # program starts
    # ------------------------------
    model_out_name = 'model_out_{}'.format(model_type)
    if model_name:
        model_out_name = model_out_name + '_' + model_name
    
    model_out_info = os.path.join(
        work_dir, 'model', '{}.csv'.format(model_out_name))
    model_outs = pandas.read_csv(model_out_info, ',')
    model_rainfall_on_radar_grids_info = {}
    for index, model_out in model_outs.iterrows():
        cur_model_name = model_out['model_name']
        cur_datetime_analysis = str(model_out['datetime_analysis'])
        cur_datetime_cold = str(model_out['datetime_cold'])
        cur_datetime_lbc = str(model_out['datetime_lbc'])
        cur_datetime_valid = str(model_out['datetime_valid'])
        cur_datetime_local_path = model_out['local_path']
        pre_datetime_local_path, pre_datetime_valid = \
             _looking_for_the_previous_fcst(
                model_outs,
                cur_model_name, cur_datetime_analysis,
                cur_datetime_cold, cur_datetime_lbc,
                cur_datetime_valid, accumulation_length)
        if not pre_datetime_local_path:
            continue
        
        cur_datetime_local_dir = os.path.dirname(cur_datetime_local_path)
        if model_type == 'wrf':
            model_lats, model_lons, model_rainfall = _get_wrf_rainfall(
                cur_datetime_local_path, pre_datetime_local_path,
                cur_datetime_valid, pre_datetime_valid)
        if model_type == 'gfs':
            model_lats, model_lons, model_rainfall = _get_gfs_rainfall(
                cur_datetime_local_path, pre_datetime_local_path,
                cur_datetime_valid, pre_datetime_valid)
        if model_type == 'ifs':
            model_lats, model_lons, model_rainfall = _get_ifs_rainfall(
                cur_datetime_local_path, pre_datetime_local_path,
                cur_datetime_valid, pre_datetime_valid)

        radar_lats, radar_lons = extract_radar_grids(
            work_dir, datetime.strptime(cur_datetime_valid, '%Y%m%d%H'))
        model_rainfall_on_radar_grids = obtain_model_rainfall_on_radar_grids(
                model_lats, model_lons, model_rainfall,
                radar_lats, radar_lons)
        cur_model_rainfall_path = os.path.join(
            cur_datetime_local_dir, 
            'model_rainfall_on_radar_grids_from_{}_to_{}.nc'.format(
                pre_datetime_valid, cur_datetime_valid))
        model_map = _assign_model_map()
        download_and_write_radar.write_radar_nc(
            'model_rainfall_on_radar_grids', cur_model_rainfall_path, 
            radar_lats, radar_lons, None, [1], 
            model_rainfall_on_radar_grids, model_map)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'model_name', cur_model_name)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'datetime_analysis', cur_datetime_analysis)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'datetime_lbc', cur_datetime_lbc)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'datetime_cold', cur_datetime_cold)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'datetime_valid', cur_datetime_valid)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'accumulation_length', accumulation_length)
        append_item_to_dict(model_rainfall_on_radar_grids_info, 
                            'local_path', cur_model_rainfall_path)



    pandas_dataset = pandas.DataFrame.from_dict(model_rainfall_on_radar_grids_info)
    csv_out_name = 'model_rainfall_on_radar_grids_{}'.format(model_type)
    if model_name:
        csv_out_name = csv_out_name + '_{}'.format(model_name)
    csv_out = os.path.join(
        work_dir, 'model', '{}.csv'.format(csv_out_name))
    csv_out_path = os.path.join(work_dir, 'model', csv_out)
    decoded_data_items = ['model_name', 
                          'datetime_analysis', 'datetime_cold', 'datetime_lbc',
                          'datetime_valid', 'accumulation_length', 'local_path']
    pandas_dataset.to_csv(
        csv_out_path, header=True, columns=decoded_data_items)


def match_model_to_conv_grids(work_dir, model_type, obs_from, obs_to, model_name=None):
    def _obtain_model_and_obs_coords(model_lats, model_lons, obs_lats, obs_lons):
        obs_lons = obs_lons % 360.0
        model_lons = model_lons % 360.0
        
        combined_model_coords_arrays = numpy.dstack(
            [model_lats.ravel(), model_lons.ravel()])[0]
        combined_obs_coords_arrays = numpy.dstack(
            [obs_lats.ravel(), obs_lons.ravel()])[0]
        return combined_model_coords_arrays, combined_obs_coords_arrays

    def _do_kdtree(combined_x_y_arrays,points):
        mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
        dist, indexes = mytree.query(points)
        return indexes

    def _get_i_j_from_model_index(
            combined_model_coords_arrays,
            model_index, model_lat, model_lon):
        found_model_latlon = []
        found_model_ij = []
        for cur_model_index in model_index:
            cur_model_coords = combined_model_coords_arrays[cur_model_index]
            cur_model_lat = cur_model_coords[0]
            cur_model_lon = cur_model_coords[1]
            i = numpy.where(model_lat == cur_model_lat)[0][0]
            j = numpy.where(model_lon == cur_model_lon)[1][0]
            found_model_latlon.append((model_lat[i, j], model_lon[i, j]))
            found_model_ij.append((i, j))
        
        return found_model_ij, found_model_latlon
            
    model_obs_match_filename = 'conv_model_obs_match_from_{}_to_{}_{}'.format(
        obs_from.strftime('%Y%m%d%H'), obs_to.strftime('%Y%m%d%H'), model_type)
    if model_name:
        model_obs_match_filename = model_obs_match_filename + '_{}'.format(model_name)
    model_obs_match_filename += '.csv'

    model_obs_match_path = os.path.join(
        work_dir, 'model', model_obs_match_filename)
    
    if os.path.exists(model_obs_match_path):
        return

    model_out_path = os.path.join(
        work_dir, 'model', 'model_out_{}'.format(model_type))
    if model_name:
        model_out_path += '_{}'.format(model_name)
    model_out_path += '.csv'
    
    model_outs = pandas.read_csv(model_out_path, ',')
    example_model_local_path = model_outs['local_path'][0]
    if model_type == 'ifs':
        example_model_var = '2 metre temperature'
        model_lats, model_lons, _ = read_ifs(
            example_model_local_path, 
            example_model_var, None)
        model_lons = model_lons % 360.0
    if model_type == 'gfs':
        example_model_var = 'Temperature'
        model_lats, model_lons, _ = read_ifs(
            example_model_local_path, 
            example_model_var, None)
        model_lons = model_lons % 360.0
    if model_type == 'wrf':
        model_lats, model_lons, _  = read_wrf(
            example_model_local_path, 'T2', None)
        model_lons = model_lons % 360.0
        
    model_shape = model_lats.shape
    conv_obs_latlon_path = os.path.join(
        work_dir, 'conv', 
        'conv_obs_latlon_from_{}_to_{}.csv'.format(
            obs_from.strftime('%Y%m%d%H'), 
            obs_to.strftime('%Y%m%d%H')))
    
    conv_obs_latlon = pandas.read_csv(conv_obs_latlon_path, ',')
    
    combined_model_coords_arrays, combined_obs_coords_arrays \
        = _obtain_model_and_obs_coords(
            model_lats, model_lons, 
            conv_obs_latlon['obs_lat'], 
            conv_obs_latlon['obs_lon'])
    
    model_index = _do_kdtree(
        combined_model_coords_arrays,
        combined_obs_coords_arrays)

    found_model_ij, found_model_latlon = _get_i_j_from_model_index(
                combined_model_coords_arrays,
                model_index, model_lats, model_lons)
    
    data_out = {}
    for ele in ['obs_lat', 'obs_lon', 'model_lat', 'model_lon', 
                'model_i', 'model_j', 'model_nx', 'model_ny']:
        data_out[ele] = []
    
    for i in range(0, len(found_model_ij)):
        data_out['obs_lat'].append(combined_obs_coords_arrays[i][0])
        data_out['obs_lon'].append(combined_obs_coords_arrays[i][1])
        data_out['model_lat'].append(found_model_latlon[i][0])
        data_out['model_lon'].append(found_model_latlon[i][1])
        data_out['model_i'].append(found_model_ij[i][0])
        data_out['model_j'].append(found_model_ij[i][1])
        data_out['model_nx'].append(int(model_shape[1]))
        data_out['model_ny'].append(int(model_shape[0]))
    pandas_dataset = pandas.DataFrame.from_dict(data_out)
    pandas_dataset = pandas_dataset.round(
        {'obs_lat': 3, 'obs_lon': 3,
         'model_lat': 3, 'model_lon': 3})

    decoded_data_items = ['obs_lat', 'obs_lon', 'model_lat', 'model_lon',
                          'model_i', 'model_j', 'model_nx', 'model_ny']

    pandas_dataset.to_csv(
        model_obs_match_path, header=True, columns=decoded_data_items)
    
def read_conv_obs(conv_obs_path, datetime_valid, conv_obs_name):
    conv_obs_outs = pandas.read_csv(conv_obs_path, ',')
    conv_obs_outs0 = conv_obs_outs.loc[(
        conv_obs_outs['obs_time']==
            int(datetime_valid.strftime('%Y%m%d%H%M%S')))]
    conv_obs_outs0 = conv_obs_outs0.round({'obs_lat': 3, 'obs_lon': 3})
    return conv_obs_outs0[['obs_id', 'obs_lat', 
                           'obs_lon', 'obs_time', 
                           conv_obs_name]]
    
            


    
    
