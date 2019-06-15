import os
import pandas
from datetime import datetime, timedelta
import numpy
from netCDF4 import Dataset
from data_download_upload import append_item_to_dict
from data_download_upload import read_ifs, read_conv_obs, read_gfs, read_wrf

from nwp_verification import CONV_MODEL_OBS_NAME_MATCH

def read_cappi(radar_cappi_list):
    def _dbz2r(dbz):
        z = 10.0**(dbz/10.0)
        r = (z/200.0)**(1.0/1.6)
        return r 
       
    for i, f in enumerate(radar_cappi_list):
        nc_fid = Dataset(f, 'r')
        radar_dbz = nc_fid.variables['reflectivity']
        radar_dbz = numpy.max(radar_dbz, 0)
        radar_r = _dbz2r(radar_dbz)
        if i == 0:
            total_radar_r = radar_r
            radar_lat = nc_fid.variables['point_latitude'][0, :]
            radar_lon = nc_fid.variables['point_longitude'][0, :]
        else:
            total_radar_r += radar_r

    return radar_lat, radar_lon, total_radar_r[0, :]


def read_model(model_data_list):
    for i, model_data_path in enumerate(model_data_list):
        nc_fid = Dataset(model_data_path, 'r')
        model_r = nc_fid.variables['model_rainfall_on_radar_grids'][0, :]
        if i == 0:
            total_model_r = model_r
            model_lat = nc_fid.variables['XLAT'][:]
            model_lon = nc_fid.variables['XLONG'][:]
        else:
            total_model_r += model_r
    return model_lat, model_lon, total_model_r


def get_radar_path(work_dir, datetime_valid, accumulation_length):
    radar_work_dir = os.path.join(work_dir, 'radar')
    radar_filename_list = []
    for i in range(0, accumulation_length):
        cur_radar_filename = 'cappi_reflectivity_{}.nc'.format(
            (datetime_valid - timedelta(seconds=3600*i)).strftime('%Y%m%d%H%M'))
        if not os.path.exists(os.path.join(radar_work_dir, cur_radar_filename)):
            return False
        radar_filename_list.append(
            os.path.join(radar_work_dir, cur_radar_filename))

    return radar_filename_list


def get_model_path(datetime_valid, model_out, 
                   verif_target_accumulation_length):
    def _determine_the_goback_hours(
            model_out, datetime_valid,
            model_name, datetime_cold, datetime_analysis,
            datetime_lbc, verif_target_accumulation_length):
        model_out0 = model_out.loc[
            (model_out['datetime_valid'] == datetime_valid) &
            (model_out['model_name'] == model_name) &               
            (model_out['datetime_cold'] == datetime_cold) &
            (model_out['datetime_analysis'] == datetime_analysis) &
            (model_out['datetime_lbc'] == datetime_lbc)]
        goback_hours = int(verif_target_accumulation_length)/int(
            model_out0['accumulation_length'])
        return goback_hours, int(model_out0['accumulation_length'])
    
    model_path_list = []
    model_name = model_out['model_name']
    datetime_cold = model_out['datetime_cold']
    datetime_analysis = model_out['datetime_analysis']
    datetime_lbc = model_out['datetime_lbc']
    goback_hours, model_data_accum_length = _determine_the_goback_hours(
            model_out, datetime_valid, 
            model_name, datetime_cold, datetime_analysis,
            datetime_lbc,
            verif_target_accumulation_length)
    for i in range(0, goback_hours):
        cur_datetime_valid = datetime_valid - timedelta(seconds=3600*i*model_data_accum_length)
        cur_model_out = model_out.loc[(model_out['datetime_valid'] == cur_datetime_valid) &
                                      (model_out['model_name'] == model_name) &
                                      (model_out['datetime_cold'] == datetime_cold) &
                                      (model_out['datetime_analysis'] == datetime_analysis) &
                                      (model_out['datetime_lbc'] == datetime_lbc)]
        if len(cur_model_out) == 1:
            cur_local_path = cur_model_out['local_path'][
                cur_model_out['local_path'].index[0]]
            if not os.path.exists(cur_local_path):
                return False
            model_path_list.append(cur_local_path)
        else:
            return False

    return model_path_list


def get_rainfall_verif_table(model_rainfall, radar_rainfall, threshold):
    hit = 0
    miss = 0
    false_alarm = 0
    correct_negative = 0
    for i in range(0, model_rainfall.shape[0]):
        for j in range(0, model_rainfall.shape[1]):
            if model_rainfall[i, j] >= threshold and radar_rainfall[i, j] >= threshold:
                hit += 1
            elif model_rainfall[i, j] < threshold and radar_rainfall[i, j] >= threshold:
                miss += 1
            elif model_rainfall[i, j] >= threshold and radar_rainfall[i, j] < threshold:
                false_alarm += 1
            elif model_rainfall[i, j] < threshold and radar_rainfall[i, j] < threshold:
                correct_negative += 1
    return hit, miss, false_alarm, correct_negative

def rainfall_verification(work_dir, rainfall_verification_task, 
                          model_type, verif_target_accumulation_length, 
                          model_name=None):

    def _obtain_the_model_on_radar_grids_info(work_dir, model_name, model_type):
        model_rainfall_on_radar_grids_filename = \
            'model_rainfall_on_radar_grids_{}'.format(model_type)
        if model_name:
            model_rainfall_on_radar_grids_filename += '_{}'.format(model_name)
        model_rainfall_on_radar_grids_filename += '.csv'
        
        model_rainfall_on_radar_grids_path = os.path.join(
            work_dir, 'model', model_rainfall_on_radar_grids_filename)
        model_rainfall_outs = pandas.read_csv(
            model_rainfall_on_radar_grids_path, ',')

        model_rainfall_outs[['datetime_analysis', 'datetime_cold', 
            'datetime_lbc', 'datetime_valid']] = \
                model_rainfall_outs[['datetime_analysis', 'datetime_cold', 
                                     'datetime_lbc', 'datetime_valid']].apply(
                    pandas.to_datetime, format='%Y%m%d%H')

        return model_rainfall_outs
    
    def _attach_hour_info_to_model_rainfall_dataset(model_rainfall_outs):
        model_rainfall_outs['datetime_analysis_hour'] = \
            model_rainfall_outs.datetime_analysis.apply(lambda x: x.hour)
        model_rainfall_outs['datetime_cold_hour'] = \
            model_rainfall_outs.datetime_cold.apply(lambda x: x.hour)
        model_rainfall_outs['datetime_lbc_hour'] = \
            model_rainfall_outs.datetime_lbc.apply(lambda x: x.hour)
        model_rainfall_outs['datetime_valid_hour'] = \
            model_rainfall_outs.datetime_valid.apply(lambda x: x.hour)
        
        return model_rainfall_outs
    
    def _extract_the_verification_task_info(rainfall_verification_task):
        required_model_name = \
            rainfall_verification_task['model_name']
        required_datetime_analysis_hour = \
            rainfall_verification_task['datetime_analysis_hour']
        required_datetime_cold_hour = \
            rainfall_verification_task['datetime_cold_hour']
        required_datetime_lbc_hour = \
            rainfall_verification_task['datetime_lbc_hour']
        required_verif_thres_list = \
            rainfall_verification_task['rainfall_threshold_list']      
        return (required_model_name, required_datetime_analysis_hour,
                required_datetime_cold_hour, required_datetime_lbc_hour,
                required_verif_thres_list)
        
    def _extract_model_rainfall_on_radar_grids_info_with_condition(
            model_rainfall_on_radar_grids_info,
            required_model_name, required_datetime_analysis_hour,
            required_datetime_cold_hour, required_datetime_lbc_hour):
        required_model_out = model_rainfall_on_radar_grids_info.loc[
            (model_rainfall_on_radar_grids_info['model_name'] == 
                        required_model_name) & 
            (model_rainfall_on_radar_grids_info['datetime_analysis_hour'] == 
                        int(required_datetime_analysis_hour)) &
            (model_rainfall_on_radar_grids_info['datetime_cold_hour'] == 
                        int(required_datetime_cold_hour)) &
            (model_rainfall_on_radar_grids_info['datetime_lbc_hour'] == 
                        int(required_datetime_lbc_hour))]
        return required_model_out
    
    def _construct_rainfall_verif_dataset(
            data_used_for_rainfall_verif, model_out,
            required_accumulation_length,
            cur_model_path_list, cur_radar_path_list,
            hit, miss, false_alarm, correct_negative,
            cur_datetime_valid, threshold):
        # append basic information
        append_item_to_dict(data_used_for_rainfall_verif,
                    'model_name', model_out['model_name'])
        append_item_to_dict(data_used_for_rainfall_verif,
                    'accumulation_length', required_accumulation_length)
        
        # apppend the files used to create this dataset
        cur_model_filename_list = [os.path.basename(cur_model_path) for
                                   cur_model_path in cur_model_path_list]
        append_item_to_dict(data_used_for_rainfall_verif,
                    'model_filename_list', cur_model_filename_list)
        cur_radar_filename_list = [os.path.basename(cur_radar_path) for
                                   cur_radar_path in cur_radar_path_list]
        append_item_to_dict(data_used_for_rainfall_verif,
                    'radar_filename_list', cur_radar_filename_list)        

        # append datetime
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_valid', cur_datetime_valid.strftime('%Y%m%d%H'))
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_analysis', model_out['datetime_analysis'].strftime('%Y%m%d%H'))
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_cold', model_out['datetime_cold'].strftime('%Y%m%d%H'))
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_lbc', model_out['datetime_lbc'].strftime('%Y%m%d%H'))

        # append hours
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_cold_hour', model_out['datetime_cold_hour'])
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_analysis_hour', model_out['datetime_analysis_hour'])
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_lbc_hour', model_out['datetime_lbc_hour'])
        append_item_to_dict(data_used_for_rainfall_verif,
                    'datetime_valid_hour', int(cur_datetime_valid.strftime('%H')))     

        # append hit, miss, false_alarm, correct_negative
        append_item_to_dict(data_used_for_rainfall_verif,
                    'threshold', threshold)
        append_item_to_dict(data_used_for_rainfall_verif,
                    'hit', hit)
        append_item_to_dict(data_used_for_rainfall_verif,
                    'miss', miss)
        append_item_to_dict(data_used_for_rainfall_verif,
                    'false_alarm', false_alarm)
        append_item_to_dict(data_used_for_rainfall_verif,
                    'correct_negative', correct_negative)

        return data_used_for_rainfall_verif

    
    def _write_verif_results(
                verif_results_dir, data_used_for_rainfall_verif):
        rainfall_verif_results_filename = 'rainfall_verif_results.csv'
        rainfall_verif_results_path = os.path.join(
            verif_results_dir, rainfall_verif_results_filename)
        pandas_dataset = pandas.DataFrame.from_dict(data_used_for_rainfall_verif)
     
        decoded_data_items = ['model_name',
                              'datetime_analysis', 'datetime_cold', 'datetime_lbc', 'datetime_valid',
                              'datetime_cold_hour', 'datetime_analysis_hour', 'datetime_lbc_hour', 'datetime_valid_hour',
                              'accumulation_length', 'threshold',
                              'hit', 'miss', 'false_alarm', 'correct_negative',
                              'model_filename_list', 'radar_filename_list']
        
        try:
            pandas_dataset.to_csv(
                rainfall_verif_results_path, header=True, 
                mode='a', columns=decoded_data_items)
    
            # drop the duplicates data
            dataset_out = pandas.read_csv(
                rainfall_verif_results_path, ',', names=decoded_data_items)
            dataset_out = dataset_out.drop_duplicates()
            dataset_out.to_csv(
                rainfall_verif_results_path, header=False)
        except KeyError:
            pass

    # --------------------------------------------------
    # program starts
    # --------------------------------------------------
    # step 1: create verif_results_dir directory
    verif_results_dir = os.path.join(work_dir, 'verif_results')
    if not os.path.exists(verif_results_dir):
        os.makedirs(verif_results_dir)

    # step 2: extract the rainfall verification task
    (required_model_name, required_datetime_analysis_hour,
     required_datetime_cold_hour, required_datetime_lbc_hour,
     required_verif_thres_list) = _extract_the_verification_task_info(
                    rainfall_verification_task)

    # step 3: get the information of model rainfall
    #         on the radar grids, and convert 
    #         all datetime from string to datetime type
    model_rainfall_on_radar_grids_info = \
        _obtain_the_model_on_radar_grids_info(
            work_dir, model_name, model_type)
    
    # step 4: add the columns of hours in pandas
    model_rainfall_on_radar_grids_info = \
        _attach_hour_info_to_model_rainfall_dataset(
            model_rainfall_on_radar_grids_info)

    # step 5: set the condition of hour on
    #         * model_name
    #         * datetime_analysis_hour
    #         * datetime_cold_hour
    #         * datetime_lbc_hour
    #         on "model_rainfall_on_radar_grids_info"
    required_model_out = \
        _extract_model_rainfall_on_radar_grids_info_with_condition(
            model_rainfall_on_radar_grids_info,
            required_model_name, required_datetime_analysis_hour,
            required_datetime_cold_hour, required_datetime_lbc_hour)

    # step 6: match the model path and radar path, and get the model and
    #         radar rainfall
    data_used_for_rainfall_verif = {}
    
    # step 6.1: start matching processing: 
    #         go through all the dataset rows with required
    #         "model_name,  datetime_analysis_hour, 
    #         datetime_cold_hour, datetime_lbc_hour"
    for index, model_out in required_model_out.iterrows():
        # step 6.1.1: what is the datatime_valid for the current row
        cur_datetime_valid = model_out['datetime_valid']
        
        # step 6.2: determine the required model paths and radar paths
        #           based on "cur_datetime_valid" and 
        #                    "verif_target_accumulation_length"
        #           * for the model: since we may need to go forward/backward to find the path, 
        #             we use the entire dataset of "required_model_out" instead of 
        #             just the current row (with only one valid time)
        #           * for the radar: we directly go to radar data directory to find the data path
        cur_model_path_list = get_model_path(
            cur_datetime_valid, required_model_out, 
            verif_target_accumulation_length)
        cur_radar_path_list = get_radar_path(
            work_dir, model_out['datetime_valid'], 
            verif_target_accumulation_length)
        
        if cur_radar_path_list and cur_model_path_list:
            # step 6.3: get the total rainfall from both radar and model
            radar_lat, radar_lon, total_radar_r = read_cappi(cur_radar_path_list)
            model_lat, model_lon, total_model_r = read_model(cur_model_path_list)
            if not total_radar_r.shape == total_model_r.shape:
                continue
               
            # step 6.5: get hits/miss/false_alarm/correct_negative based on threshold
            for thres in required_verif_thres_list:
                hit, miss, false_alarm, correct_negative = \
                    get_rainfall_verif_table(
                        total_model_r, total_radar_r, thres)

                # step 6.5.1: construct the verification dataset with the elements of
                #           * model_name
                #           * accumulation_length
                #           * threshold
                #           * model_filename_list and radar_filename_list
                #           * datetime_valid + its hour
                #           * datetime_analysis + its hour
                #           * datetime_cold + its hour
                #           * datetime_lbc + its hour
                #           * hits/miss/false_alarm/correct_negative
                data_used_for_rainfall_verif = _construct_rainfall_verif_dataset(
                            data_used_for_rainfall_verif, model_out,
                            verif_target_accumulation_length,
                            cur_model_path_list, cur_radar_path_list,
                            hit, miss, false_alarm, correct_negative,
                            cur_datetime_valid, thres)

    _write_verif_results(
        verif_results_dir, data_used_for_rainfall_verif)
    

def conv_verification(
        work_dir, model_type, 
        conv_conv_verification_task,
        obs_from, obs_to, 
        model_name=None):
    
    def _get_model_out_info(model_type, model_name, work_dir):
        # step 1: locate the model out information
        model_out_info_filename = 'model_out_{}'.format(model_type)
        if model_name:
            model_out_info_filename = \
                model_out_info_filename + '_{}'.format(model_name)
        model_out_info_filename += '.csv'
        
        model_out_info_path = os.path.join(
            work_dir, 'model', model_out_info_filename)
        model_out_info = pandas.read_csv(
                model_out_info_path, ',')
    
        model_out_info[['datetime_analysis', 'datetime_cold', 
            'datetime_valid']] = \
                model_out_info[['datetime_analysis', 'datetime_cold', 
                                'datetime_valid']].apply(
                    pandas.to_datetime, format='%Y%m%d%H')

        try:
            model_out_info[['datetime_lbc']] = model_out_info [['datetime_lbc']].apply(pandas.to_datetime, format='%Y%m%d%H')
        except ValueError:
            pass
        return model_out_info


    def _attach_hour_info_to_model_dataset(model_outs):
        model_outs['datetime_analysis_hour'] = \
            model_outs.datetime_analysis.apply(lambda x: x.hour)
        model_outs['datetime_cold_hour'] = \
            model_outs.datetime_cold.apply(lambda x: x.hour)
        try:
            model_outs['datetime_lbc_hour'] = \
                model_outs.datetime_lbc.apply(lambda x: x.hour)
        except AttributeError:
           model_outs['datetime_lbc_hour'] = None
        model_outs['datetime_valid_hour'] = \
            model_outs.datetime_valid.apply(lambda x: x.hour)
        
        return model_outs

    def _extract_the_verification_task_info(conv_verification_task):
        required_model_name = \
            conv_verification_task['model_name']
        required_datetime_analysis_hour = \
            conv_verification_task['datetime_analysis_hour']
        required_datetime_cold_hour = \
            conv_verification_task['datetime_cold_hour']
        required_datetime_lbc_hour = \
            conv_verification_task['datetime_lbc_hour']
        required_conv_field_list = \
            conv_verification_task['conv_field_list']      
        return (required_model_name, required_datetime_analysis_hour,
                required_datetime_cold_hour, required_datetime_lbc_hour,
                required_conv_field_list)

    def _extract_model_out_with_condition(
            model_out_info,
            required_model_name, required_datetime_analysis_hour,
            required_datetime_cold_hour, required_datetime_lbc_hour):
        required_model_out = model_out_info.loc[
            (model_out_info['model_name'] == 
                        required_model_name) & 
            (model_out_info['datetime_analysis_hour'] == 
                        int(required_datetime_analysis_hour)) &
            (model_out_info['datetime_cold_hour'] == 
                        int(required_datetime_cold_hour))]
        try:
            required_model_out = required_model_out.loc[
                (model_out_info['datetime_lbc_hour'] == 
                        int(required_datetime_lbc_hour))]
        except TypeError:
            pass

        return required_model_out
    
    def _get_model_obs_out_info(work_dir, obs_from, obs_to, model_type, model_name):
        model_obs_match_filename = 'conv_model_obs_match_from_{}_to_{}_{}'.format(
            obs_from.strftime('%Y%m%d%H'), obs_to.strftime('%Y%m%d%H'), model_type)
        if model_name:
            model_obs_match_filename += '_{}'.format(model_name)

        model_obs_match_filename += '.csv'
        model_obs_match_path = os.path.join(
            work_dir, 'model', model_obs_match_filename)
        model_obs_match_info = pandas.read_csv(
                model_obs_match_path, ',')
        return model_obs_match_info

    def _get_match_pairs(
            conv_verif_out,
            model_lat, model_lon, model_data,
            conv_obs, model_obs_match_info,
            conv_verif_field,
            model_type, model_name,
            datetime_valid,
            datetime_analysis,
            datetime_lbc,
            datetime_valid_hour,
            datetime_analysis_hour,
            datetime_lbc_hour):

        for index, cur_obs in conv_obs.iterrows():
            cur_obs_lat = cur_obs['obs_lat']
            cur_obs_lon_360 = cur_obs['obs_lon'] % 360.0
            cur_obs_lon = cur_obs['obs_lon']
            cur_model_obs_match_info = model_obs_match_info.loc[
                (model_obs_match_info['obs_lat'] == numpy.float64(cur_obs_lat)) &
                (model_obs_match_info['obs_lon'] == numpy.float64(cur_obs_lon_360))]
            if len(cur_model_obs_match_info) != 1:
                continue
            cur_model_i = int(cur_model_obs_match_info['model_i'])
            cur_model_j = int(cur_model_obs_match_info['model_j'])
            model_lat_from_ij = float(model_lat[cur_model_i, cur_model_j])
            model_lon_from_ij = float(model_lon[cur_model_i, cur_model_j])
            cur_model_lat = float(cur_model_obs_match_info['model_lat'])
            cur_model_lon = float(cur_model_obs_match_info['model_lon'])
            
            if numpy.isclose(model_lat_from_ij, cur_model_lat) and numpy.isclose(
                    model_lon_from_ij, cur_model_lon):
                cur_model_data_value = model_data[cur_model_i, cur_model_j]
                cur_obs_data = conv_obs.loc[
                    (conv_obs['obs_lat'] == numpy.float64(cur_obs_lat)) &
                    (conv_obs['obs_lon'] == numpy.float64(cur_obs_lon))]
                if len(cur_obs_data) == 1:
                    cur_obs_lon = cur_obs_lon % 360.0
                    cur_obs_data_value = cur_obs_data[
                        CONV_MODEL_OBS_NAME_MATCH[conv_verif_field]]
                    cur_obs_data_value_index = cur_obs_data_value.index[0]
                    cur_obs_data_value = cur_obs_data_value[cur_obs_data_value_index]
                    append_item_to_dict(conv_verif_out, 'obs_lat', cur_obs_lat)
                    append_item_to_dict(conv_verif_out, 'obs_lon', cur_obs_lon) 
                    append_item_to_dict(conv_verif_out, 'model_lat', cur_model_lat)
                    append_item_to_dict(conv_verif_out, 'model_lon', cur_model_lon)
                    append_item_to_dict(conv_verif_out, 'conv_field', conv_verif_field)
                    append_item_to_dict(conv_verif_out, 'datetime_valid', 
                                        datetime_valid.strftime('%Y%m%d%H')) 
                    append_item_to_dict(conv_verif_out, 'datetime_analysis', 
                                        datetime_analysis.strftime('%Y%m%d%H'))
                    try:
                        append_item_to_dict(conv_verif_out, 'datetime_lbc', 
                                        datetime_lbc.strftime('%Y%m%d%H'))
                    except AttributeError:
                        append_item_to_dict(conv_verif_out, 'datetime_lbc', None)
                        pass

                    append_item_to_dict(conv_verif_out, 'datetime_valid_hour', 
                                        int(datetime_valid_hour)), 
                    append_item_to_dict(conv_verif_out, 'datetime_analysis_hour', 
                                        int(datetime_analysis_hour)) 
                    try:
                        append_item_to_dict(conv_verif_out, 'datetime_lbc_hour', 
                                            int(datetime_lbc_hour))
                    except TypeError:
                        append_item_to_dict(conv_verif_out, 'datetime_lbc_hour', None)
                        pass

                    append_item_to_dict(conv_verif_out, 'model_value', 
                                        cur_model_data_value)
                    append_item_to_dict(conv_verif_out, 'obs_value', 
                                        cur_obs_data_value)
                    if model_name:
                        append_item_to_dict(conv_verif_out, 'model_name', 
                                        model_name)
                    else:
                        append_item_to_dict(conv_verif_out, 'model_name',
                                        model_type)
        return conv_verif_out        
     

    def _write_verif_results(
                verif_results_dir, data_used_for_conv_verif):
        conv_verif_results_filename = 'conv_verif_results.csv'
        conv_verif_results_path = os.path.join(
            verif_results_dir, conv_verif_results_filename)
        pandas_dataset = pandas.DataFrame.from_dict(data_used_for_conv_verif)

        decoded_data_items = ['model_name',
                              'datetime_valid', 'datetime_analysis', 'datetime_lbc',
                              'datetime_valid_hour', 'datetime_analysis_hour', 'datetime_lbc_hour',
                              'obs_lat', 'obs_lon', 'model_lat', 'model_lon', 'conv_field',
                              'model_value', 'obs_value']
        try:
            pandas_dataset.to_csv(
                conv_verif_results_path, header=True,
                mode='a', columns=decoded_data_items)

            # drop the duplicates data
            dataset_out = pandas.read_csv(
                conv_verif_results_path, ',', names=decoded_data_items)
            dataset_out = dataset_out.drop_duplicates()
            dataset_out.to_csv(
                conv_verif_results_path, header=False)
        except KeyError:
            pass



    def _check_model_shape(model_obs_match_info):
        """check if all the model shapes are the same"""
        for index, cur_info in model_obs_match_info.iterrows():
            if index == 0:
                nx = cur_info['model_nx']
                ny = cur_info['model_ny']
            else:
                if (nx != cur_info['model_nx']) or (ny != cur_info['model_ny']):
                    raise Exception('model shape is not consistent')
            return nx, ny

    # --------------------------------------------------
    # program starts
    # --------------------------------------------------
    # step 0: create the verification directory
    verif_results_dir = os.path.join(work_dir, 'verif_results')
    if not os.path.exists(verif_results_dir):
        os.makedirs(verif_results_dir)

    # step 1: locate the model out information
    model_out_info = _get_model_out_info(
        model_type, model_name, work_dir)

    # step 2: get the model and obs match info
    model_obs_match_info = _get_model_obs_out_info(
        work_dir, obs_from, obs_to, model_type, model_name)

    # step 3: get the model shape from the model-obs match info
    #         * since we get the model lat/lon from model i/j,
    #           the model shape must be consistent
    nx, ny = _check_model_shape(model_obs_match_info)
    
    # step 4: return the elements from required task
    required_model_name, required_datetime_analysis_hour, \
        required_datetime_cold_hour, required_datetime_lbc_hour, \
        required_conv_field_list = _extract_the_verification_task_info(
        conv_conv_verification_task)

    # step 5: attach hour to the model_info out
    model_out_info = \
        _attach_hour_info_to_model_dataset(model_out_info)

    # step 6: add condition on the model_out_info with:
    #         * model_name
    #         * datetime_analysis_hour
    #         * datetime_cold_hour
    #         * datetime_lbc_hour
    #         * conv field list
    required_model_out = \
        _extract_model_out_with_condition(
            model_out_info, required_model_name, 
            required_datetime_analysis_hour,
            required_datetime_cold_hour, 
            required_datetime_lbc_hour)

    # step 7: get the observation path
    conv_obs_path = os.path.join(
        work_dir, 'conv', 
        'conv_obs_from_{}_to_{}.csv'.format(
            obs_from.strftime('%Y%m%d%H'), 
            obs_to.strftime('%Y%m%d%H')))
       
    # step 8: start verifications by:
    #         going through all the rows from model out 
    #         (e.g., datetime_valid by datetime_valid)
    conv_verif_out = {}
    for index, cur_model_out in required_model_out.iterrows():
        cur_model_path = cur_model_out['local_path']
        cur_datetime_valid = cur_model_out['datetime_valid']

        for conv_verif_field in required_conv_field_list:
            # step 8.1: get the model data based on the previous 
            #           model size (nx and ny)
            if model_type == 'gfs':
                cur_model_lat, cur_model_lon, cur_model_data = read_gfs(
                    cur_model_path, conv_verif_field, 
                    None, nx=int(nx), ny=int(ny))
            elif model_type == 'ifs':
                cur_model_lat, cur_model_lon, cur_model_data = read_ifs(
                    cur_model_path, conv_verif_field, 
                    None, nx=int(nx), ny=int(ny))
            elif model_type == 'wrf':
                cur_model_lat, cur_model_lon, cur_model_data = read_wrf(
                    cur_model_path, conv_verif_field, 
                    None)

            # step 8.2: get the observations
            cur_conv_obs = read_conv_obs(
                    conv_obs_path, cur_datetime_valid, 
                    CONV_MODEL_OBS_NAME_MATCH[conv_verif_field])

            # step 8,3: get match pair
            conv_verif_out = _get_match_pairs(
                    conv_verif_out,
                    cur_model_lat, cur_model_lon, 
                    cur_model_data, cur_conv_obs, 
                    model_obs_match_info,
                    conv_verif_field,
                    model_type, model_name,
                    cur_model_out['datetime_valid'],
                    cur_model_out['datetime_analysis'],
                    cur_model_out['datetime_lbc'],
                    cur_model_out['datetime_valid_hour'],
                    cur_model_out['datetime_analysis_hour'],
                    cur_model_out['datetime_lbc_hour'])


    # step 9: write verification results
    _write_verif_results(
                verif_results_dir, conv_verif_out)





    
 
