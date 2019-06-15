import subprocess
import os
from AMPSAws import resources, utils, connections
from obs2aws import tools
import Queue
from OBS2r import args_opts, cli_obs2little_r, s3_download
from datetime import timedelta
import pandas
from glob import glob

NZ_AREA = [-49.2334480286, -31.6743965149, 161.043701172, -178.119750977]
CONV_DATA_TO_USE = ['metar', 'synop', 'nzaws'] # 'amvir nzaws metar amdar amvvis ascat synop ship amvwv buoy'
FIELDS_TO_GET_FROM_DDB = [
    'obs_id', 'report_type_geohash','valid_time',
    'latitude','longitude','pressure','pressureReducedToMeanSeaLevel',
    'airTemperature','dewPointTemperature','dewpointTemperature','windDirection','windSpeed',
    'windDirectionAt10M','windSpeedAt10M','elevation','flightLevel']
RINSTANCE = resources.running_on_ec2()


def get_resource(obs_from, obs_region):
    conv = tools.byteify(utils.get_conventions('research'))
    
    if RINSTANCE == False:
        keys_path = conv['research']['kelburn']['path_to_iam_keys']
        kelburn_table_name = conv['prod']['kelburn']["archived_observations_database_prefix"]
        table_name = kelburn_table_name.replace('<yyyy>', str(obs_from.year))
        resource = connections.get_resource('dynamodb', region_name = obs_region, 
                                                        status = 'research',
                                                        role_name='amps-research-gp',
                                                        keys_path=keys_path)
    else:
        research_table_name = conv['prod'][obs_region][
            "archived_observations_database_prefix"]
        table_name = research_table_name.replace('<yyyy>', str(
            obs_from.year))
        resource = connections.get_resource('dynamodb', region_name = 
            obs_region, status = 'research')
        
    return table_name, resource


def q2l(ddb_row_queue):
    obs_list = []
    while True:
        ddb_row = ddb_row_queue.get()
        ddb_row = s3_download.row_postprocess(ddb_row)
        if 'windDirection' in ddb_row.keys():
            if ddb_row['windDirection'] == 'Variable':
                ddb_row_queue.task_done()
                continue
        obs_list.append(ddb_row)
        ddb_row_queue.task_done()
        if ddb_row_queue.qsize() == 0:
            break
    
    return obs_list



def write_obs(obs_list, out_filename, output_format):

    if not output_format in ['ascii', 'csv']:
        raise Exception('output format {} is not '
                        'supported'.format(output_format))

    df = pandas.DataFrame(columns=['obs_id', 'obs_time', 'obs_lat',
                                   'obs_lon', 'obs_pressure', 
                                   'obs_pressure_reduced_to_msl',
                                   'obs_temperature', 'obs_dewpoint',
                                   'obs_winddir', 'obs_windspd'])
    for cobs in obs_list:
        cobs_id = cobs_lat = cobs_lon = cobs_p = cobs_pmsl = \
            cobs_t = cobs_td = cobs_dir = cobs_spd = cobs_ele = -88888.0
        if 'obs_id' in cobs.keys():
            cobs_id = cobs['obs_id']
        if 'valid_time' in cobs.keys():
            cobs_time = cobs['valid_time']
        if 'latitude' in cobs.keys():
            cobs_lat = cobs['latitude']
        if 'longitude' in cobs.keys():
            cobs_lon = cobs['longitude']            
        if 'pressure' in cobs.keys():
            cobs_p = cobs['pressure']                 
        if 'pressureReducedToMeanSeaLevel' in cobs.keys():
            cobs_pmsl = cobs['pressureReducedToMeanSeaLevel']
        if 'airTemperature' in cobs.keys():
            cobs_t = cobs['airTemperature']
        if 'dewpointTemperature' in cobs.keys():
            cobs_td = cobs['dewpointTemperature']
        if 'windDirection' in cobs.keys():
            cobs_dir = cobs['windDirection']
        if 'windSpeed' in cobs.keys():
            cobs_spd = cobs['windSpeed']
        # if 'obs_elevation' in cobs.keys():
        #    cobs_ele = cobs['obs_elevation']
        #if 'height' in cobs.keys():
        #    cobs_hgt = cobs['obs_height']

        cline = {'obs_id': cobs_id, 'obs_time': cobs_time, 'obs_lat': cobs_lat,
                     'obs_lon': cobs_lon, 'obs_pressure': cobs_p,
                     'obs_pressure_reduced_to_msl': cobs_pmsl,
                     'obs_temperature': cobs_t, 'obs_dewpoint': cobs_td,
                     'obs_winddir': cobs_dir, 'obs_windspd': cobs_spd}
        df = df.append(cline, ignore_index=True)
    
    if output_format == 'ascii':
        fout = open(out_filename + '.ascii', "w")
        fout.write(df.to_string())
        fout.close()
    elif output_format == 'csv':
        df.to_csv(path_or_buf=out_filename + '.csv')


def get_conv_obs(obs_from, obs_to, obs_region, work_dir):
    if not os.path.exists(os.path.join(work_dir, 'conv')):
        os.makedirs(os.path.join(work_dir, 'conv'))
    
    obs_filename = 'conv_obs_from_{}_to_{}'.format(
        obs_from.strftime('%Y%m%d%H'), obs_to.strftime('%Y%m%d%H'))
    if os.path.exists(os.path.join(
        work_dir, 'conv','{}.csv'.format(obs_filename))):
        return
    
    cur_obs = obs_from
    obs_list_out = []
    while cur_obs <= obs_to:
        cur_obs_start = cur_obs - timedelta(seconds=300)
        cur_obs_end = cur_obs + timedelta(seconds=300)
        table_name, resource = get_resource(cur_obs_start, obs_region)
        ddb_row_queue = Queue.Queue()
        cli_obs2little_r.read_ddb_rows(resource, ddb_row_queue, table_name,
                     FIELDS_TO_GET_FROM_DDB, CONV_DATA_TO_USE,
                     cur_obs_start, cur_obs_end, 
                     cur_obs_end + timedelta(seconds = 3*3600),
                     NZ_AREA[1], NZ_AREA[0], NZ_AREA[2], NZ_AREA[3])
    
        obs_list_out.extend(q2l(ddb_row_queue))
        ddb_row_queue.join()
        cur_obs += timedelta(seconds=3600.0)
        
    write_obs(obs_list_out, os.path.join(work_dir, 'conv', obs_filename), 'csv')

def get_unique_latlons_from_conv_obs(obs_from, obs_to, work_dir):
    conv_obs_latlon_filename = 'conv_obs_latlon_from_{}_to_{}.csv'.format(
        obs_from.strftime('%Y%m%d%H'), obs_to.strftime('%Y%m%d%H'))
    
    if os.path.exists(os.path.join(
            work_dir, 'conv', conv_obs_latlon_filename)):
        return

    obs_filename = 'conv_obs_from_{}_to_{}.csv'.format(
        obs_from.strftime('%Y%m%d%H'), obs_to.strftime('%Y%m%d%H'))
    conv_obs_path = os.path.join(work_dir, 'conv', obs_filename)
    conv_obs_data = pandas.read_csv(conv_obs_path, ',')
    conv_obs_data = conv_obs_data.round(
        {'obs_lat': 3, 'obs_lon': 3})
    conv_obs_info = conv_obs_data[['obs_id', 'obs_lat', 'obs_lon']
            ].drop_duplicates()
    conv_obs_info.to_csv(
        path_or_buf=os.path.join(
            work_dir, 'conv', conv_obs_latlon_filename))
    
    
    
    
    
