'''main script for gfs download and decoding'''
import argparse
from datetime import datetime, timedelta
import logging
import os
from nwp_verification import data_download_upload, obs_download, radar_download, verif
from nwp_verification import RAINFALL_VERIFICATION_TASKS, CONV_VERIFICATION_TASKS

LOGGER = logging.getLogger()

'''
return parser.parse_args(['--wrf_analysis_time', '2019041201', 
                          '--work_dir', 'winds_plot_test', 
                          '--model_name', 'nz3kmN-NCEP-var', 
                          '--wrf_key_on_s3', 's3://metservice-research-us-west-2/research/experiments/sijin/OnDemand/nz3kmN_exp/nz3km_no_velocity_no_vert_no_ref_adjust21/keys/nz3kmN-NCEP-var/2019041201/a83eaf53-3945-4a1d-86d5-38e3e3eda64e.yaml',
                          '--fcst_length', '1', 
                          # '--download_wrf',
                          '--role_name', 'amps-research-framework'])
'''

def valid_datetime(timestamp):
    '''turn a timestamp into a datetime object'''
    try:
        return datetime.strptime(timestamp, "%Y%m%d%H")
    except ValueError:
        msg = "Not a valid date: '{}'.".format(timestamp)
        raise argparse.ArgumentTypeError(msg)


def setup_parser():
    parser = argparse.ArgumentParser(description='NWP verifications')
    parser.add_argument('--work_dir',
                        help="work_dir", required=False)
    parser.add_argument('--download_gfs', dest='download_gfs',
                        help='download gfs data from S3 ',
                        action='store_true')
    parser.add_argument('--download_obs', dest='download_obs',
                        help='download observations from DDB ',
                        action='store_true')
    parser.add_argument('--download_radar', dest='download_radar',
                        help='download radar data from S3 ',
                        action='store_true')
    parser.add_argument('--run_verification', dest='run_verification',
                        help='run_verification for GFS ',
                        action='store_true')
    parser.add_argument('--analysis_time_start', type=valid_datetime,
                        help="analysis_time_start", required=False)
    parser.add_argument('--analysis_time_end', type=valid_datetime,
                        help="analysis_time_end", required=False)
    parser.add_argument('--analysis_hour_interval',
                        help="analysis_hour_interval", required=False)
    parser.add_argument('--fcst_length',
                        help="fcst_length", required=False)

    # ------------------------------------------
    # optional (AWS configs)
    # ------------------------------------------
    parser.add_argument('-s', '--status',
                        help="[research], pilot or prod",
                        default='research')
    parser.add_argument('-r', '--region', default='us-west-2',
                        help="AWS region where the input data is. [us-west-2]")
    parser.add_argument('-rn', '--role_name', default=None,
                        help="role name, only used in local machine")


    # ------------------------------------------
    # optional (if download_wrf)
    # ------------------------------------------
    parser.add_argument('--gfs_dir_on_s3', dest='gfs_dir_on_s3',
                        help='gfs_dir from S3')

    # ------------------------------------------
    # optional (if run_verification)
    # ------------------------------------------
    parser.add_argument('--verif_types', nargs='+', required=False,
                        default=['rainfall', 'conv'],
                        help='verif types to be used')

    # ------------------------------------------
    # optional (if run_verification + rainfall)
    # ------------------------------------------
    parser.add_argument('--accumulation_hours_list', nargs='+', required=False,
                        default=[1, 3, 6],
                        help='accumulation hours for rainfall verif')

    return parser.parse_args(['--work_dir', '/home/szhang/nwp_verification', 
                              '--download_gfs',
                              '--download_obs',
                              '--download_radar',
                              '--run_verification',
                              '--verif_types', 'conv',
                              '--analysis_time_start', '2019060612',
                              '--analysis_time_end', '2019060612',
                              '--analysis_hour_interval', '6',
                              '--fcst_length', '3', 
                              '--role_name', 'amps-research-framework',
                              '--gfs_dir_on_s3', 's3://metservice-research-us-west-2-vault/incoming-external-data/gfs/det'])

    return parser.parse_args()


def main_radar():
    args = setup_parser()
    
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if args.download_gfs:
        data_info = data_download_upload.download_model(
            args, model_type='gfs')
        data_download_upload.convert_unique_dict_to_pandas(
            args.work_dir, data_info, 'gfs')

    if args.download_obs:
        obs_download.get_conv_obs(
            args.analysis_time_start, 
            args.analysis_time_end + timedelta(
                seconds=3600*int(args.fcst_length)), 
            args.region, args.work_dir)
        obs_download.get_unique_latlons_from_conv_obs(
            args.analysis_time_start, 
            args.analysis_time_end + timedelta(
                seconds=3600*int(args.fcst_length)),
            args.work_dir)

    if args.download_radar:
        radar_download.run_get_ref(
            args.work_dir, 
            args.analysis_time_start, 
            args.analysis_time_end, 
            int(args.fcst_length))

    if args.run_verification:
        if 'rainfall' in args.verif_types:
            data_download_upload.write_model_rainfall_on_radar_grids(
                args.work_dir, 'gfs', accumulation_length=1)
            for task_name in RAINFALL_VERIFICATION_TASKS:
                for accumulation_hours in args.accumulation_hours_list:
                    verif.rainfall_verification(
                        args.work_dir, 
                        RAINFALL_VERIFICATION_TASKS[task_name], 
                        'gfs', accumulation_hours)

        if 'conv' in args.verif_types:
            # create model coords on the observation space
            data_download_upload.match_model_to_conv_grids(
                args.work_dir, 'gfs',
                args.analysis_time_start, 
                args.analysis_time_end + timedelta(
                    seconds=3600*int(args.fcst_length)))
            for task_name in CONV_VERIFICATION_TASKS:
                verif.conv_verification(
                    args.work_dir, 'gfs',
                    CONV_VERIFICATION_TASKS[task_name],
                    args.analysis_time_start,
                    args.analysis_time_end + timedelta(
                            seconds=3600*int(args.fcst_length)))
    print('done')


if __name__ == '__main__':
    main_radar()
