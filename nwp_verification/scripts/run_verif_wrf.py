'''main script for radar download and decoding'''
import argparse
from datetime import datetime, timedelta
import logging
import os
from nwp_verification import data_download_upload, obs_download, radar_download, verif
from nwp_verification import RAINFALL_VERIFICATION_TASKS, CONV_VERIFICATION_TASKS
LOGGER = logging.getLogger()


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
    parser.add_argument('--model_name',
                        help="model_name", required=False)
    parser.add_argument('--download_wrf', dest='download_wrf',
                        help='download wrf data from S3 ',
                        action='store_true')
    parser.add_argument('--download_obs', dest='download_obs',
                        help='download observations from DDB ',
                        action='store_true')
    parser.add_argument('--download_radar', dest='download_radar',
                        help='download radar from S3 ',
                        action='store_true')
    parser.add_argument('--run_verification', dest='run_verification',
                        help='run WRF verifications ',
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
    parser.add_argument('--wrf_cycle_flag_dir', 
                        dest='wrf_cycle_flag_dir',
                        help='wrf_cycle_flag_dir from S3')

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
                        default=[1, 3, 6, 12],
                        help='accumulation hours for rainfall verif')


    return parser.parse_args(['--work_dir', '/home/szhang/nwp_verification', 
                              '--model_name', 'nz3kmN-NCEP-var',
                              '--download_wrf',
                              '--download_obs',
                              '--download_radar',
                              '--run_verification',
                              '--verif_types', 'conv',
                              '--analysis_time_start', '2019060421',
                              '--analysis_time_end', '2019060421',
                              '--analysis_hour_interval', '1',
                              '--fcst_length', '1', 
                              '--role_name', 'amps-research-framework',
                              '--wrf_cycle_flag_dir', 's3://metservice-research-us-west-2/research/experiments/sijin/OnDemand/nz3kmN_exp/nz3kmN-NCEP-2019060412/cycle_flag'])

    # return parser.parse_args()


def main_radar():
    args = setup_parser()
    
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    if args.download_wrf:
        data_info = data_download_upload.download_model(args, model_type='wrf')
        data_download_upload.convert_unique_dict_to_pandas(
            args.work_dir, data_info, 'wrf', model_name=args.model_name)
    
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
            args.work_dir, args.analysis_time_start, args.analysis_time_end, 
            int(args.fcst_length))
    
    if args.run_verification:
        if 'rainfall' in args.verif_types:
            data_download_upload.write_model_rainfall_on_radar_grids(
                args.work_dir, 'wrf', accumulation_length=1, model_name=args.model_name)
            for task_name in RAINFALL_VERIFICATION_TASKS:
                for accumulation_hours in args.accumulation_hours_list:
                    verif.rainfall_verification(
                        args.work_dir, 
                        RAINFALL_VERIFICATION_TASKS[task_name], 
                        'wrf', accumulation_hours,
                        model_name=args.model_name)
        if 'conv' in args.verif_types:
            # create model coords on the observation space
            data_download_upload.match_model_to_conv_grids(
                args.work_dir, 'wrf',
                args.analysis_time_start, 
                args.analysis_time_end + timedelta(
                    seconds=3600*int(args.fcst_length)), 
                model_name=args.model_name)
            for task_name in CONV_VERIFICATION_TASKS:
                verif.conv_verification(
                    args.work_dir, 'wrf',
                    CONV_VERIFICATION_TASKS[task_name],
                    args.analysis_time_start,
                    args.analysis_time_end + timedelta(
                            seconds=3600*int(args.fcst_length)),
                    model_name=args.model_name)
    print('done')


if __name__ == '__main__':
    main_radar()
