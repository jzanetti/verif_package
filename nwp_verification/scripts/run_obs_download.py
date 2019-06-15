'''main script for radar download and decoding'''
import argparse
from datetime import datetime
import logging
import os
from nwp_verification import obs_download


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
    parser.add_argument('--obs_start', type=valid_datetime, required=True,
                        help='NWP analysis time as yyyymmddhhmm UTC')
    parser.add_argument('--obs_end', type=valid_datetime, required=True,
                        help='NWP analysis time as yyyymmddhhmm UTC')
    parser.add_argument('--work_dir', type=valid_datetime, required=True,
                        help='work_dir')

    return parser.parse_args()


def main_radar():
    args = setup_parser()
    
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    obs_download.download_little_r(
        args.status, args.region, 
        args.obs_start.strftime('%Y%m%d%H%M'), args.obs_end.strftime('%Y%m%d%H%M'), 
        args.work_dir)
    
    print('done')


if __name__ == '__main__':
    main_radar()
