CONV_VERIFICATION_TASKS = {
    'task_ifs1': {
        'model_name': 'ifs',
        'datetime_analysis_hour': 12,
        'datetime_cold_hour': 12,
        'datetime_lbc_hour': 12,
        'conv_field_list': ['2 metre temperature',
                            '2 metre dewpoint temperature',
                            '10 metre speed']},
    'task_gfs1': {
        'model_name': 'gfs',
        'datetime_analysis_hour': 12,
        'datetime_cold_hour': 12,
        'datetime_lbc_hour': 12,
        'conv_field_list': ['2 metre temperature',
                            '2 metre dewpoint temperature',
                            '10 metre speed']},
    'task_wrf1': {
        'model_name': 'nz3kmN-NCEP-var',
        'datetime_analysis_hour': 21,
        'datetime_cold_hour': 12,
        'datetime_lbc_hour': None,
        'conv_field_list': ['T2',
                            'td2',
                            '10 metre speed']},
    }


CONV_MODEL_OBS_NAME_MATCH = {
    '2 metre temperature': 'obs_temperature',
    '2 metre dewpoint temperature': 'obs_dewpoint',
    '10 metre speed': 'obs_windspd',
    'T2': 'obs_temperature',
    'td2': 'obs_dewpoint'
    }

RAINFALL_VERIFICATION_TASKS = {
    'task_wrf1': {
        'model_name': 'nz3kmN-NCEP-var',
        'datetime_analysis_hour': 14,
        'datetime_cold_hour': 6,
        'datetime_lbc_hour': 6,
        'rainfall_threshold_list': [0.2, 1.0, 5.0]},
    'task_wrf2': {
        'model_name': 'nz3kmN-NCEP-var',
        'datetime_analysis_hour': 15,
        'datetime_cold_hour': 6,
        'datetime_lbc_hour': 6,
        'rainfall_threshold_list': [0.2, 1.0, 5.0]},
    'task_ifs1': {
        'model_name': 'ifs',
        'datetime_analysis_hour': 12,
        'datetime_cold_hour': 12,
        'datetime_lbc_hour': 12,
        'rainfall_threshold_list': [0.6, 3.0, 15.0]},
    'task_gfs1': {
        'model_name': 'gfs',
        'datetime_analysis_hour': 12,
        'datetime_cold_hour': 12,
        'datetime_lbc_hour': 12,
        'rainfall_threshold_list': [0.6, 3.0, 15.0]},
    }
