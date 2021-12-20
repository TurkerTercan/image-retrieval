import os

import yaml

UCM_LABEL_INDICES = {
    'agricultural': 0,
    'airplane': 1,
    'baseballdiamond': 2,
    'beach': 3,
    'buildings': 4,
    'chaparral': 5,
    'denseresidential': 6,
    'forest': 7,
    'freeway': 8,
    'golfcourse': 9,
    'harbor': 10,
    'intersection': 11,
    'mediumresidential': 12,
    'mobilehomepark': 13,
    'overpass': 14,
    'parkinglot': 15,
    'river': 16,
    'runway': 17,
    'sparseresidential': 18,
    'storagetanks': 19,
    'tenniscourt': 20
}

UCM_LABEL_NAMES = {UCM_LABEL_INDICES[name]: name for name in UCM_LABEL_INDICES}


def parse_config(filename):
    if os.path.exists(filename):
        try:
            if filename.endswith('.yaml'):
                with open(filename, 'r') as file_handle:
                    return yaml.load(file_handle, Loader=yaml.FullLoader)
            else:
                raise ValueError(f'The type of the configuration file '
                                 f'could not be determined from the extension "{filename}".')
        except IOError as error:
            raise ValueError(f'The config file could not be loaded. Exception: "{error}".')