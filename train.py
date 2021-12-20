from log import get_logger
from data import load_archive


def train(opts):
    logger = get_logger()
    logger.info(f'Loading dataset from "{opts["data"]["filename"]}".')
    dataset = load_archive(opts['data']['filename'], opts['data']['batch_size'],
                           opts['data']['shuffle_size'], opts['data'].get('num_parallel_calls', 10),
                           opts['data'].get('num_parallel_calls', 10), opts['data'].get('shuffle_seed', None))

    model_desc = '\n\t'.join([f'{k} = {opts["model_arch"]["k"]}' for k in opts['model_arch']])
    logger.info(f'Creating new triplet model with the configuration:\n\t{model_desc}')

    model

