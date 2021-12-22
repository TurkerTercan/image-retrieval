import argparse
import sys
import os
import tensorflow as tf

from utils import parse_config
from log import setup_logging, get_logger
from train import train


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config')
    parser.add_argument('log_file')
    return parser.parse_args()


if __name__ == '__main__':
    logger = None
    try:
        args = parse_args()
        config = parse_config(args.config)
        output_path = config['general']['output_path']
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        setup_logging(os.path.join(output_path, args.log_file))
        logger = get_logger()
        for name, step in [('train', train)]:
            opts = config.get(name, {'enabled': False})
            opts.update(config['general'])
            opts['other'] = dict()
            opts['other'].update(config)

            if opts['enabled']:
                logger.info(f'Perform step {name}')
                step(opts)
            else:
                logger.info(f'Skip step {name}')

    except ValueError as error:
        func = logger.error if logger else print
        func(f'An error occurred: {error}.')
    except KeyboardInterrupt:
        func = logger.error if logger else print
        func('The experiment has been manually cancelled')
        try:
            sys.exit(1)
        except SystemError:
            os._exit(1)
