import os
import tensorflow as tf
import numpy as np

from log import get_logger
from data import load_archive
from triplet_model import TripletModel
from triplet_loss import TripletLoss


def train(opts):
    logger = get_logger()
    logger.info(f'Loading dataset from "{opts["data"]["filename"]}".')
    dataset = load_archive(opts['data']['filename'], opts['data']['batch_size'],
                           opts['data']['shuffle_size'], opts['data'].get('num_parallel_calls', 10),
                           opts['data'].get('num_parallel_calls', 10), opts['data'].get('shuffle_seed', None))

    model_desc = '\n\t'.join([f'{k} = {opts["model_arch"][k]}' for k in opts['model_arch']])
    logger.info(f'Creating new triplet model with the configuration:\n\tmodel = {opts["model"]}\n\t{model_desc}')
    model_arch = opts['model_arch']
    model = TripletModel(opts['model'], **model_arch)
    loss = TripletLoss(opts['margin'], opts)

    loss.model = model
    loss.beta = opts['triplets'].get('beta', 0.0)
    learning_rate = opts['optimizer'].get('learning_rate_decay', None)
    if learning_rate and learning_rate['enabled']:
        del learning_rate['enabled']
        opts['optimizer']['args']['learning_rate'] = tf.optimizers.schedules.ExponentialDecay(**learning_rate)

    optimizer = tf.optimizers.get({
        'class_name': opts['optimizer']['name'],
        'config': opts['optimizer'].get('args', dict())
    })

    if isinstance(optimizer.learning_rate, tf.optimizers.schedules.LearningRateSchedule):
        def learning_rate(step): return optimizer.learning_rate(step)
    else:
        def learning_rate(step): return optimizer.learning_rate

    model.loss = loss
    model.custom_input_shape = dataset.element_spec[0]
    model.loss_metric = tf.metrics.Mean('train_loss', dtype=tf.float32)
    model.summary_writer = tf.summary.create_file_writer(os.path.join(opts['output_path'], 'training'))
    model.optimizer = optimizer
    model.learning_rate = learning_rate

    visualize = opts.get('visualize_triplets', False)
    evaluation_period = opts.get('evaluation_period', 0)
    batch_portion = opts.get('batch_portion', 1.0)

    data_size = opts['data']['size']
    batch_size = opts['data']['batch_size']

    num_batches = int(np.ceil(float(data_size) / batch_size))
    all_triplets = 0

    for epoch in range(opts['epoch']):
        logger.info(f'Epoch {epoch + 1:03d}')
        model.loss.current_epoch = epoch

        all_triplets = _train(epoch, model, dataset, num_batches, batch_portion, all_triplets, visualize)

    model.summary_writer.close()


def _train(epoch, model, dataset, num_batches, batch_portion, all_triplets, visualize=False):
    logger = get_logger()
    loss = model.loss

    progress_bar = tf.keras.utils.Progbar(num_batches, unit_name='batch')
    progress_bar.update(0)

    num_triplets = 0

    for batch_index, batch in enumerate(dataset):
        images = batch[0]
        labels = batch[1]['label']

        loss.image_batch = images
        loss.batch_index = batch_index

        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss_value = loss(labels, logits) + sum(model.losses)
            model.loss_metric(loss_value)

        num_triplets += int(loss.num_last_positive_triplets)

        gradients = tape.gradient(loss_value, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        progress_bar.add(1)

    logger.info(f'Current loss: {model.loss_metric.result():0.6f}')
    logger.info(f'Sampled triplets: {num_triplets} new, {all_triplets + num_triplets} total')

    return num_triplets
