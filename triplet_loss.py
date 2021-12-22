import tensorflow as tf
import numpy as np


class TripletSelection(object):
    def __init__(self, opts):
        super(object, self).__init__()

        self.anchor_selection = anchor_selection(opts)
        self.triplet_selection = triplet_selection(opts)

        self.visualize_triplets = opts.get('visualize', False)
        self.margin = opts['margin']

    def select(self, features, feature_distances, labels, label_distances, loss):
        batch_size = len(features)
        base_mask = select_distinct_triplets(batch_size)
        anchor_mask = self.anchor_selection(features, feature_distances, labels, label_distances)
        triplet_mask = self.triplet_selection(features, feature_distances, labels, label_distances)

        return tf.logical_and(base_mask, tf.logical_and(anchor_mask, triplet_mask))


def anchor_selection(opts):
    try:
        if opts['anchors']['selection'] == 'random':
            num_anchors = opts['anchors']['number']
            return lambda features, feature_distances, labels, label_distances: select_random_anchors(len(features),
                                                                                                      num_anchors)
        else:
            raise ValueError(f'No valid anchor selection method was given.')
    except Exception as error:
        raise ValueError(f'An error occurred during creation of the anchor selection function: {str(error)}')


def triplet_selection(opts):
    if opts['triplets']['selection'] == 'random':
        num_elements = opts['triplets']['num_elements']
        return lambda features, feature_distances, labels, label_distances: select_random_triplets(feature_distances,
                                                                                                   num_elements)
    else:
        raise ValueError(f'No valid triplet selection method was given.')


def select_distinct_triplets(batch_size):
    indices_equal = tf.cast(tf.eye(batch_size), tf.bool)
    indices_not_equal = tf.logical_not(indices_equal)

    i_not_j = tf.expand_dims(indices_not_equal, 2)
    i_not_k = tf.expand_dims(indices_not_equal, 1)
    j_not_k = tf.expand_dims(indices_not_equal, 0)

    return tf.logical_and(tf.logical_and(i_not_j, i_not_k), j_not_k)


def select_random_anchors(batch_size, num_anchors):
    anchor_indices = np.random.randint(low=0, high=batch_size, size=num_anchors)
    return create_anchor_mask(anchor_indices, batch_size)


def create_anchor_mask(indices, batch_size):
    zeros = np.zeros((batch_size, batch_size, batch_size))
    zeros[indices, :, :] = 1
    return tf.cast(tf.constant(zeros), tf.bool)


def select_random_triplets(feature_distances, num_elements):
    batch_size = len(feature_distances)

    base_mask = np.ndarray((batch_size, batch_size, batch_size), dtype=bool)
    base_mask[:, :, :] = True

    positive_mask = np.ndarray((batch_size, batch_size, batch_size), dtype=bool)
    positive_mask[:, :, :] = False

    negative_mask = np.ndarray((batch_size, batch_size, batch_size), dtype=bool)
    negative_mask[:, :, :] = False

    # will be fixed with feature sort
    for a in range(batch_size):
        pos_indices = []
        neg_indices = []
        pos_indices.append(a)
        neg_indices.append(a)

        positive_mask[a, np.random.choice(pos_indices, size=num_elements, replace=True), :] = True
        negative_mask[a, :, np.random.choice(neg_indices, size=num_elements, replace=True)] = True

    return tf.cast(tf.constant(np.logical_and(positive_mask, negative_mask)), tf.bool)


class TripletLoss(tf.losses.Loss):
    def __init__(self, margin, opts, **kwargs):
        super(TripletLoss, self).__init__(**kwargs)

        self.margin = margin
        self.selection_strategy = TripletSelection(opts)
        self.num_last_triplets = 0
        self.total_triplets = 0
        self.num_last_positive_triplets = 0

        self.last_triplet = None
        self.last_label_distances = None
        self.last_feature_distances = None
        self.last_loss = 0

        self.initial_training = True

    def call(self, y_true, y_pred):
        feature_distances = calculate_pairwise_feature_distances(y_pred)
        label_distances = calculate_pairwise_label_distances(y_true)
        anchor_positive_distances = tf.expand_dims(feature_distances, 2)
        anchor_negative_distances = tf.expand_dims(feature_distances, 1)

        triplet_loss = anchor_positive_distances - anchor_negative_distances + self.margin
        mask = self.selection_strategy.select(y_pred, feature_distances, y_true, label_distances, self)

        triplet_loss = tf.multiply(tf.cast(mask, tf.float32), triplet_loss)
        triplet_loss = tf.maximum(triplet_loss, 0.0)

        valid_triplets = tf.cast(mask, tf.float32)
        num_positive_triplets = tf.reduce_sum(valid_triplets)

        self.num_last_triplets = tf.reduce_sum(tf.cast(mask, tf.int32))
        self.num_last_positive_triplets = num_positive_triplets
        self.total_triplets += num_positive_triplets

        indices = np.where(mask)
        if len(indices[0]) > 0:
            anchor_index, positive_index, negative_index = indices[0][0], indices[1][0], indices[2][0]
            self.last_triplet = (anchor_index, positive_index, negative_index)
            self.last_label_distances = (
                label_distances[anchor_index, positive_index], label_distances[anchor_index, negative_index])

            feature_distance_anchor_positive = feature_distances[anchor_index, positive_index]
            feature_distance_anchor_negative = feature_distances[anchor_index, negative_index]

            self.last_feature_distances = (feature_distance_anchor_positive, feature_distance_anchor_negative)
            self.last_loss = np.amax(
                [0, feature_distance_anchor_positive - feature_distance_anchor_negative + self.margin])

            del feature_distances
            del label_distances
            del anchor_positive_distances
            del anchor_negative_distances
            del valid_triplets
            del mask

            return tf.reduce_sum(triplet_loss) / (num_positive_triplets + 1e-16)


def calculate_pairwise_feature_distances(features):
    dot_product = tf.matmul(features, tf.transpose(features))
    square_norm = tf.linalg.diag_part(dot_product)
    distances = tf.expand_dims(square_norm, 0) - 2.0 * dot_product + tf.expand_dims(square_norm, 1)
    distances = tf.maximum(distances, 0.0)

    mask = tf.cast(tf.equal(distances, 0.0), tf.float32)
    distances = distances + mask * 1e-16
    distances = tf.sqrt(distances)
    distances = distances * (1.0 - mask)

    # maybe something wrong
    return distances * (1.0 - mask)


def calculate_pairwise_label_distances(labels):
    y_y = [x[1][0] for x in enumerate(labels.numpy())]
    a = np.ndarray((len(y_y), len(y_y)))
    x_i, y_i = 0, 0
    for x in y_y:
        y_i = 0
        for y in y_y:
            a[x_i][y_i] = abs(x - y) / 21
            y_i += 1
        x_i += 1
    return tf.convert_to_tensor(a, np.float32)
