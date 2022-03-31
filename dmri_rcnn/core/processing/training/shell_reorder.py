'''Shell Reorder Class'''

import tensorflow as tf

from .base import DatasetMapper
from .utils import spherical_distances, update_1d_array


class ShellReorder(DatasetMapper):
    '''Reorders Shells using tf.Graph implementation for intra-shell datasets'''

    def __init__(self, in_num, out_num, random_seed=True):
        '''Intra-Shell subset reorder dataset mapping

        Args:
            in_num (int): Number of qspace vols to use as input per
            out_num (int): Number of qspace vols to use as output
            random_seed (bool): use a random choice to select initial qspace vol per set.
                If `False`: select first available index.

        Call Arguments:
            dataset (tf.data.Dataset): dataset object with the following structure per example:
                {
                    'mask': (tf.Tensor) shape -> (i, j, k),
                    'data_use': {
                        `shell`: {
                            'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32,
                            'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32,
                            'bval': (tf.Tensor) shape -> (fs,) dtype -> tf.float32,
                        }
                    },
                }

        Output Spec:
            a tf.data.Dataset object with the same structure as input, with different ordering
                of qspace dimension for every entry.
        '''
        self.q_in = tf.constant(in_num)
        self.q_out = tf.constant(out_num)
        self.subset_size = tf.constant(in_num + out_num)
        self.random_seed = tf.constant(random_seed)

    @tf.function
    def apply(self, data):
        '''Applies algorithm to data

        Args:
            data (Dict[Any,Any]):
                'mask': (tf.Tensor) shape -> (i, j, k) dtype -> tf.int8
                'data_use': (Dict[int,Dict])
                    `shell`: (Dict[str,tf.tensor])
                        'dmri': (tf.Tensor) shape -> (i, j, k, fs) dtype -> tf.float32
                        'bvec': (tf.Tensor) shape -> (3, fs) dtype -> tf.float32
                        'bval': (tf.Tensor) shape -> (fs,) dtype -> tf.float32

        Returns:
            out_data (Dict[Any,Any]): Same entries as input, with different qspace ordering.
        '''
        out_data, data_use = {'mask': data['mask']}, {}
        for shell in data['data_use']:
            data_use[shell] = {}

            dmri = data['data_use'][shell]['dmri']
            bvec = data['data_use'][shell]['bvec']
            bval = data['data_use'][shell]['bval']

            order = self._get_order(bvec)

            data_use[shell]['dmri'] = tf.gather(dmri, order, axis=-1)
            data_use[shell]['bvec'] = tf.gather(bvec, order, axis=-1)
            data_use[shell]['bval'] = tf.gather(bval, order, axis=-1)

        out_data['data_use'] = data_use

        return out_data

    @tf.function
    def _get_order(self, bvec):
        '''Gets optimised qspace order given bvecs and subset size

        Args:
            bvec (tf.Tensor): shape -> (3, fs), dtype -> tf.float32

        Returns:
            order (tf.Tensor): Reordering array, shape -> (fs,)
                dtype -> tf.int32
        '''

        num_sets = tf.shape(bvec)[1] // self.subset_size  # Number of full sets

        # Instantiate order and order_mask, `order` is used to collect the indice order
        # of volumes, whilst `order_mask` is used to set the current number.
        order = tf.zeros(tf.shape(bvec)[1], dtype=tf.int32)
        order_mask = tf.zeros(tf.shape(bvec)[1], dtype=tf.bool)

        bvec_index = tf.range(tf.shape(bvec)[1])
        sph_dists = spherical_distances(bvec, bvec)  # -> (b, b)

        for _ in tf.range(num_sets):

            # Input array
            order, order_mask = self._get_subset_order(
                bvec_index, self.q_in, order, order_mask, sph_dists
            )

            # Output array
            order, order_mask = self._get_subset_order(
                bvec_index, self.q_out, order, order_mask, sph_dists
            )

        # Get remainder indices (these remain unchanged)
        order, order_mask = self._get_remainder_set(order, order_mask, bvec_index)

        return order

    @tf.function
    def _get_subset_order(self, bvec_index, subset_size, order, order_mask, sph_dists):
        '''Gets order of qspace volumes for a given subset.

        Args:
            bvec_index (tf.Tensor): Indices of qspace volumes.
                shape -> (fs,), dtype -> tf.int32
            subset_size (int): Size of subset to group into.
            order (tf.Tensor): Currently assigned reordered indices.
                Initially all zeros. shape -> (fs,), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of currently assigned indices
                within `order`. `True` entries = assigned.
                shape -> (fs,), dtype -> tf.bool
            sph_dists (tf.Tensor): Spherical distances matrix.
                shape -> (fs, fs), dtype -> tf.float32

        Returns:
            order (tf.Tensor): Currently assigned reordered indices.
                shape -> (fs,), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of currently assigned indices
                within `order`. `True` entries = assigned.
                shape -> (fs,), dtype -> tf.bool
        '''
        # Mask previously selected co-ordinates
        avail_mask = self._get_avail_mask(order, order_mask)

        # Instantiate subset pick
        subset_pick = tf.zeros(subset_size, dtype=tf.int32)

        # Set initial point
        start = self._get_starting_point(bvec_index, avail_mask)
        subset_pick = update_1d_array(
            subset_pick, tf.expand_dims(start, axis=0), tf.constant([0])
        )

        # Instantiate pick mask
        pick_mask = self._get_pick_mask(tf.shape(bvec_index)[0], start)

        # Update avail_mask
        avail_mask = update_1d_array(
            avail_mask, tf.constant([False]), tf.expand_dims(start, axis=0)
        )

        for idx in tf.range(1, subset_size):
            pick = self._get_furthest_point(
                sph_dists, pick_mask, avail_mask, bvec_index
            )

            # Update pick_mask, avail mask, and subset_pick mask
            pick_mask = update_1d_array(
                pick_mask, tf.constant([1]), tf.expand_dims(pick, axis=0)
            )
            avail_mask = update_1d_array(
                avail_mask, tf.constant([False]), tf.expand_dims(pick, axis=0)
            )
            subset_pick = update_1d_array(
                subset_pick, tf.expand_dims(pick, axis=0), tf.expand_dims(idx, axis=0)
            )

        # Now update order and order_mask
        order, order_mask = self._update_order(
            order, order_mask, bvec_index, subset_pick
        )

        return order, order_mask

    @tf.function
    def _get_remainder_set(self, order, order_mask, bvec_index):
        '''Gets remaining b-vector indices and updates `order` and
            `order_mask`

        Args:
            order (tf.Tensor): Previously set qspace indices
                shape -> (fs, ), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of currently assigned indices
                within `order`. shape -> (fs,), dtype -> tf.bool
            bvec_index (tf.Tensor): Indices of qspace volumes.
                shape -> (fs,), dtype -> tf.int32

        Returns:
            order (tf.Tensor): Updated qspace indices
                shape -> (fs, ), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of updated assigned indices
                within `order`. shape -> (fs,), dtype -> tf.bool
        '''
        # Get taken index
        taken_index = tf.boolean_mask(order, order_mask)
        # Get remainder (even if none)
        index_mask = tf.ones_like(order, dtype=tf.int32)
        index_mask = update_1d_array(
            index_mask, tf.zeros_like(taken_index, dtype=tf.int32), taken_index
        )
        remainder = tf.boolean_mask(bvec_index, index_mask)

        # Get indices in order and order_mask to update
        order_index = tf.boolean_mask(bvec_index, tf.logical_not(order_mask))

        # Update order and order_mask
        order = update_1d_array(order, remainder, order_index)
        order_mask = update_1d_array(
            order_mask, tf.ones_like(remainder, dtype=tf.bool), order_index
        )

        return order, order_mask

    @tf.function
    def _get_avail_mask(self, order, order_mask):
        '''Gets mask of available qspace indices.

        Args:
            order (tf.Tensor): Previously set qspace indices
                shape -> (fs, ), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of currently assigned indices
                within `order`. `True` entries = assigned.
                shape -> (fs,), dtype -> tf.bool

        Returns:
            mask (tf.Tensor): Availible indices mask.
                shape -> (fs,), dtype -> tf.bool
        '''
        indices = tf.expand_dims(tf.boolean_mask(order, order_mask), axis=-1)
        updates = tf.ones(len(indices), dtype=tf.int32)
        shape = tf.expand_dims(tf.shape(order)[0], axis=0)
        mask = tf.scatter_nd(indices, updates, shape)
        mask = tf.logical_not(tf.cast(mask, tf.bool))

        return mask

    @tf.function
    def _get_starting_point(self, bvec_index, avail_mask):
        '''Gets starting point of subset order

        Args:
            bvec_index (tf.Tensor): Indices of qspace volumes.
                shape -> (fs,), dtype -> tf.int32
            avail_mask (tf.Tensor): Availible indices mask.
                shape -> (fs,), dtype -> tf.bool

        Returns:
            (tf.Tensor): Starting point, shape -> (), dtype -> tf.int32
        '''
        avail_index = tf.boolean_mask(bvec_index, avail_mask)
        if self.random_seed:
            return tf.random.shuffle(avail_index)[0]
        return avail_index[0]

    @tf.function
    def _get_pick_mask(self, length, start):
        '''Gets initial pick mask

        Args:
            length (tf.Tensor): Length of qspace dim,
                shape -> (), dtype -> tf.int32
            start (tf.Tensor): Starting point
                shape -> (), dtype -> tf.int32

        Returns:
            mask (tf.Tensor): initial pick mask, where
                the `start` entry is == 1, else 0.
                shape -> (length,), dtype -> tf.int32
        '''
        indices = tf.expand_dims(tf.expand_dims(start, axis=0), axis=0)
        updates = tf.constant([1])
        shape = tf.expand_dims(length, axis=0)
        mask = tf.scatter_nd(indices, updates, shape)

        return mask

    @tf.function
    def _get_furthest_point(self, sph_dists, pick_mask, avail_mask, bvec_index):
        '''Gets furthest qspace point from previously selected points in set.

        Args:
            sph_dists (tf.Tensor): Spherical distances matrix
                shape -> (fs, fs), dtype -> tf.float32
            pick_mask (tf.Tensor): picked entries encoded as 1, else 0.
                shape -> (length,), dtype -> tf.int32
            avail_mask (tf.Tensor): Availible indices mask.
                shape -> (fs,), dtype -> tf.bool
            bvec_index (tf.Tensor): Indices of qspace volumes.
                shape -> (fs,), dtype -> tf.int32

        Returns:
            (tf.Tensor) -> Index of furthest point.
                shape -> (), dtype -> tf.int32
        '''
        # Grab all points in current set
        selection = tf.boolean_mask(sph_dists, pick_mask, axis=0)

        # Reshape so bvec dim is outer most
        selection = tf.transpose(selection, perm=[1, 0])

        # get indice array for used points
        notavail_index = tf.boolean_mask(bvec_index, ~avail_mask)
        notavail_index = tf.expand_dims(notavail_index, axis=1)

        # get update array
        upd_array = tf.tile(
            tf.zeros_like(notavail_index, dtype=tf.float32), [1, tf.shape(selection)[1]]
        )

        # update selection
        selection = tf.tensor_scatter_nd_update(selection, notavail_index, upd_array)

        # Reduce 1st dimension to get smallest distance from any point
        selection = tf.reduce_min(selection, axis=1)

        # Get maximum distance
        dist = tf.reduce_max(selection)

        # Return index where this distance is
        return tf.cast(tf.where(dist == selection)[0, 0], tf.int32)

    @tf.function
    def _update_order(self, order, order_mask, bvec_index, pick):
        '''Updates `order` and `order_mask` tensors given picked indices.

        Args:
            order (tf.Tensor): Previously set qspace indices
                shape -> (fs, ), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of currently assigned indices
                within `order`. shape -> (fs,), dtype -> tf.bool
            bvec_index (tf.Tensor): Indices of qspace volumes.
                shape -> (fs,), dtype -> tf.int32
            spick (tf.Tensor): Picked entries of subset encoded as 1, else 0.
                shape -> (subset_size,), dtype -> tf.int32

        Returns:
            order (tf.Tensor): Updated qspace indices
                shape -> (fs, ), dtype -> tf.int32
            order_mask (tf.Tensor): Mask of updated assigned indices
                within `order`. shape -> (fs,), dtype -> tf.bool
        '''
        # Get unused order mask
        not_order_mask = tf.logical_not(order_mask)
        # Get next `subset_size` indexes from bvec_index
        index = tf.boolean_mask(bvec_index, not_order_mask)[0 : tf.shape(pick)[0]]
        # Update the masks with these values
        order = update_1d_array(order, pick, index)
        order_mask = update_1d_array(
            order_mask, tf.ones_like(pick, dtype=tf.bool), index
        )

        return order, order_mask
