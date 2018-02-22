import numpy as np
from keras.models import Input, Model
from .keras_classifier import KerasClassifier
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda, Concatenate, Flatten


class TensorialLogisticRegressionClassifier(KerasClassifier):
    '''
    This is an implementation of logistic regression, where observations are assumed to be matrices instead of vectors.
    Hence, instead of finding a weight vector, matrices containing projection vectors in their columns must be found.
    There are no constraints on the projection vectors. A matrix containing projection vectors is found for each of the
    two modes (column and row modes). All pair-wise interactions are allowed, where each pair consists of one projection
    vector from one mode and one projection vector from the other mode.

    A version of this method, where only interactions between the i'th vector from one mode and the i'th vector from
    the other mode is allowed, is described in Appendix A in
    http://www.jmlr.org/papers/volume8/dyrholm07a/dyrholm07a.pdf.

    The current implementation, allowing all interactions, is described in section 8.1.3 in
    http://orbit.dtu.dk/files/123933965/phd408_Frolich_L.pdf.
    '''

    def __init__(self):
        super().__init__(compile_kwargs={'optimizer': 'adam'}, n_classes=2, fit_kwargs={'epochs': 20, 'verbose': 0})

    @staticmethod
    def build_model(input_shape, output_shape, Ks=None):
        """
        Builds the model from the shapes. Must return a Keras model
        """

        n_modes = len(input_shape)

        if n_modes==2:
            I, J = input_shape
        else:
            raise ValueError("Only support for two modes in observations implemented")

        if Ks is None:
            Ks = 3

        if np.alen(Ks)==1:
            Ks = np.repeat(Ks, n_modes)
        elif np.alen(Ks)!=n_modes:
            raise ValueError("The length of Ks must be either one or equal to the number of modes in the data but was"
                             "{Ks}")
        elif not np.alen(np.unique(Ks))==1:
            raise ValueError("All entries in the vector Ks must be equal, as different numbers of factors for "
                             "different modes is not yet implemented.")

        xs = Input(shape=input_shape)
        h = ModeDot(mode=1)(xs)
        h = ModeDot(mode=2)(h)
        h = ElementWiseMult()(h)
        h = Flatten()(h)
        h = Lambda(lambda x: K.sum(x, axis=1))(h) #, output_shape=(1,)
        h = AddScalar()(h)
        h = Lambda(lambda x: x[:, np.newaxis])(h) #, output_shape=(None, 1)
        h = Activation('sigmoid')(h)
        hclas0 = Lambda(lambda x: 1-x)(h) # , output_shape=(None, 1)
        hs = Concatenate(axis=1)([hclas0, h])

        return Model(inputs=xs, outputs=hs)


class AddScalar(Layer):

    def __init__(self, **kwargs):
        super(AddScalar, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,),
                                      initializer='uniform',
                                      trainable=True)
        super(AddScalar, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x+self.kernel #.bias_add(x, bias=self.kernel)


class ElementWiseMult(Layer):

    def __init__(self, factors=[3, 3], **kwargs):
        self.factors = factors
        super(ElementWiseMult, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.factors[0], self.factors[1]),
                                      initializer='uniform',
                                      trainable=True)
        super(ElementWiseMult, self).build(input_shape)

    def call(self, x):
        return K.dot(x, self.kernel)


class ModeDot(Layer):

    def __init__(self, mode, factors=3, **kwargs):
        self.mode = mode
        self.factors = factors
        super(ModeDot, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[self.mode], self.factors),
                                      initializer='uniform',
                                      trainable=True)
        super(ModeDot, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return self.mode_dot(tensor=x, matrix_or_vector=K.transpose(self.kernel), mode=self.mode)

    def compute_output_shape(self, input_shape):
        new_shape = list(input_shape)
        new_shape[self.mode] = self.factors
        return tuple(new_shape)


    @staticmethod
    def unfold(tensor, mode):
        """
        Translated from tensorly, which assumes the mxnet or pytorch backend.
        https://github.com/tensorly/tensorly/blob/master/tensorly/base.py

        Returns the mode-`mode` unfolding of `tensor` with modes starting at `0`.

        Parameters
        ----------
        tensor : ndarray
        mode : int, default is 0
               indexing starts at 0, therefore mode is in ``range(0, tensor.ndim)``

        Returns
        -------
        ndarray
            unfolded_tensor of shape ``(tensor.shape[mode], -1)``
        """
        new_dimensions = np.array(range(0, len(tensor.shape)))
        new_dimensions[mode] = 0
        new_dimensions[0] = mode
        return K.reshape(x=K.permute_dimensions(tensor, pattern=new_dimensions), shape=(np.int(tensor.shape[mode]), -1))

    @staticmethod
    def fold(unfolded_tensor, mode, shape):
        """
        Translated from tensorly, which assumes the mxnet or pytorch backend.
        https://github.com/tensorly/tensorly/blob/master/tensorly/base.py

        Refolds the mode-`mode` unfolding into a tensor of shape `shape`

            In other words, refolds the n-mode unfolded tensor
            into the original tensor of the specified shape.

        Parameters
        ----------
        unfolded_tensor : ndarray
            unfolded tensor of shape ``(shape[mode], -1)``
        mode : int
            the mode of the unfolding
        shape : tuple
            shape of the original tensor before unfolding

        Returns
        -------
        ndarray
            folded_tensor of shape `shape`
        """
        full_shape = list(shape)
        mode_dim = full_shape[mode]

        full_shape[mode] = full_shape[0]
        full_shape[0] = mode_dim

        modes = np.array(range(0, len(full_shape)))
        modes[0] = mode
        modes[mode] = 0

        for ishape in range(0, len(full_shape)):
            try:
                full_shape[ishape] = np.int(full_shape[ishape])
            except TypeError:
                full_shape[ishape] = -1

        myreshaped = K.reshape(unfolded_tensor, full_shape)

        return K.permute_dimensions(myreshaped, pattern=modes)

    def mode_dot(self, tensor, matrix_or_vector, mode):
            """
            Translated from tensorly, which assumes the mxnet or pytorch backend.
            https://github.com/tensorly/tensorly/blob/master/tensorly/tenalg/n_mode_product.py

            n-mode product of a tensor by a matrix at the specified mode.
            Mathematically: :math:`\\text{tensor} \\times_{\\text{mode}} \\text{matrix or vector}`
            Parameters
            ----------
            tensor : ndarray
                tensor of shape ``(i_1, ..., i_k, ..., i_N)``
            matrix_or_vector : ndarray
                1D or 2D array of shape ``(J, i_k)`` or ``(i_k, )``
                matrix or vectors to which to n-mode multiply the tensor
            mode : int
            Returns
            -------
            ndarray
                `mode`-mode product of `tensor` by `matrix_or_vector`
                * of shape :math:`(i_1, ..., i_{k-1}, J, i_{k+1}, ..., i_N)` if matrix_or_vector is a matrix
                * of shape :math:`(i_1, ..., i_{k-1}, i_{k+1}, ..., i_N)` if matrix_or_vector is a vector
            """
            # the mode along which to fold might decrease if we take product with a vector
            fold_mode = mode
            new_shape = list(tensor.shape)

            if K.ndim(matrix_or_vector) == 2:  # Tensor times matrix
                # Test for the validity of the operation
                if matrix_or_vector.shape[1] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned in mode-{2} multiplication: {3} (mode {2}) != {4} (dim 1 of matrix)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[1]
                        ))
                new_shape[mode] = matrix_or_vector.shape[0]

            elif K.ndim(matrix_or_vector) == 1:  # Tensor times vector
                if matrix_or_vector.shape[0] != tensor.shape[mode]:
                    raise ValueError(
                        'shapes {0} and {1} not aligned for mode-{2} multiplication: {3} (mode {2}) != {4} (vector size)'.format(
                            tensor.shape, matrix_or_vector.shape, mode, tensor.shape[mode], matrix_or_vector.shape[0]
                        ))
                if len(new_shape) > 1:
                    new_shape.pop(mode)
                    fold_mode -= 1
                else:
                    new_shape = [1]

            else:
                raise ValueError('Can only take n_mode_product with a vector or a matrix.'
                                 'Provided array of dimension {} not in [1, 2].'.format(K.ndim(matrix_or_vector)))

            return self.fold(unfolded_tensor=K.dot(matrix_or_vector, self.unfold(tensor, mode)), mode=fold_mode, shape=new_shape)