#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:57:33 2025

@author: wouya
"""

import numpy as np
from abc import ABC, abstractmethod
import odl
from odl.contrib.tensorflow import as_tensorflow_layer
import tensorflow as tf


class ForwardModel(ABC):
    # Defining the forward operators used. For customization, create a subclass of forward_model, implementing
    # the abstract classes.
    name = 'abstract'

    def __init__(self, size):
        self.size = size

    @abstractmethod
    def get_image_size(self):
        # Returns the image size in the format (width, height)
        pass

    @abstractmethod
    def get_measurement_size(self):
        # Returns the measurement size in the format (width, height)
        pass

    # All inputs to the evaluation methods have the format [width, height, channels]
    @abstractmethod
    def forward_operator(self, image):
        # The forward operator
        pass

  
    @abstractmethod
    def inverse(self, measurement):
        # An approximate (possibly regularized) inverse of the forward operator.
        # Used as starting point and for training
        pass


    # Input in the form [batch, width, height, channels]
    @abstractmethod
    def tensorflow_operator(self, tensor):
        # The forward operator as tensorflow layer. Needed for evaluation during training
        pass


class CT(ForwardModel):
    # a model for computed tomography on image of size 64x64. Allows for one color channel only.

    name = 'Computed_Tomography'
    def __init__(self, size):
        super().__init__(size)
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')

        geometry = odl.tomo.parallel_beam_geometry(self.space, num_angles=30)
        op = odl.tomo.RayTransform(self.space, geometry)

        # Ensure operator has fixed operator norm for scale invariance
        opnorm = odl.power_method_opnorm(op)
        self.operator = (1 / opnorm) * op
        self.fbp = opnorm * odl.tomo.fbp_op(op)
        self.adjoint_operator = (1 / opnorm)*op.adjoint

        # Create tensorflow layer from odl operator
        #self.ray_transform = as_tensorflow_layer(self.operator, 'RayTransform')

    def get_image_size(self):
        return self.space.shape

    def get_measurement_size(self):
        return self.operator.range.shape

    def forward_operator(self, image):
        assert len(image.shape) == 3
        assert image.shape[-1] == 1
        ip = self.space.element(image[..., 0])
        result = np.expand_dims(self.operator(ip), axis=-1)
        return result

    def forward_operator_adjoint(self, measurement):
        assert len(measurement.shape) == 3
        assert measurement.shape[-1] == 1
        ip = self.operator.range.element(measurement[..., 0])
        result = np.expand_dims(self.adjoint_operator(ip), axis=-1)
        return result

    def inverse(self, measurement):
        assert len(measurement.shape) == 3
        assert measurement.shape[-1] == 1
        m = self.operator.range.element(measurement[..., 0])
        return np.expand_dims(self.fbp(m), axis=-1)

    def tensorflow_operator(self, tensor):
        return self.operator(tensor)

    def get_odl_operator(self):
        return self.operator


class Denoising(ForwardModel):
    name = 'Denoising'

    def __init__(self, size):
        super(Denoising, self).__init__(size)
        self.size = size
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')
        self.operator = odl.IdentityOperator(self.space)

    def get_image_size(self):
        return self.size

    def get_measurement_size(self):
        return self.size

    def forward_operator(self, image):
        return image

    def forward_operator_adjoint(self, measurement):
        return measurement

    def inverse(self, measurement):
        return measurement

    def tensorflow_operator(self, tensor):
        return tensor

    def get_odl_operator(self):
        return self.operators


def forward_convolution_fft(x, h):
    """
    Perform convolution via FFT on an RGB image.
    
    Args:
        x: Tensor of shape [H, W, 3] or [N, H, W, 3], float32
        h: PSF kernel of shape [kH, kW], float32
    
    Returns:
        y: Convolved image, same shape as x
    """
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    if x.shape.rank == 3:
        x = tf.expand_dims(x, 0)  # add batch dim: [1, H, W, 3]
    elif x.shape.rank != 4:
        raise ValueError("x must have shape [H,W,3] or [N,H,W,3]")

    N, H, W, C = x.shape

    # Pad kernel to image size (H x W)
    h = tf.expand_dims(h, axis=-1)  # [kH,kW,1]
    h_padded = tf.image.pad_to_bounding_box(h, 0, 0, H, W)  # [H,W,1]
    h_padded = tf.squeeze(h_padded, -1)  # [H,W]
    h_fft = tf.signal.fft2d(tf.cast(h_padded, tf.complex64))  # [H,W]

    # Prepare output
    y_channels = []

    # Process each channel independently
    for c in range(C):
        x_c = x[..., c]  # [N,H,W]
        x_c_fft = tf.signal.fft2d(tf.cast(x_c, tf.complex64))
        y_c_fft = x_c_fft * h_fft
        y_c = tf.math.real(tf.signal.ifft2d(y_c_fft))
        y_channels.append(y_c)

    # Stack channels back
    y = tf.stack(y_channels, axis=-1)  # [N,H,W,3]

    if y.shape[0] == 1:
        y = tf.squeeze(y, 0)  # remove batch dim if input was [H,W,3]

    return y
    
def tsvd_deconvolution_fft(y, h, alpha):
    """
    Perform truncated SVD deconvolution on an RGB image via FFT.
    
    Args:
        y: Blurred RGB image tensor, shape [H,W,3] or [N,H,W,3], float32
        h: PSF kernel tensor, shape [kH,kW], float32
        alpha: threshold for the singluar values, float
    
    Returns:
        x_rec: Recovered RGB image, same shape as y
    """

    y = tf.convert_to_tensor(y, dtype=tf.float32)

    if y.shape.rank == 3:
        y = tf.expand_dims(y, 0)  # add batch dim
    elif y.shape.rank != 4:
        raise ValueError("y must have shape [H,W,3] or [N,H,W,3]")

    N, H, W, C = y.shape

    # Pad PSF to image size
    h = tf.expand_dims(h, axis=-1)  # [kH, kW, 1]
    h_padded = tf.image.pad_to_bounding_box(h, 0, 0, H, W)  # [H, W, 1]
    h_padded = tf.squeeze(h_padded, -1)  # [H, W]
    H_hat = tf.signal.fft2d(tf.cast(h_padded, tf.complex64))  # [H, W]

    mask = tf.cast(tf.abs(H_hat) > alpha, tf.float32)

    x_channels = []

    for c in range(C):
        y_c = y[..., c]  # [N, H, W]
        y_c_fft = tf.signal.fft2d(tf.cast(y_c, tf.complex64))  # FFT of blurred channel
        
        # Truncated inverse filter: invert only frequencies with singular values >= threshold
        inv_filter = tf.where(mask > 0, 1.0 / (H_hat + 1e-12), tf.convert_to_tensor(0.0,dtype=tf.complex64))  # avoid div by zero

        x_c_fft = y_c_fft * inv_filter  # apply truncated inverse filter
        x_c = tf.math.real(tf.signal.ifft2d(x_c_fft))  # inverse FFT to get deblurred channel

        x_channels.append(x_c)

    # Stack channels back
    x_rec = tf.stack(x_channels, axis=-1)  # [N, H, W, 3]

    if x_rec.shape[0] == 1:
        x_rec = tf.squeeze(x_rec, 0)

    return x_rec


class Convolution(ForwardModel):
    name = "Gaussian"
    
    def __init__(self, size,alpha,kernel):
        super().__init__(size)
        self.size = size
        self.space = odl.uniform_discr([-64, -64], [64, 64], [self.size[0], self.size[1]],
                                       dtype='float32')

        self.alpha = alpha
        self.kernel = tf.convert_to_tensor(kernel, dtype=tf.float32)


    def get_image_size(self):
        return self.size

    def get_measurement_size(self):
        return self.size

    def forward_operator(self, image):
        return self.tensorflow_operator(image)

    def inverse(self, measurement):
        return tsvd_deconvolution_fft(measurement,self.kernel,self.alpha)

    def tensorflow_operator(self, tensor):
        return forward_convolution_fft(tensor, self.kernel)
