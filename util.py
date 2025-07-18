#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 27 13:26:54 2025

@author: wouya
"""

import os
import tensorflow as tf
import numpy as np
import fnmatch
# from skimage.measure import compare_ssim as ssim

# def quality(truth, recon):
#     # for fixed images truth and reconstruction, evaluates average l2 value and ssim score
#     recon = cut_image(recon)
#     l2 = np.average(np.sqrt(np.sum(np.square(truth - recon), axis = (1,2,3))))
#     psnr = - 10 * np.log10(np.average(np.square(truth - recon)))
#     amount_images = truth.shape[0]
#     ssi = 0
#     for k in range(amount_images):
#         ssi = ssi + ssim(truth[k,...,0], cut_image(recon[k,...,0]))
#     ssi = ssi/amount_images
#     return [l2, psnr, ssi]

def cut_image(pic):
    # hard cut image to [0,1]
    pic = np.maximum(pic, 0.0)
    pic = np.minimum(pic, 1.0)
    return pic

def normalize_image(pic):
    # normalizes image to average 0 and variance 1
    av = np.average(pic)
    pic = pic - av
    sigma = np.sqrt(np.average(np.square(pic)))
    pic = pic/(sigma + 1e-8)
    return pic

def scale_to_unit_intervall(pic):
    # scales image to unit interval
    min = np.amin(pic)
    pic = pic - min
    max = np.amax(pic)
    pic = pic/(max+ 1e-8)
    return pic

def create_single_folder(folder):
    # creates folder and catches error if it exists already
    if not os.path.exists(folder):
        try:
            os.makedirs(folder)
        except OSError:
            pass

def find(pattern, path):
    # finds all files with defined pattern in path and all of its subfolders
    result = []
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(os.path.join(root, name).replace("\\", "/"))
    return result

def lrelu(x):
    # leaky rely
    return (tf.nn.relu(x) - 0.1*tf.nn.relu(-x))

def l2_norm(tensor):
    # l2 norm for a tensor in (batch, x, y, channel) format
    return np.mean(np.sqrt(np.sum(np.square(tensor), axis=(1,2,3))))

def dilated_conv_layer(inputs, name, filters=16, kernel_size=(5, 5), padding="same", rate = 1,
                                  activation=lrelu, reuse=False):
    # # a dilated convolutional layer
    # inputs_dim = inputs.get_shape().as_list()
    # input_channels = inputs_dim[3]
    # with tf.variable_scope(name, reuse=reuse):
    #     weights = tf.get_variable(name='weights', shape=[kernel_size[0], kernel_size[1], input_channels, filters],
    #                         initializer=tf.contrib.layers.xavier_initializer())
    #     bias = tf.get_variable(name='bias', shape=[1, 1, 1, filters],
    #                         initializer=tf.zeros_initializer)
    # conv = tf.nn.atrous_conv2d(inputs, weights, rate = rate, padding=padding)
    # output = activation(tf.add(conv,bias))
    # return output
    pass

def image_l1(inputs):
    # contracts an image tensor of shape [batch, size, size, channels] to its l1 values along the size dimensions
    return tf.reduce_mean(tf.abs(inputs), axis = (1,2))


def lipschitz_ratio(critic, x, y):
    """
    Computes the Lipschitz ratio |f(x) - f(y)| / ||x - y|| for each pair (x_i, y_i).
    
    Args:
        critic: A tf.keras.Model or callable that takes images and returns scalar outputs.
        x: Tensor of shape (batch_size, 128, 128, 1) — first set of grayscale images.
        y: Tensor of shape (batch_size, 128, 128, 1) — second set of grayscale images.
    
    Returns:
        Tensor of shape (batch_size,) containing Lipschitz ratios.
    """

    fx = critic(x)
    fy = critic(y)

    # Ensure outputs are scalar per sample
    fx = tf.reshape(fx, [-1])
    fy = tf.reshape(fy, [-1])
    
    # Compute numerator |f(x) - f(y)|
    output_diff = tf.abs(fx - fy)

    # Compute denominator ||x - y||_2 (flattened)
    input_diff = tf.reshape(x - y, [x.shape[0], -1])
    input_norm = tf.norm(input_diff, axis=1) + 1e-12  # avoid division by zero

    # Compute the ratio
    lipschitz_ratios = output_diff / input_norm
    return lipschitz_ratios

def gradient_norm_on_interpolates(critic, real_images, recon_images):
    """
    Computes the gradient norm ||∇_x Psi(x)|| on interpolated samples between real and reconstructed images.

    Args:
        critic: A tf.keras.Model or callable.
        real_images: Tensor of shape (batch_size, 128, 128, 1).
        recon_images: Tensor of shape (batch_size, 128, 128, 1).
    
    Returns:
        Tensor of shape (batch_size,) with gradient norms.
    """
    # Sample epsilon uniformly for interpolation
    epsilon = tf.random.uniform(shape=[real_images.shape[0], 1, 1, 1], minval=0.0, maxval=1.0)
    interpolated = epsilon * real_images + (1 - epsilon) * recon_images
    interpolated = tf.Variable(interpolated)  # required for gradient tracking

    with tf.GradientTape() as tape:
        tape.watch(interpolated)
        predictions = critic(interpolated)
    
    # Compute gradients of critic output w.r.t. input
    gradients = tape.gradient(predictions, interpolated)
    gradients = tf.reshape(gradients, [gradients.shape[0], -1])
    grad_norms = tf.norm(gradients, axis=1)

    return grad_norms


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.

    Args:
        size: int, kernel size (must be odd: e.g., 3, 5, 7, …)
        sigma: float, standard deviation of Gaussian

    Returns:
        kernel: (size,size) ndarray, normalized to sum to 1
    """
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
    kernel = kernel / np.sum(kernel)
    return kernel