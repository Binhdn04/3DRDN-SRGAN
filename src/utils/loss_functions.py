import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
mae = tf.keras.losses.MeanAbsoluteError()

def supervised_loss(real, fake):
#     loss = mse(real, fake) # L2 Loss
    loss = mae(real, fake)  # L1 Loss
    return loss

def generator_loss(fake):
    loss = -tf.reduce_mean(fake)
    return tf.dtypes.cast(loss, tf.float64)

def discriminator_loss(real, fake):
    loss = tf.reduce_mean(fake) - tf.reduce_mean(real)
    return tf.dtypes.cast(loss, tf.float64)

def cycle_loss(real, cycled):
    loss = tf.reduce_mean(tf.abs(real - cycled))
    return tf.dtypes.cast(loss, tf.float64)

def identity_loss(real, same):
    loss = tf.reduce_mean(tf.abs(real - same))
    return tf.dtypes.cast(loss, tf.float64)
# For Mixed Precision Training
def psnr_and_ssim_loss(im1, im2):
    im1 = tf.cast(im1, tf.float32)
    im2 = tf.cast(im2, tf.float32)
    psnr = tf.image.psnr(im1, im2, max_val=1.0).numpy()
    ssim = tf.image.ssim(im1, im2, max_val=1.0).numpy()
    psnr = round(psnr, 3)
    ssim = round(ssim, 3)
    return psnr, ssim
"""
def psnr_and_ssim_loss(im1, im2):
    psnr = tf.image.psnr(im1, im2, 1).numpy()
    ssim = tf.image.ssim(im1, im2, 1).numpy()
    psnr = round(psnr,3)
    ssim = round(ssim,3)
    return psnr,ssim
"""