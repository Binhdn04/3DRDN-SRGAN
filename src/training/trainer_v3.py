# -*- coding: utf-8 -*-
import sys
import time
import signal
import datetime
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from src.utils.logger import log
from src.models.model_v3 import Model3DRDN
from src.utils.plotting import generate_images
from src.preprocessing.data_preprocessing import get_preprocessed_data, PATCH_SIZE


def signal_handler(sig, frame):
    stop_log = "The training process was stopped at {}".format(time.ctime())
    log(stop_log)
    m.save_models("stopping_epoch")
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def evaluation_loop(dataset, PATCH_SIZE):
    output_data = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float32')
    hr_data     = np.empty((0,PATCH_SIZE,PATCH_SIZE,PATCH_SIZE,1), 'float32')
    for lr_image, hr_image in dataset:        
        # Ensure input data is float32
        lr_image = tf.cast(lr_image, tf.float32)
        hr_image = tf.cast(hr_image, tf.float32)
        
        output = m.generator_g(lr_image, training=False).numpy()
        output_data = np.append(output_data, output, axis=0)
        hr_data     = np.append(hr_data, hr_image.numpy(), axis=0)
    
    # Ensure consistent dtypes for TensorFlow operations
    output_data = tf.squeeze(tf.cast(output_data, tf.float32))
    hr_data     = tf.squeeze(tf.cast(hr_data, tf.float32))
    
    mean_errors = []
    std_errors  = []
    
    # Compute metrics with consistent float32 dtype
    psnr        = tf.image.psnr(output_data, hr_data, 1.0)
    psnr        = psnr[psnr!=float("inf")]
    ssim        = tf.image.ssim(output_data, hr_data, 1.0)
    abs_err     = tf.math.abs(hr_data - output_data)
    
    for v in [psnr, ssim, abs_err]:
        mean_errors.append(round(tf.reduce_mean(v).numpy(), 3))
        std_errors.append(round(tf.math.reduce_std(v).numpy(), 3))
    return mean_errors, std_errors


def generate_random_image_slice(sample_image, PATCH_SIZE, str1, str2=""):
    comparison_image_lr, comparison_image_hr = sample_image
    
    # Ensure consistent dtypes
    comparison_image_lr = tf.cast(comparison_image_lr, tf.float32)
    comparison_image_hr = tf.cast(comparison_image_hr, tf.float32)
    
    comparison_image_lr = tf.expand_dims(comparison_image_lr, axis=0)
    prediction_image    = tf.squeeze(m.generator_g(comparison_image_lr, training=False)).numpy()
    comparison_image_lr = tf.squeeze(comparison_image_lr).numpy()
    comparison_image_hr = tf.squeeze(comparison_image_hr).numpy()
    generate_images(prediction_image, comparison_image_lr, comparison_image_hr, PATCH_SIZE, str1, str2)


def main_loop(NO_OF_RDB_BLOCKS, GROWTH_RATE, NO_OF_DENSE_LAYERS,
              LR_G, LR_D, EPOCHS, BATCH_SIZE,
              LAMBDA_ADV, LAMBDA_GRD_PEN, CRIT_ITER, TRAIN_ONLY,
              MODEL, RESIDUAL_SCALING, EPOCH_START):

    begin_log = (
        '\n### Began training {} at {} with parameters: '
        'Epochs={}, RDB Blocks={}, Growth Rate={}, Dense Layers={}, '
        'Batch Size={}, LR_G={}, LR_D={}, Lambda Adversarial Loss={}, Lambda Gradient Penalty={}, '
        'Critic iterations={}, Training only={}, Residual Scaling={},  Starting Epoch={}\n'
    ).format(
        MODEL, time.ctime(),
        EPOCHS, NO_OF_RDB_BLOCKS, GROWTH_RATE, NO_OF_DENSE_LAYERS,
        BATCH_SIZE, LR_G, LR_D, LAMBDA_ADV, LAMBDA_GRD_PEN,
        CRIT_ITER, TRAIN_ONLY, RESIDUAL_SCALING, EPOCH_START
    )

    
    log(begin_log)
    log("Mixed Precision Training is enabled")

    training_start = time.time()

    log("Setting up Data Pipeline")
    VALIDATION_BATCH_SIZE = 3
    train_dataset, N_TRAINING_DATA, valid_dataset, N_VALIDATION_DATA, test_dataset, N_TESTING_DATA = get_preprocessed_data(BATCH_SIZE, VALIDATION_BATCH_SIZE)
    pipeline_seconds = time.time() - training_start
    pipeline_t_log = "Pipeline took {} to set up".format(datetime.timedelta(seconds=pipeline_seconds))
    log(pipeline_t_log)
    
    nu_data_log = "Number of Training Data: {}, Number of Validation Data: {}, Number of Testing Data: {}".format(N_TRAINING_DATA, N_VALIDATION_DATA, N_TESTING_DATA)
    log(nu_data_log)

    valid_dataset = valid_dataset.repeat().as_numpy_iterator()
         
    global m
    m = Model3DRDN(
        PATCH_SIZE=PATCH_SIZE,
        NO_OF_RDB_BLOCKS=NO_OF_RDB_BLOCKS,
        GROWTH_RATE=GROWTH_RATE,
        NO_OF_DENSE_LAYERS=NO_OF_DENSE_LAYERS,
        BATCH_SIZE=BATCH_SIZE,
        LR_G=LR_G,
        LR_D=LR_D,
        LAMBDA_ADV=LAMBDA_ADV,
        LAMBDA_GRD_PEN=LAMBDA_GRD_PEN,
        MODEL=MODEL,
        CRIT_ITER=CRIT_ITER,
        TRAIN_ONLY=TRAIN_ONLY,
        RESIDUAL_SCALING=RESIDUAL_SCALING
    )
       
    #Initial Random Slice Image Generation
    valid_batch = [next(valid_dataset)]
    (va_psnr, va_ssim, va_error), (va_psnr_std, va_ssim_std, va_error_std) = evaluation_loop(valid_batch, PATCH_SIZE)
    sample_image = (valid_batch[0][0][0], valid_batch[0][1][0])
    generate_random_image_slice(sample_image, PATCH_SIZE, 'a_first_plot_{}'.format(EPOCH_START), str2="")

    evaluation_log = "Before training: MAE = "+str(va_error)+" ± "+str(va_error_std)+", PSNR = "+str(va_psnr)+" ± "+str(va_psnr_std)+", SSIM = "+str(va_ssim)+" ± "+str(va_ssim_std)
    log(evaluation_log)    

    for epoch in range(EPOCH_START, EPOCH_START+EPOCHS):

        log("Began epoch {} at {}".format(epoch, time.ctime()))

        epoch_start = time.time()
        
        #TRAINING
        try:
            for lr, hr in train_dataset:
                m.training(lr, hr, epoch)
        except tf.errors.ResourceExhaustedError:
            log("Encountered OOM Error at {} !".format(time.ctime()))
            m.save_models(epoch)
            return
                
        #Validation
        valid_batch = [next(valid_dataset)]
        (va_psnr, va_ssim, va_error), (va_psnr_std, va_ssim_std, va_error_std) = evaluation_loop(valid_batch, PATCH_SIZE)
        sample_image = (valid_batch[0][0][0], valid_batch[0][1][0])
        generate_random_image_slice(sample_image, PATCH_SIZE, "epoch_{}".format(epoch), str2=" Epoch: {}".format(epoch))

        with m.summary_writer_valid.as_default():
            tf.summary.scalar('Mean Absolute Error', va_error, step=epoch)
            tf.summary.scalar('PSNR', va_psnr, step=epoch)
            tf.summary.scalar('SSIM', va_ssim, step=epoch)
        
        #Epoch Logging
        log("Finished epoch {} at {}.".format(epoch, time.ctime()))
        epoch_seconds = time.time() - epoch_start
        epoch_t_log = "Epoch took {}".format(datetime.timedelta(seconds=epoch_seconds))
        log(epoch_t_log)
        evaluation_log = "After epoch: MAE = "+str(va_error)+" ± "+str(va_error_std)+", PSNR = "+str(va_psnr)+" ± "+str(va_psnr_std)+", SSIM = "+str(va_ssim)+" ± "+str(va_ssim_std)
        log(evaluation_log)

        if (epoch + 1) % 30 == 0:
            m.save_models(epoch)
            
    m.save_models("last_epoch")

    #Testing
    generate_random_image_slice(sample_image, PATCH_SIZE, 'z_testing_plot_{}'.format(EPOCH_START))
    (test_psnr, test_ssim, test_error), (test_psnr_std, test_ssim_std, test_error_std) = evaluation_loop(test_dataset, PATCH_SIZE)
        
    #Training Cycle Meta Data Logging
    evaluation_log = "After training: MAE = "+str(test_error)+" ± "+str(test_error_std)+", PSNR = "+str(test_psnr)+" ± "+str(test_psnr_std)+", SSIM = "+str(test_ssim)+" ± "+str(test_ssim_std)
    log(evaluation_log)
    log("Finished training at {}".format(time.ctime()))
    training_seconds = time.time() - training_start
    training_t_log = "Training took {}".format(datetime.timedelta(seconds=training_seconds))
    log(training_t_log)