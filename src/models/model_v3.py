import tensorflow as tf
from src.utils.logger import log
from src.utils.loss_functions import supervised_loss, generator_loss, discriminator_loss

# Enable Mixed Precision Training
tf.keras.mixed_precision.set_global_policy('mixed_float16')


class Generator:
    
    WEIGHTS_INIT_DICT    = {"HeUniform"     : tf.keras.initializers.HeUniform(),
                            "HeNormal"      : tf.keras.initializers.HeNormal(), 
                            "GlorotUniform" : tf.keras.initializers.GlorotUniform(),
                            "GlorotNormal"  : tf.keras.initializers.GlorotNormal(),}

    ACTIVATION_FUNC_DICT = {"SWISH"         : tf.keras.layers.Activation(tf.nn.silu),
                            "LEAKYRELU"     : tf.keras.layers.LeakyReLU(alpha=0.2),
                            "RELU"          : tf.keras.layers.ReLU(),
                            "ELU"           : tf.keras.layers.ELU(),}
    
    def __init__(self, PATCH_SIZE=40, NO_OF_RDB_BLOCKS=8, GROWTH_RATE=16, NO_OF_DENSE_LAYERS=4,
                 UTILIZE_BIAS=True, WEIGHTS_INIT="HeUniform", ACTIVATION_FUNC="LEAKYRELU", 
                 RESIDUAL_SCALING=0.2):
        self.GROWTH_RATE           = GROWTH_RATE
        self.PATCH_SIZE            = PATCH_SIZE
        self.NO_OF_RDB_BLOCKS      = NO_OF_RDB_BLOCKS
        self.NO_OF_DENSE_LAYERS    = NO_OF_DENSE_LAYERS
        self.UTILIZE_BIAS          = UTILIZE_BIAS
        self.RESIDUAL_SCALING      = RESIDUAL_SCALING
        self.WEIGHTS_INIT          = self.WEIGHTS_INIT_DICT[WEIGHTS_INIT]
        self.ACTIVATION_FUNC       = self.ACTIVATION_FUNC_DICT[ACTIVATION_FUNC]

        self.filter_size           = 3
        self.feature_channels      = 32

    def dense_layer(self, inputs, growth_rate, name=None):
        """Single dense layer with ReLU activation"""
        x = self.ACTIVATION_FUNC(inputs)
        x = tf.keras.layers.Conv3D(growth_rate, self.filter_size, 
                                  kernel_initializer=self.WEIGHTS_INIT,
                                  use_bias=self.UTILIZE_BIAS, 
                                  padding='same', name=name)(x)
        return x

    def residual_dense_block(self, inputs, block_idx):
        """Residual Dense Block (RDB)"""
        identity = inputs
        
        conv_outputs = [inputs]
        
        for i in range(self.NO_OF_DENSE_LAYERS):
            concat_input = tf.keras.layers.Concatenate()(conv_outputs)
            conv_out = self.dense_layer(concat_input, self.GROWTH_RATE, 
                                      name=f'RDB_{block_idx}_dense_{i}')
            conv_outputs.append(conv_out)
        concat_all = tf.keras.layers.Concatenate()(conv_outputs[1:])  
        local_fusion = tf.keras.layers.Conv3D(self.feature_channels, 1,
                                            kernel_initializer=self.WEIGHTS_INIT,
                                            use_bias=self.UTILIZE_BIAS,
                                            padding='same', 
                                            name=f'RDB_{block_idx}_local_fusion')(concat_all)
        
        scaled_fusion = tf.keras.layers.Lambda(lambda x: x * self.RESIDUAL_SCALING)(local_fusion)
        output = tf.keras.layers.Add()([identity, scaled_fusion])
        
        return output

    def create_generator(self):
        inputs = tf.keras.layers.Input(shape=[self.PATCH_SIZE, self.PATCH_SIZE, self.PATCH_SIZE, 1])
        
        shallow_features = tf.keras.layers.Conv3D(self.feature_channels, self.filter_size,
                                                 kernel_initializer=self.WEIGHTS_INIT,
                                                 use_bias=self.UTILIZE_BIAS,
                                                 padding='same', name='shallow_conv')(inputs)
        
        global_residual = shallow_features
        rdb_output = shallow_features
        rdb_outputs = []
        
        for i in range(self.NO_OF_RDB_BLOCKS):
            rdb_output = self.residual_dense_block(rdb_output, i)
            rdb_outputs.append(rdb_output)
        
        # Global feature fusion
        concat_rdbs = tf.keras.layers.Concatenate()(rdb_outputs)
        
        global_fusion_1 = tf.keras.layers.Conv3D(self.feature_channels, 1,
                                                kernel_initializer=self.WEIGHTS_INIT,
                                                use_bias=self.UTILIZE_BIAS,
                                                padding='same', name='global_fusion_1')(concat_rdbs)
        
        global_fusion_2 = tf.keras.layers.Conv3D(self.feature_channels, self.filter_size,
                                                kernel_initializer=self.WEIGHTS_INIT,
                                                use_bias=self.UTILIZE_BIAS,
                                                padding='same', name='global_fusion_2')(global_fusion_1)
        
        # Global residual connection
        deep_features = tf.keras.layers.Add()([global_fusion_2, global_residual])
        
        # Reconstruction layer
        reconstruction = tf.keras.layers.Conv3D(1, self.filter_size,
                                               kernel_initializer=self.WEIGHTS_INIT,
                                               use_bias=self.UTILIZE_BIAS,
                                               padding='same', 
                                               dtype='float32',
                                               name='reconstruction')(deep_features)
        
        # Final residual connection with input
        final_output = tf.keras.layers.Add()([reconstruction, 
                                             tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(inputs)])
        
        return tf.keras.Model(inputs=inputs, outputs=final_output)


class Discriminator:
    
    def __init__(self, PATCH_SIZE=40, UTILIZE_BIAS=True):
        self.k                     = 32
        self.filter_size           = 3
        self.PATCH_SIZE            = PATCH_SIZE
        self.UTILIZE_BIAS          = UTILIZE_BIAS
        self.WEIGHTS_INIT          = tf.keras.initializers.HeUniform()
        
    def conv_lay_norm_lrelu(self, filter_number, strides):
        result = tf.keras.Sequential()
        result.add(tf.keras.layers.Conv3D(filter_number, self.filter_size, kernel_initializer=self.WEIGHTS_INIT, 
                                          strides=strides, use_bias=self.UTILIZE_BIAS, padding='same'))
        result.add(tf.keras.layers.LayerNormalization())
        result.add(tf.keras.layers.LeakyReLU(alpha=0.2))    
        return result

    def create_discriminator(self):
        inputs = tf.keras.layers.Input(shape=[self.PATCH_SIZE,self.PATCH_SIZE,self.PATCH_SIZE,1])
        conv1 = tf.keras.layers.Conv3D(self.k, self.filter_size, kernel_initializer=self.WEIGHTS_INIT,  
                                       use_bias=self.UTILIZE_BIAS, padding='same')(inputs)
        lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1)
        conv_lay_norm_lrelu1 = self.conv_lay_norm_lrelu(self.k, 2)(lrelu1)
        conv_lay_norm_lrelu2 = self.conv_lay_norm_lrelu(2*self.k, 1)(conv_lay_norm_lrelu1)
        conv_lay_norm_lrelu3 = self.conv_lay_norm_lrelu(2*self.k, 2)(conv_lay_norm_lrelu2)
        conv_lay_norm_lrelu4 = self.conv_lay_norm_lrelu(4*self.k, 1)(conv_lay_norm_lrelu3)
        conv_lay_norm_lrelu5 = self.conv_lay_norm_lrelu(4*self.k, 2)(conv_lay_norm_lrelu4)
        conv_lay_norm_lrelu6 = self.conv_lay_norm_lrelu(8*self.k, 1)(conv_lay_norm_lrelu5)
        conv_lay_norm_lrelu7 = self.conv_lay_norm_lrelu(8*self.k, 2)(conv_lay_norm_lrelu6)
        flatten = tf.keras.layers.Flatten()(conv_lay_norm_lrelu7)
        dense1 = tf.keras.layers.Dense(1024)(flatten)
        lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)(dense1)
        output = tf.keras.layers.Dense(1, dtype='float32')(lrelu2)
        return tf.keras.Model(inputs=inputs, outputs=output)


class Model3DRDN:

    def __init__(self, PATCH_SIZE=40, NO_OF_RDB_BLOCKS=8, GROWTH_RATE=16, NO_OF_DENSE_LAYERS=4, 
                 BATCH_SIZE=2, LR_G=1e-4, LR_D=1e-4, LAMBDA_ADV=0.01, LAMBDA_GRD_PEN=10, 
                 MODEL="3DRDN", CRIT_ITER=3, TRAIN_ONLY='', RESIDUAL_SCALING=0.2):
        assert(MODEL in ["3DRDN", "3DRDN-WGAN"])
        self.MODEL                 = MODEL
        self.CRIT_ITER             = CRIT_ITER
        self.TRAIN_ONLY            = TRAIN_ONLY
        self.NO_OF_RDB_BLOCKS      = NO_OF_RDB_BLOCKS
        self.GROWTH_RATE           = GROWTH_RATE
        self.NO_OF_DENSE_LAYERS    = NO_OF_DENSE_LAYERS
        self.RESIDUAL_SCALING      = RESIDUAL_SCALING
        self.PATCH_SIZE            = PATCH_SIZE
        self.BATCH_SIZE            = BATCH_SIZE
        self.LR_G                  = LR_G
        self.LR_D                  = LR_D
        self.LAMBDA_ADV            = LAMBDA_ADV
        self.LAMBDA_GRD_PEN        = LAMBDA_GRD_PEN
        
        self.summary_writer_train  = tf.summary.create_file_writer("plots/training")
        self.summary_writer_valid  = tf.summary.create_file_writer("plots/validation")
        
        gen                        = Generator(PATCH_SIZE=PATCH_SIZE, 
                                             NO_OF_RDB_BLOCKS=NO_OF_RDB_BLOCKS,
                                             GROWTH_RATE=GROWTH_RATE,
                                             NO_OF_DENSE_LAYERS=NO_OF_DENSE_LAYERS,
                                             RESIDUAL_SCALING=RESIDUAL_SCALING)
        self.generator_g           = gen.create_generator()
        self.generator_g_optimizer = tf.keras.optimizers.Adam(LR_G)

        self.load_generator_g()
        if MODEL=="3DRDN":
            return
        disc                           = Discriminator(PATCH_SIZE=PATCH_SIZE)
        self.discriminator_y           = disc.create_discriminator()
        self.discriminator_y_optimizer = tf.keras.optimizers.Adam(LR_D)

        self.lambda_adv                = LAMBDA_ADV
        self.lambda_grad_pen = tf.constant(LAMBDA_GRD_PEN, dtype=tf.float32)
        self.load_discriminator_y()
    
    def load_generator_g(self, tensor_path = "checkpoints/tensor_checkpoints/generator_checkpoint/G/"):
        ckpt = tf.train.Checkpoint(generator_g=self.generator_g, generator_g_optimizer=self.generator_g_optimizer)
        self.gen_g_ckpt_manager = tf.train.CheckpointManager(ckpt, tensor_path, max_to_keep=1)
        if self.gen_g_ckpt_manager.latest_checkpoint:
            status = ckpt.restore(self.gen_g_ckpt_manager.latest_checkpoint)
            status.expect_partial()
            log('Generator G latest checkpoint restored!!')
        else:
            log("No checkpoint found for generator G! Staring from scratch!")
    
    def load_pretrained_generator(self, tensor_path = "/home/nhabdoan/thinclient_drives/Binh/New3DRDN_NotSkull/generator_checkpoint/G"):
        ckpt = tf.train.Checkpoint(generator_g=self.generator_g, generator_g_optimizer=self.generator_g_optimizer)
        self.gen_g_ckpt_manager = tf.train.CheckpointManager(ckpt, tensor_path, max_to_keep=1)
        if self.gen_g_ckpt_manager.latest_checkpoint:
            status = ckpt.restore(self.gen_g_ckpt_manager.latest_checkpoint)
            status.expect_partial()
            log('Pretrained Generator G latest checkpoint restored!!')
        else:
            log("No checkpoint found for generator G! Staring from scratch!")    

    def load_discriminator_y(self, tensor_path = "checkpoints/tensor_checkpoints/discriminator_checkpoint/Y/"):
        ckpt = tf.train.Checkpoint(discriminator_y=self.discriminator_y, 
            discriminator_y_optimizer=self.discriminator_y_optimizer)
        self.disc_y_ckpt_manager = tf.train.CheckpointManager(ckpt, tensor_path, max_to_keep=1)
        if self.disc_y_ckpt_manager.latest_checkpoint:
            status2 = ckpt.restore(self.disc_y_ckpt_manager.latest_checkpoint)
            status2.expect_partial()
            log('Discriminator Y latest checkpoint restored!!')
        else:
            log("No checkpoint found for discriminator Y! Staring from scratch!")

    def save_models(self, epoch):
        ckpt_save_path = self.gen_g_ckpt_manager.save()
        log('Saving Generator G checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))
        if self.MODEL=="3DRDN":
            return
        ckpt_save_path = self.disc_y_ckpt_manager.save()
        log('Saving Discriminator Y checkpoint for epoch {} at {}'.format(epoch, ckpt_save_path))

        
    @tf.function
    def gen_g_supervised_train_step(self, real_x, real_y, epoch):
        with tf.GradientTape() as tape:
            fake_y             = self.generator_g(real_x, training=True)
            gen_g_super_loss   = supervised_loss(real_y, fake_y)
        gradients_of_generator = tape.gradient(gen_g_super_loss, self.generator_g.trainable_variables)
        self.generator_g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator_g.trainable_variables))
        with self.summary_writer_train.as_default():
            tf.summary.scalar('Generator G Total Loss', gen_g_super_loss, step=epoch)
            tf.summary.scalar('Mean Absolute Error', gen_g_super_loss, step=epoch)

    @tf.function
    def gen_g_gan_train_step(self, real_x, real_y, epoch):
        with tf.GradientTape() as tape:
            fake_y           = self.generator_g(real_x, training=True)
            disc_fake_y      = self.discriminator_y(fake_y, training=True)
            gen_g_super_loss = supervised_loss(real_y, fake_y)
            gen_g_adv_loss   = self.lambda_adv*generator_loss(disc_fake_y)
            total_gen_g_loss = tf.cast(gen_g_super_loss, tf.float64) + tf.cast(gen_g_adv_loss, tf.float64)
        generator_g_gradient = tape.gradient(total_gen_g_loss, self.generator_g.trainable_variables)
        self.generator_g_optimizer.apply_gradients(zip(generator_g_gradient, self.generator_g.trainable_variables))
        with self.summary_writer_train.as_default():
            tf.summary.scalar('Generator G Total Loss', total_gen_g_loss, step=epoch)
            tf.summary.scalar('Generator G Adversarial Loss', gen_g_adv_loss, step=epoch)
            tf.summary.scalar('Mean Absolute Error', gen_g_super_loss, step=epoch)

    @tf.function
    def disc_y_train_step(self, real_x, real_y, epoch):
        # Cast inputs to float32 from the beginning
        real_x = tf.cast(real_x, tf.float32)
        real_y = tf.cast(real_y, tf.float32)
        
        batch_size = tf.shape(real_x)[0]
        with tf.GradientTape() as tape:
            fake_y = self.generator_g(real_x, training=True)
            fake_y = tf.cast(fake_y, tf.float32)
            
            disc_real_y = self.discriminator_y(real_y, training=True)
            disc_fake_y = self.discriminator_y(fake_y, training=True)
            
            differences_y = fake_y - real_y
            
            alpha = tf.random.uniform([batch_size, 1, 1, 1, 1], 0, 1, dtype=tf.float32)
            interpolates_y = real_y + (alpha * differences_y)
            
            with tf.GradientTape() as t:
                t.watch(interpolates_y)
                pred_y = self.discriminator_y(interpolates_y, training=True)
            gradients_y = t.gradient(pred_y, [interpolates_y])[0]
            slopes_y = tf.sqrt(tf.reduce_sum(tf.square(gradients_y), axis=[1, 2, 3, 4]))
            gradient_penalty_y = tf.reduce_mean((slopes_y-1.)**2)
            
            # Cast disc_y_loss to float32 to match gradient_penalty_y
            disc_y_loss = tf.cast(discriminator_loss(disc_real_y, disc_fake_y), tf.float32)
            gradient_penalty_term = self.lambda_grad_pen * gradient_penalty_y
            total_disc_y_loss = disc_y_loss + gradient_penalty_term
        
        discriminator_y_gradient = tape.gradient(total_disc_y_loss, self.discriminator_y.trainable_variables)
        self.discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradient, self.discriminator_y.trainable_variables))
        
        with self.summary_writer_train.as_default():
            tf.summary.scalar('Discriminator Y Loss', total_disc_y_loss, step=epoch)

    def training(self, real_x, real_y, epoch):
        if self.MODEL=="3DRDN":
            self.gen_g_supervised_train_step(real_x, real_y, epoch)
            return
        elif self.MODEL=="3DRDN-WGAN":
            if self.TRAIN_ONLY=="DISCRIMINATORS":
                self.disc_y_train_step(real_x, real_y, epoch)
                return
            else:
                for _ in range(self.CRIT_ITER):
                    self.disc_y_train_step(real_x, real_y, epoch)
                self.gen_g_gan_train_step(real_x, real_y, epoch)
                return