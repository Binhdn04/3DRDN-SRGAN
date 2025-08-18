import tensorflow as tf
import os
import argparse
from model  import Generator, Discriminator

class CheckpointToSavedModelConverter:
    def __init__(self, model_type="3DRDN", patch_size=40, db=3, du=4):

        assert model_type in ["3DRDN", "3DRDN-WGAN"], f"Unsupported model type: {model_type}"
        self.model_type = model_type
        self.patch_size = patch_size
        self.db = db
        self.du = du
        
    def create_generator(self):        
        gen = Generator(PATCH_SIZE=self.patch_size, 
                       NO_OF_DENSE_BLOCKS=self.db, 
                       NO_OF_UNITS_PER_BLOCK=self.du)
        return gen.create_generator()
    
    def create_discriminator(self):
        disc = Discriminator(PATCH_SIZE=self.patch_size)
        return disc.create_discriminator()
    
    def load_generator_from_checkpoint(self, checkpoint_path):
        print(f"Loading generator from: {checkpoint_path}")
        generator = self.create_generator()
        optimizer = tf.keras.optimizers.Adam(1e-4)
        ckpt = tf.train.Checkpoint(generator_g=generator, generator_g_optimizer=optimizer)
        
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        
        if ckpt_manager.latest_checkpoint:
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print(f"Successfully loaded generator from {ckpt_manager.latest_checkpoint}")
            return generator
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    def load_discriminator_from_checkpoint(self, checkpoint_path):
        print(f"Loading discriminator from: {checkpoint_path}")

        discriminator = self.create_discriminator()
        optimizer = tf.keras.optimizers.Adam(1e-4)
        ckpt = tf.train.Checkpoint(discriminator_y=discriminator, discriminator_y_optimizer=optimizer)
        
        # Load checkpoint
        ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=1)
        
        if ckpt_manager.latest_checkpoint:
            status = ckpt.restore(ckpt_manager.latest_checkpoint)
            status.expect_partial()
            print(f"Successfully loaded discriminator from {ckpt_manager.latest_checkpoint}")
            return discriminator
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    def save_as_savedmodel(self, model, output_path, model_name):

        save_path = os.path.join(output_path, f"{model_name}_savedmodel")
        os.makedirs(save_path, exist_ok=True)
        
        input_shape = [None, self.patch_size, self.patch_size, self.patch_size, 1]
        input_spec = tf.TensorSpec(shape=input_shape, dtype=tf.float64)
        
        @tf.function(input_signature=[input_spec])
        def inference_func(x):
            return model(x, training=False)
        
        tf.saved_model.save(model, save_path, signatures={'serving_default': inference_func})
        print(f"Saved {model_name} to {save_path}")
        
        return save_path
    
    def convert_generator_only(self, generator_checkpoint_path, output_path):
        print("Converting 3DRDN generator...")
        generator = self.load_generator_from_checkpoint(generator_checkpoint_path)
        
        os.makedirs(output_path, exist_ok=True)
        saved_path = self.save_as_savedmodel(generator, output_path, "3DRDN_generator")
        
        return {"generator": saved_path}
    
    def convert_generator_and_discriminator(self, generator_checkpoint_path, discriminator_checkpoint_path, output_path):
        print("Converting 3DRDN-WGAN generator and discriminator...")
        generator = self.load_generator_from_checkpoint(generator_checkpoint_path)
        discriminator = self.load_discriminator_from_checkpoint(discriminator_checkpoint_path)
        
        os.makedirs(output_path, exist_ok=True)
        saved_paths = {}
        saved_paths["generator"] = self.save_as_savedmodel(generator, output_path, "3DRDN_WGAN_generator")
        saved_paths["discriminator"] = self.save_as_savedmodel(discriminator, output_path, "3DRDN_WGAN_discriminator")
        
        return saved_paths

def main():
    parser = argparse.ArgumentParser(description='Convert 3DRDN/3DRDN-WGAN checkpoints to SavedModel')
    parser.add_argument('--model_type', type=str, required=True,
                       choices=['3DRDN', '3DRDN-WGAN'],
                       help='Type of model to convert')
    parser.add_argument('--generator_checkpoint', type=str, required=True,
                       help='Path to generator checkpoint directory')
    parser.add_argument('--discriminator_checkpoint', type=str, default=None,
                       help='Path to discriminator checkpoint directory (required for 3DRDN-WGAN)')
    parser.add_argument('--output_path', type=str, required=True,
                       help='Output directory for SavedModel files')
    parser.add_argument('--patch_size', type=int, default=40,
                       help='Input patch size')
    parser.add_argument('--dense_blocks', type=int, default=3,
                       help='Number of dense blocks')
    parser.add_argument('--dense_units', type=int, default=4,
                       help='Number of dense units per block')
    
    args = parser.parse_args()
    
    if args.model_type == '3DRDN-WGAN' and args.discriminator_checkpoint is None:
        parser.error("--discriminator_checkpoint is required for 3DRDN-WGAN model")
    
    converter = CheckpointToSavedModelConverter(
        model_type=args.model_type,
        patch_size=args.patch_size,
        db=args.dense_blocks,
        du=args.dense_units
    )
    
    print(f"Converting {args.model_type} model...")
    print(f"Generator checkpoint: {args.generator_checkpoint}")
    if args.discriminator_checkpoint:
        print(f"Discriminator checkpoint: {args.discriminator_checkpoint}")
    print(f"Output path: {args.output_path}")
    print("-" * 50)
    
    try:
        if args.model_type == "3DRDN":
            saved_paths = converter.convert_generator_only(
                args.generator_checkpoint, 
                args.output_path
            )
        else:  # 3DRDN-WGAN
            saved_paths = converter.convert_generator_and_discriminator(
                args.generator_checkpoint,
                args.discriminator_checkpoint,
                args.output_path
            )
        
        print("\n" + "="*60)
        print("CONVERSION COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Saved models:")
        for model_name, path in saved_paths.items():
            print(f"  {model_name}: {path}")
        print("\nYou can now load these models using:")
        print("  model = tf.saved_model.load('/path/to/savedmodel')")
            
    except Exception as e:
        print(f"\nError during conversion: {str(e)}")
        raise

if __name__ == "__main__":
    main()