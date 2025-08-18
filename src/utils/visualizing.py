import os
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
def save_slices(image, output_dir, subject_id):

    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)
    x_mid = image.shape[0] // 2
    y_mid = image.shape[1] // 2
    z_mid = image.shape[2] // 2

    slices = {
        "axial": np.rot90(image[:, :, z_mid], k=1),
        "coronal": np.rot90(image[:, y_mid, :].T, k=2),
        "sagittal": np.fliplr(np.rot90(np.flipud(np.rot90(image[x_mid, :, :].T, k=2)), k=2))
    }

    for plane, img_slice in slices.items():
        plt.figure(figsize=(4, 4))
        plt.imshow(img_slice, cmap="gray")
        plt.axis("off")
        save_path = os.path.join(subject_dir, f"{subject_id}_{plane}.png")
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
        plt.close()


nii_path = ""
output_nii = nib.load(nii_path)
output_data = output_nii.get_fdata()

# Extract subject ID from path (assuming BIDS format: sub-<subject_id>)
subject_id = os.path.basename(nii_path).split("_")[0].replace("sub-", "")

# Save slices
save_slices(output_data, 
            output_dir="original_images", 
            subject_id=subject_id)