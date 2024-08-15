import os
import cv2
import albumentations as A

# Deklarasikan pipeline augmentasi
'''
bisa disesuaikan dengan augmentasi yang disediakan pada library Albumentation.
https://demo.albumentations.ai/
'''
augmentations = [
    ("hori", A.Compose([A.HorizontalFlip()])),
    ("blur", A.Compose([A.MotionBlur(blur_limit=(5, 5), allow_shifted=True)])),
    ("noise", A.Compose([A.ISONoise(intensity=(0.4, 0.4), color_shift=(0.3, 0.3))])),
    ("bright", A.Compose([A.RandomBrightnessContrast(brightness_limit=(-0.4, 0.4),
                                                     contrast_limit=(-0.3, 0.3),
                                                     brightness_by_max=True)])),
    ("equalize", A.Compose([A.Equalize(mode='cv', by_channels=True)])),
    ("rotate", A.Compose([A.Rotate(limit=(-90, 90), mask_value=None,
                                   rotate_method='largest_box', crop_border=False)])),
    ("distort", A.Compose([A.OpticalDistortion(distort_limit=(-0.38, -0.38),
                                               shift_limit=(-0.17, -0.17),
                                               interpolation=1,
                                               border_mode=1,
                                               value=(0, 0, 0),
                                               mask_value=None)]))
]

# Direktori parent gambar
directory = "DIR-MAINFOLDER"

# Iterasi melalui setiap file dalam direktori
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):  # tambahkan jenis file gambar lain jika diperlukan
        # Baca gambar dengan OpenCV dan ubah ke ruang warna RGB
        image_path = os.path.join(directory, filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Augmentasi gambar
        for aug_name, augmentation in augmentations:
            transformed = augmentation(image=image)["image"]

            # Simpan gambar augmentasi
            output_path = os.path.join(directory, f"augmented_{aug_name}_" + filename)
            cv2.imwrite(output_path, cv2.cvtColor(transformed, cv2.COLOR_RGB2BGR))