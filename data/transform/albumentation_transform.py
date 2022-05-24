import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
# Reference: https://albumentations.ai/docs/getting_started/mask_augmentation/

def albumentation_transform(patch_size, image, mask):
    transform = A.Compose([
                           A.ToFloat(max_value=255.0),
                           # Using Imagenet stats. TODO replace this.   
                           A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=1.0, p=1.0),
                           # A.Resize(height=patch_size, width=patch_size, p=1),
                           A.RandomCrop(width=patch_size, height=patch_size),
                           ToTensorV2()
    ])
    transformed = transform(image=image, mask=mask)
    # ToFloat is not applied on mask image. 
    transformed["mask"] = (transformed["mask"]/ 255.0)
    return transformed