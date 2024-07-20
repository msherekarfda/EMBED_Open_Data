import albumentations as albu
from albumentations.pytorch import ToTensorV2




def get_transforms(augment=False):
    if augment:    
        return albu.Compose([
                #albu.Normalize(mean=[0.304,0.304,0.304], std=[0.203,0.203,0.203]),
                #albu.PadIfNeeded (min_height=1536, min_width=1536, border_mode=0, value=0, p=1.0),
                #albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=[-0.15, 0.15], rotate_limit=45, interpolation=2, border_mode=0, value=0, mask_value=0, p=0.5),
                #albu.Flip(p=0.75),
                albu.CoarseDropout(max_holes=1, max_height=0.33, max_width=0.33,  min_height=0.02, min_width=0.02, fill_value=[77.52425988, 77.52425988, 77.52425988], p=0.5),
                ToTensorV2(),
            ],
            p=1,
        )
    else:
        return albu.Compose([
                #albu.Normalize(mean=[0.304,0.304,0.304], std=[0.203,0.203,0.203]),
                ToTensorV2(),
            ],
            p=1,
        )