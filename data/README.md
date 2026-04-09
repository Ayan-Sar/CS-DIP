# Dataset Download Instructions

This directory is used to store benchmark datasets for CS-DIP evaluation.
The datasets are **not** included in the repository due to their size.

## Required Datasets

| Dataset   | Images | Primary Use          | Source                                      |
|-----------|--------|----------------------|---------------------------------------------|
| Set5      | 5      | Super-Resolution     | Bevilacqua et al., BMVC 2012               |
| Set14     | 14     | SR / Denoising       | Zeyde et al., LNCS 2010                    |
| BSD68     | 68     | Image Denoising      | Roth & Black, IJCV 2009                    |
| Urban100  | 100    | Geometric SR         | Huang et al., CVPR 2015                    |

## Download Instructions

### Option 1: Manual Download
Download from commonly used research repositories:

- **Set5 & Set14:** [https://github.com/jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR)
- **BSD68:** [https://github.com/clausmichele/CBSD68-dataset](https://github.com/clausmichele/CBSD68-dataset)
- **Urban100:** [https://github.com/jbhuang0604/SelfExSR](https://github.com/jbhuang0604/SelfExSR)

### Option 2: Consolidated Collection
- [https://github.com/majedelhelou/denoising_datasets](https://github.com/majedelhelou/denoising_datasets)

## Expected Directory Structure

After downloading, organize the images as follows:

```
data/
├── Set5/
│   ├── baby.png
│   ├── bird.png
│   ├── butterfly.png
│   ├── head.png
│   └── woman.png
├── Set14/
│   ├── baboon.png
│   ├── barbara.png
│   └── ... (14 images)
├── BSD68/
│   ├── test001.png
│   └── ... (68 images)
└── Urban100/
    ├── img001.png
    └── ... (100 images)
```

> **Note:** The exact filenames do not matter. CS-DIP loads all image files
> (`.png`, `.jpg`, `.bmp`, `.tif`) from each dataset directory automatically.
