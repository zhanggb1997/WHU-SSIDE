# WHU-SSIDE: Satellite Stereo Image Disparity Estimation

The WHU-SSIDE data and codes of Meta-MRGE.

## 📖 Overview

WHU-SSIDE is a comprehensive project focused on satellite stereo image disparity estimation, featuring a high-quality dataset and advanced disparity estimation algorithms. This project addresses two major challenges in satellite stereo disparity estimation: limitations in dataset quality and diversity, and the difficulty of achieving synergistic optimization across large-scale and small-scale disparity regions.


## 🎯 Key Features
#### 🌟 WHU-SSIDE Dataset

*Large Scale*:   Contains 3,737 satellite stereo image pairs

*Challenging Scenes*:   Specifically includes high-rise building zones with enormous disparities

*High-precision Ground Truth*:   Incorporates a multi-error compensation mechanism ensuring sub-pixel accuracy (MAE < 1 pixel)

*High Reliability*:   Establishes a highly reliable and challenging benchmark for satellite stereo disparity estimation

[Dataset-google drive](https://drive.google.com/drive/folders/1-E1LT_HW_qB0g7gqBh4wKPQskVMuzM85?usp=sharing)

#### 🚀 Meta-MRGE Algorithm 

*Metadata Guidance*:   Embeds satellite imaging parameters to guide high-precision disparity estimation

*Multi-range Geometric Encoding*:   Constructs multi-level geometry-aware cost volume

*Iterative Refinement*:   Implements iterative disparity refinement



### 📊 Performance Highlights

| Metric | WHU-SSIDE | WHU-Stereo |
|--------|-----------|------------|
| D1     | 8.35%     | 9.74%      |
| EPE    | 1.2585    | 1.4986     |
| RMSE   | 4.2812    | 3.6976     |

#### Results on WHU-SSIDE Dataset

*End-Point Error (EPE)*:   1.2585 (lowest)

*Root Mean Square Error (RMSE)*:   4.2812 (lowest)

##### Results on WHU-Stereo Dataset

*D1 Metric*:   3.01% improvement over second-best method

*EPE*:   Improvement of 0.2265

*RMSE*:   Improvement of 0.2492

> **Note**: The method shows significant improvements in large-disparity regions. Ablation studies confirm that the metadata embedding mechanism enhances feature matching robustness.


```
WHU-SSIDE/

├── 📦 Dataset/                    # Dense matching dataset (cloud storage links)
│   ├── download_links.md          # Dataset download instructions
│   ├── dataset_structure.md       # Dataset structure documentation
│   ├── metadata_example.md        # Example of meta data
│   └── data_example.png           # Example of satellite stereo image pairs 
|
├── 🔧 StereoDatasetConstruction/  # Dataset construction code
│   ├── rpc/                       # RPC model utils code
│   ├── image/                     # Remote Sensing image process code
|   .
|   .
|   .
│   └── main.py                    # Execute main function for data construction
|
├── 🤖  DenseMatchingModel/        # Dense matching model code
│   ├── models/                    # Model source code
│   ├── data/                      # Training configurations
│   ├── core/                      # Pre-trained weights
│   ├── config/                    # Pre-trained weights
│   ├── utils/                     # Tools
│   └── main.py                    # Execute main function for dense match model train / inference
|
├── 📚 Docs/                       # Documentation
│   ├── paper.pdf                  # Research paper
│   ├── citation.bib               # Citation information
│   └── tutorial.md                # Tutorial documentation
|
├── 📄 requirements                # Requirements file
└── 📄 LICENSE                     # License file
```

## 🚀 Quick Start

#### 1. Dataset Download

[Dataset-google drive](https://drive.google.com/drive/folders/1-E1LT_HW_qB0g7gqBh4wKPQskVMuzM85?usp=sharing)


#### 2. Environment Setup

``` pip install -r requirements.txt ```

#### 3. Model Training/Testing

```
cd DenseMatchingModel

python main.py
```


## 🤝 Contributing

The numerical calculations in this article have been done on the supercomputing system in the Supercomputing Center, Wuhan University, Wuhan, China.

We welcome contributions from the community! You can contribute by:
1. Submitting issues for bug reports or feature requests
2. Creating pull requests to improve the codebase
3. Sharing use cases and experimental results

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📞 Contact

- Project Homepage: [GitHub Repository](https://github.com/zhanggb1997/WHU-SSIDE)
- Issue Tracker: [GitHub Issues](https://github.com/zhanggb1997/WHU-SSIDE/issues)
- Email: [zhanggb1997@whu.edu.cn]

---

<div align="center">
  
**🌟 If this project is helpful to you, please give us a star!** ⭐

</div>
