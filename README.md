***
This is the official repo for the paper [Robust Template Matching via Hierarchical Convolutional Features from a Shape Biased CNN](https://arxiv.org/abs/2007.15817). For more details, please refer to 

```
@InProceedings{gao2021robust,
      title={Robust Template Matching via Hierarchical Convolutional Features from a Shape Biased CNN}, 
      author={Bo Gao and M. W. Spratling},
      year={2021},
      eprint={arXiv:2007.15817},
}
```
# Motivation
The idea is straightforward which is to investigate if enhancing the CNN's encoding of shape information can produce more distinguishable features that improve the performance of template matching. 
# Results 
<img src="https://raw.githubusercontent.com/iminfine/Deep-DIM/master/figure/results.PNG" width="450"/> 

# Dependencies
- Dependencies in our experiment, not necessary to be exactly same version but later version is preferred
- python=3.7
- pytorch=1.2.0
# Demo Code
Download the pretrained model from [here](https://drive.google.com/file/d/1bx2oDhnD9gUA5jhzgGXRmynhh4XPyBGV/view?usp=sharing) and put it into ./model 

The results of using features from all combinations of three layers can be downloaded from [here](https://drive.google.com/file/d/1b4O1At_q7Q-Ib6drlLFEcv5iSY4O4uBx/view?usp=sharing).
## Run DIM on BBS dataset using features form the best combination.
```bash
python deep_DIM.py --Dataset BBS --Mode Best 
```
## Run DIM on BBS dataset using features from all possible combinations of three layers.
```bash
python deep_DIM.py --Dataset BBS --Mode All 
```

# What is DIM?
DIM is a recent state-of-the-art template matching method using the mechanism explaining away. An illustration of explaining away can be found below. Red rectangle area in the left image (template image) is the template for matching, and four same size green rectangle areas are the additional templates. These templates compete with each other to be matched to the search image (Right). To be specific, only one template is supported to be the best matching one at every location whereas the similarities of others are suppressed or explained away. The corresponding matching results are shown in the right. The details of DIM can be found [here](https://nms.kcl.ac.uk/michael.spratling/Doc/dim_patchmatching.pdf).

<img src="https://raw.githubusercontent.com/iminfine/Deep-DIM/master/figure/DIM.png" width="800"/> 
