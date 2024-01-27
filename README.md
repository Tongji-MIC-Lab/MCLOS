#  Memory-Based Contrastive Learning for Incremental Few-Shot Semantic Segmentation via Optimized Sampling
Yuxuan Zhang, Miaojing Shi, Taiyi Su, and Hanli Wang
### OverView

Incremental few-shot semantic segmentation (IFSS) aims to incrementally expand a semantic segmentation model's ability to identify new classes based on few samples. However, it grapples with the dual challenges of catastrophic forgetting due to feature drift in old classes, and overfitting triggered by inadequate samples for pixel-level segmentation in new classes. To address these issues, a novel approach integrating pixel-wise and region-wise contrastive learning, complemented by an optimized example and anchor sampling strategy. The proposed method incorporates a region memory and pixel memory designed to explore the high-dimensional embedding space more effectively. The memory, retaining the feature embeddings of known classes, facilitates the calibration and alignment of seen class features during the learning process of new classes. To further mitigate overfitting, the proposed approach implements optimized example and anchor sampling strategies. Extensive experiments show the competitive performance of the proposed method. 

### Methods

The pipeline of the proposed method for IFSS is shown in Fig. 1. Dynamic memory aims to preserve features at two levels of granularity, reducing the model's tendency to forget base classes. Optimized sampling focuses on selecting valuable anchors and positive/negative samples. Combined with pixel-region contrastive learning of old class embeddings, it aids in calibrating and aligning old and new class features during the learning of new classes.

<p align="center">
<image src="source/fig1.jpg" width="650">
<br/><font>Fig. 1. Overall schematics of the proposed incremental few-shot segmentation method.</font>
</p>

### Result

We compare our method with the state-of-the-art IFSS methods. The comparison results are shown in Table 1-2. The remarkable performances on the datasets demonstrate the superiority of our method. We conduct extensive ablation experiments to verify the design of dynamic memory and different sampling strategies on the proposed method, the experimental results are given in Table 3-4.

<p align="center">
<font>Table 1.  Comparison with state-of-the-art methods on the PASCAL VOC 2012 datasets.</font><br/>
<image src="source/table1.jpg" width="650">
</p>

<p align="center">
<font>Table 2.  Comparison with state-of-the-art methods on the COCO dataset.</font><br/>
<image src="source/table2.jpg" width="650">
</p>

<p align="center">
<font>Table 3.  Ablation study on VOC to compare different memory designs.</font><br/>
<image src="source/table3.jpg" width="350">
</p>

<p align="center">
<font>Table 4.  Ablation study on VOC to compare different sampling strategies.</font><br/>
<image src="source/table4.jpg" width="350">
</p>

Fig. 2 shows the qualitative comparisons between our method and PIFS. As the figure
shows, our method provides more precise segmentation masks than PIFS.

<p align="center">
<image src="source/fig2.jpg" width="350">
<br/><font>Fig. 2. The qualitative comparison between PIFS and ours.</font>
</p>

As shown in Fig. 3, the learned pixel embeddings by ours become more compact and well separated, suggesting that our method shapes a well-structured semantic feature space by employing pixel-region contrastive learning.

<p align="center">
<image src="source/fig3.jpg" width="350">
<br/><font>Fig. 3. T-SNE visualization of features learned with (left) PIFS and (right) our method.</font>
</p>

## How to run
### Requirements
We have simple requirements:
The main requirements are:
```
python > 3.1
pytorch > 1.6
```
If you want to install a custom environment for this codce, you can run the following using [conda](https://docs.conda.io/projects/conda/en/latest/commands/install.html):
```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
conda install tensorboard
conda install jupyter
conda install matplotlib
conda install tqdm
conda install imageio

pip install inplace-abn
conda install -c conda-forge pickle5
```

### Datasets 
In the benchmark there are two datasets: Pascal-VOC 2012 and COCO (object only).
For the COCO dataset, we followed the COCO-stuff splits and annotations, that you can see [here](https://github.com/nightrome/cocostuff/).

To download dataset, follow the scripts: `data/download_voc.sh`, `data/download_coco.sh` 

To use the annotations of COCO-Stuff in our setting, you should preprocess it by running the provided script. \
Please, remember to change the path in the script before launching it!
`python data/coco/make_annotation.py`

Finally, if your datasets are in a different folder, make a soft-link from the target dataset to the data folder.
We expect the following tree:
```
/data/voc/dataset
    /annotations
        <Image-ID>.png
    /images
        <Image-ID>.png
        
/data/coco/dataset
    /annotations
        /train2017
            <Image-ID>.png
        /val2017
            <Image-ID>.png
    /images
        /train2017
            <Image-ID>.png
        /val2017
            <Image-ID>.png
```

### ImageNet Pretrained Models
After setting the dataset, you download the models pretrained on ImageNet using [InPlaceABN](https://github.com/mapillary/inplace_abn).
[Download](https://drive.google.com/file/d/1rQd-NoZuCsGZ7_l_X9GO1GGiXeXHE8CT/view) the ResNet-101 model (we only need it but you can also [download other networks](https://github.com/mapillary/inplace_abn) if you want to change it).
Then, put the pretrained model in the `pretrained` folder.


### Run
We provide different scripts to run the experiments (see `run` folder).
In the following, we describe the basic structure of them.

First, you should run the base step (or step 0).
```
exp --method FT --name FT --epochs 30 --lr 0.01 --batch_size 24
```
In this example, we are running the fine-tuning method (FT). For other methods (COS, SPN, DWI, RT) you can change the method name.
WI, PIFS and ours rely on the COS in the step 0, while FT, AMP, LWF, ILT, MIB rely on the FT one. 

After this, you can run the incremental steps.
There are a few options: (i) the task, (ii) the number of images (n_shot), and (iii) the sampling split (i_shot).

i) The list of tasks is:
```
voc:
    5-0, 5-1, 5-2, 5-3
coco:
    20-0, 20-1, 20-2, 20-3
```
For multi-step, you can append an `m` after the task (e.g., `5-0m`)

ii) We tested 1, 2, and 5 shot. You can specify it with the `nshot` option.

iii) We used three random sampling. You can specify it with the `ishot` option.

The training will produce both an output on the terminal and it will log on tensorboard at the `logs/<Exp_Name>` folder.
After the training, it will append a row in the csv file `logs/results/<dataset>/<task>.csv`.

### Acknowledgement
This repo is mainly built based on [PIFS](https://github.com/fcdl94/FSS). Thanks for their great work!

### Citiation
Please cite the following paper if you find this work useful:

Yuxuan Zhang, Miaojing Shi, Taiyi Su, and Hanli Wang, Memory-Based Contrastive Learning with Optimized Sampling for Incremental Few-Shot Semantic Segmentation (ISCAS'24), accepted, 2024.
