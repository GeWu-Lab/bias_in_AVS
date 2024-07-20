# 💬 bias_in_AVS
Official repository for "Unveiling and Mitigating Bias in Audio Visual Segmentation" in ACM MM 2024.

**Paper Title: "Unveiling and Mitigating Bias in Audio Visual Segmentation"**

**Authors: [Peiwen Sun](https://peiwensun2000.github.io), [Honggang Zhang](https://teacher.bupt.edu.cn/zhanghonggang/en/index/40466/list/) and [Di Hu](https://dtaoo.github.io/index.html)**

**Accepted by: The 32nd ACM International Conference on Multimedia (ACM MM 2024)**

🚀: Project page here: [Project Page](https://gewu-lab.github.io/bias_in_AVS/)

📄: Paper here: [Paper](https://arxiv.org/placeholder)

🔍: Supplementary material: [Supplementary](https://arxiv.org/placeholder)
## Abstract
ACommunity researchers have developed a range of advanced audio-visual segmentation models aimed at improving the quality of sounding objects' masks. While masks created by these models may **initially appear plausible**, they occasionally exhibit anomalies with **incorrect grounding logic**. We attribute this to real-world inherent preferences and distributions as a simpler signal for learning than the complex audio-visual grounding, which leads to the disregard of important modality information. Generally, the anomalous phenomena are often complex and cannot be directly observed systematically. In this study, we made a pioneering effort with the proper synthetic data to categorize and analyze phenomena as two types **“audio priming bias”** and **“visual prior”** according to the source of anomalies. For audio priming bias, to enhance audio sensitivity to different intensities and semantics, a perception module specifically for audio perceives the latent semantic information and incorporates information into a limited set of queries, namely active queries. Moreover, the interaction mechanism related to such active queries in the transformer decoder is customized to adapt to the need for interaction regulating among audio semantics. For visual prior, multiple contrastive training strategies are explored to optimize the model by incorporating a biased branch, without even changing the structure of the model. During experiments, observation demonstrates the presence and the impact that has been produced by the biases of the existing model. Finally, through experimental evaluation of AVS benchmarks, we demonstrate the effectiveness of our methods in handling both types of biases, achieving competitive performance across all three subsets. [Page](https://gewu-lab.github.io/bias\_in\_AVS/)

<img width="1009" alt="image" src="image/teaser.png">

## Code instruction

The overall training pipeline follows the list below.

1. Data Preparation
    * Prepare the training data
    * Download the pretrained ckpt
    * (Optional) Download the well-trained ckpt for finetuning
2. Audio pretransform
    * Audio clustering follows the HuBERT clustering pipeline in [github](https://github.com/bshall/hubert)
    * Audio classification follows the BETAS pipeline in [github](https://github.com/microsoft/unilm/tree/master/beats)
    * Save the cluster or class information in `pkl`
3. (Optional) When training AVSS dataset, we gradually add `v1s, v2, v1m` in the data pool. It brings minor benefits to the performance of a curriculum training strategy.
4. Training the model with the debias strategy.
5. Evaluating on AVS Benchmark.


### Data Preparation

Please refer to the link [AVSBenchmark](https://github.com/OpenNLPLab/AVSBench) to download the datasets. You can put the data under `data` folder or rename your own folder. Remember to modify the path in config files. The `data` directory is as below:
```
|--data
 |--v2
 |--v1m
 |--v1s
 |--metadata.csv
```
Note: v1s is also known as S4, and v1m is also known as MS3. The AVSBench benchmark is strictly followed.

We use Mask2Former model with Swin-B pre-trained on ADE20k as the backbone, which can be downloaded in this [link](https://drive.google.com/file/d/15wI-2M3Cfovl6oNTvBSQfDYKf5FmqooD/view?usp=drive_link). Don't forget to modify the `placeholder` in python files to your own path.

Our well trained model can be downloaded in this [link](https://drive.google.com/placeholder). Don't forget to modify the `placeholder` in Python files to your own path.

### Training
For S4 and MS3 subtasks, you can simply modify config in python files and replace the `pkl` path of pre-transform of clustering or classification:  
~~~shell
cd AVS
sh 
~~~
For AVSS subtask, the procedure is basically the same,
~~~shell
cd AVS
sh 
~~~

### Testing
Normally, just like the former works, test can be done during training. However, we still are able to make small changes on the training code. For example, comment out the training part and the remaining part is just testing.


### Download checkpoints

We also provide pre-trained models for all three subtasks. You can download them from the [following links]().

## Citation
If you find this work useful, please consider citing it.

~~~BibTeX
@article{sun2024unveiling,
          title={Unveiling and Mitigating Bias in Audio Visual Segmentation},
          author={Sun, Peiwen and Zhang, Honggang and Hu, Di},
          journal={Proceedings of the 32nd ACM International Conference on Multimedia (ACM MM)},
          year={2024},
 }
~~~

