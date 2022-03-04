# **TSGV-Learning-List**

## **说明**

总结了2017年至今在TSGV方向上的相关工作。
Temporal Sentence Grounding in Videos (**TSGV**) 
Natural Language Video Localization (**NLVL**)
Video Moment Retrieval (**VMR**)
该任务的目标是给定一段语言描述，在一个未经裁剪的长视频中定位出该语言所描述的视频片段。

## **目录**

- [数据集](#数据集)
- [相关工作](#相关工作)
  - [Survey](#survey)
  - [Sliding Window-based Method](#Sliding-Window-based-Method)
  - [Proposal Generated Method](#Proposal-Generated-Method) 
  - [Anchor-based Method](#Anchor-based-Method)
  - - [Standard Anchor-based Method](#Standard-Anchor-based-Method)
  - - [2D-Map Anchor-based Method](#2D-Map-Anchor-based-Method)
  - [Regression-based Method](#Regression-based-Method)
  - [Span-based Method](#Span-based-Method)
  - [Reinforcement Leaning-based Method](#Reinforcement-Leaning-based-Method)
  - [Other Supervised Method](#Other-Supervised-Method)
  - [Weakly-supervised TSGV Method](#Weakly-supervised-TSGV-Method)
  - - [Multi-Instance Learning Method](#Multi-Instance-Learning-Method)
  - - [Reconstruction-based Method](#Reconstruction-based-Method)
  - - [Other Weakly-supervised Method](#Other-Weakly-supervised-Method)
- [参考](#开源工作)
<!-- - [开源工作](#开源工作) -->


## **数据集**

| **Dataset** | **Video Source** | **Domain** |
|:-----------:|:----------------:|:----------:|
| [**TACoS**](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/vision-and-language/tacos-multi-level-corpus)            |       Kitchen        |   Cooking    |
| [**Charades-STA**](https://prior.allenai.org/projects/charades)            |         Homes         |      Indoor Activity      |
| [**ActivityNet Captions**](http://activity-net.org/download.html)            |      Youtube            |     Open       |
| [**DiDeMo**](<https://github.com/LisaAnne/LocalizingMoments>)            |          Flickr        |     Open       |
|  [**MAD**](https://github.com/Soldelli/MAD)           |      Movie            |    Open        |

## **相关工作**

### **Survey**

- [A survey of temporal activity localization via language in untrimmed videos](https://ieeexplore.ieee.org/abstract/document/9262795). *in ICCST 2020*
- [A survey on natural language video localization](https://arxiv.org/abs/2104.00234). *in ArXiv 2021*
- [A survey on temporal sentence grounding in videos](https://arxiv.org/abs/2109.08039). *in ArXiv 2021*
- [The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions](https://arxiv.org/abs/2201.08071). *in ArXiv 2022*

### **Sliding Window-based Method**
*Sliding window-based method adopts a multi-scale sliding windows (SW) to generate proposal candidates.*
- **CTRL:** [Tall: Temporal activity localization via language query](https://arxiv.org/abs/1705.02101). *in ICCV 2017*. [code](https://github.com/jiyanggao/TALL)
- **MCN:** [Localizing moments in video with natural language](https://arxiv.org/abs/1708.01641). *in ICCV 2017*
- **ROLE:** [Crossmodal moment localization in videos](https://dl.acm.org/doi/10.1145/3240508.3240549). *in ACM MM 2018*
- **ACRN:** [Attentive moment retrieval in videos](https://dl.acm.org/doi/10.1145/3209978.3210003). *in SIGIR 2018*
- **MAC:** [Mac: Mining activity concepts for language-based temporal localization](https://arxiv.org/abs/1811.08925). *in WACV 2019*. [code](https://github.com/runzhouge/MAC)
- **MCF:** [Multi-modal circulant fusion for video-to-language and backward](https://www.ijcai.org/proceedings/2018/0143.pdf). *in IJCAI 2018*
- **MLLC:** [Localizing moments in video with temporal language](https://aclanthology.org/D18-1168.pdf). *in EMNLP 2018*
- **TCMN:** [Exploiting temporal relationships in video moment localization with natural language](https://arxiv.org/abs/1908.03846). *in ACM MM 2019*. [code](https://github.com/Sy-Zhang/TCMN-Release)
- **ASST:** [An attentive sequence to sequence translator for localizing video clips by natural language](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8931634). *in TMM 2020*. [code]( https://github.com/NeonKrypton/ASST)
- **SLTA:** [Cross-modal video moment etrieval with spatial and language-temporal attention](https://dl.acm.org/doi/10.1145/3323873.3325019). *in ICMR 2019*. [code](https://github.com/BonnieHuangxin/SLTA)
- **MMRG:** [Multi-modal relational graph for cross-modal video moment retrieval](https://openaccess.thecvf.com/content/CVPR2021/papers/Zeng_Multi-Modal_Relational_Graph_for_Cross-Modal_Video_Moment_Retrieval_CVPR_2021_paper.pdf). *in CVPR 2021*
- **I$^2$N:** [Interaction-integrated network for natural language moment localization](https://ieeexplore.ieee.org/document/9334438). *in TIP 2021*

### **Proposal Generated Method**
*Proposal generated (PG) method alleviates the computation burden of SW-based methods and generates proposals conditioned on the query.*
- **Text-to-Clip:** [Text-toclip video retrieval with early fusion and re-captioning](http://cs-people.bu.edu/hxu/arxiv-version-Text-to-Clip.pdf). *in ArXiv 2018*
- **QSPN:** [Multilevel language and vision integration for text-to-clip retrieval](https://arxiv.org/abs/1804.05113). *in AAAI 2019*
- **SAP:** [Semantic proposal for activity localization in videos via sentence query](https://ojs.aaai.org/index.php/AAAI/article/view/4830/4703). *in AAAI 2019*
- **BPNet:** [Boundary proposal network for two-stage natural language video localization](https://arxiv.org/abs/2103.08109). *in AAAI 2021*
- **APGN:** [Adaptive proposal generation network for temporal sentence localization in videos](https://aclanthology.org/2021.emnlp-main.732.pdf). *in EMNLP 2021*
- **LP-Net:** [Natural language video localization with learnable moment proposals](https://aclanthology.org/2021.emnlp-main.327.pdf). *in EMNLP 2021*
- **CMHN:** [Video moment localization via deep cross-modal hashing](https://ieeexplore.ieee.org/document/9416231) *in TIP 2021*.

### **Anchor-based Method**
*Anchor-based methods incorporates proposal generation into answer prediction and maintains the proposals with various learning modules.*
#### **Standard Anchor-based Method**
- **TGN:** [Temporally grounding natural sentence in video](https://aclanthology.org/D18-1015.pdf). *in EMNLP 2018*. [code](https://github.com/JaywongWang/TGN)
- **MAN:** [Man: Moment alignment network for natural language moment retrieval via iterative graph adjustmen](https://openaccess.thecvf.com/content_CVPR_2019/papers/Zhang_MAN_Moment_Alignment_Network_for_Natural_Language_Moment_Retrieval_via_CVPR_2019_paper.pdf). *in CVPR 2019*
- **SCDM:** [Semantic conditioned dynamic modulation for temporal sentence grounding in videos](https://proceedings.neurips.cc/paper/2019/file/6883966fd8f918a4aa29be29d2c386fb-Paper.pdf). *in NeurIPS 2019*. [code](https://github.com/yytzsy/SCDM)
- **CMIN:** [Cross-modal interaction networks for query-based moment retrieval in videos](https://arxiv.org/abs/1906.02497). *in SIGIR 2019*. [code](https://github.com/ikuinen/CMIN_moment_retrieval)
- **SCDM$^*$:** [Semantic conditioned dynamic modulation for temporal sentence grounding in videos](https://ieeexplore.ieee.org/document/9263333). *in TPAMI 2020*
- **CMIN$^*$:** [Moment retrieval via cross-modal interaction networks with query reconstruction](https://ieeexplore.ieee.org/document/8962274). *in TIP 2020*
- **CBP:** [Temporally grounding language queries in videos by contextual boundary-aware prediction](https://arxiv.org/abs/1909.05010). *in AAAI 2020*. [code](https://github.com/JaywongWang/CBP)
- **FIAN:** [Finegrained iterative attention network for temporal language localization in videos](https://arxiv.org/abs/2008.02448). *in ACM MM 2020*
- **HDRR:** [Hierarchical deep residual reasoning for temporal moment localization](https://arxiv.org/abs/2111.00417). *ACM MM Asia 2021*
- **MIGCN:** [Multi-modal interaction graph convolutional network for temporal language localization in videos](https://arxiv.org/abs/2110.06058). *in TIP 2021*
- **CSMGAN:** [Jointly cross- and self-modal graph attention network for query-based moment localization](https://arxiv.org/abs/2008.01403). *in ACM MM 2020*
- **RMN:** [Reasoning step-by-step: Temporal sentence localization in videos via deep rectification-modulation network](https://aclanthology.org/2020.coling-main.167.pdf). *in COLING 2020*
- **IA-Net:** [Progressively guide to attend: An iterative alignment framework for temporal sentence grounding](https://aclanthology.org/2021.emnlp-main.733.pdf). *in EMNLP 2021*
- **DCT-Net:** [Dct-net: A deep co-interactive transformer network for video temporal grounding](https://www.sciencedirect.com/science/article/abs/pii/S0262885621000883). *in IVC 2021*

#### **2D-Map Anchor-based Method**
- **2D-TAN:** [Learning 2d temporal adjacent networks formoment localization with natural language](https://arxiv.org/abs/1912.03590). *in AAAI 2020*. [code](https://github.com/microsoft/2D-TAN)
- **MATN:** [Multi-stage aggregated transformer network for temporal language localization in videos](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_Multi-Stage_Aggregated_Transformer_Network_for_Temporal_Language_Localization_in_Videos_CVPR_2021_paper.pdf). *in CVPR 2021*
- **SMIN:** [Structured multi-level interaction network for video moment localization via language query](https://openaccess.thecvf.com/content/CVPR2021/papers/Wang_Structured_Multi-Level_Interaction_Network_for_Video_Moment_Localization_via_Language_CVPR_2021_paper.pdf). *in CVPR 2021*
- **RaNet:** [Relation-aware video reading comprehension for temporal language grounding](https://aclanthology.org/2021.emnlp-main.324.pdf). *in EMNLP 2021*. [code](https://github.com/Huntersxsx/RaNet)
- **FVMR:** [Fast video moment retrieval](https://openaccess.thecvf.com/content/ICCV2021/papers/Gao_Fast_Video_Moment_Retrieval_ICCV_2021_paper.pdf). *in ICCV 2021*
- **MS 2DTAN:** [Multi-scale 2d temporal adjacency networks for moment localization with natural language](https://arxiv.org/abs/2012.02646). *in TPAMI 2021*. [code](https://github.com/microsoft/2D-TAN/tree/ms-2d-tan)
- **PLN:** [Progressive localization networks for language-based moment localization](https://arxiv.org/abs/2102.01282). *in ArXiv 2021*
- **CLEAR:** [Coarseto-fine semantic alignment for cross-modal moment localization](https://liqiangnie.github.io/paper/Coarse-to-Fine%20Semantic%20Alignment%20for%20Cross-Modal%20Moment%20Localization.pdf). *in TIP 2021*
- **STCM-Net:** [Stcm-net: A symmetrical one-stage network for temporal language localization in videos](https://www.sciencedirect.com/science/article/abs/pii/S0925231221016945). *in Neurocomputing 2022*
- **VLG-Net:** [Vlg-net: Videolanguage graph matching network for video grounding](https://openaccess.thecvf.com/content/ICCV2021W/CVEU/papers/Soldan_VLG-Net_Video-Language_Graph_Matching_Network_for_Video_Grounding_ICCVW_2021_paper.pdf). *in ICCV Workshop 2021*
- **SV-VMR:** [Diving into the relations: Leveraging semantic and visual structures for video moment retrieval](https://ieeexplore.ieee.org/document/9428369). *in ICME 2021*
- **MMN:** [Negative sample matters: A renaissance of metric learning for temporal grounding](https://arxiv.org/abs/2109.04872). *in AAAI 2022*. [code](https://github.com/MCG-NJU/MMN)

### **Regression-based Method**
*Regression-based method computes a time pair ($t_s$, $t_e$) and compares the computed pair with ground-truth ($τ_s$, $τ_e$) for model optimization.*
- **ABLR:** [To find where you talk: Temporal sentence localization in video with attention based location regression](https://arxiv.org/abs/1804.07014). *in AAAI 2019*. [code](https://github.com/yytzsy/ABLR_code)
- **ExCL:** [ExCL: Extractive Clip Localization Using Natural Language Descriptions](https://aclanthology.org/N19-1198.pdf). *in NAACL 2019*
- **DEBUG:** [DEBUG: A dense bottomup grounding approach for natural language video localization](https://aclanthology.org/D19-1518.pdf). *in EMNLP 2019*
- **GDP:** [Rethinking the bottom-up framework for query-based video localization](https://ojs.aaai.org/index.php/AAAI/article/view/6627). *in AAAI 2020*
- **CMA::** [A simple yet effective method for video temporal grounding with cross-modality attention](https://arxiv.org/abs/2009.11232). *in ArXiv 2020*
- **DRN:** [Dense regression network for video grounding](https://arxiv.org/abs/2004.03545). *in CVPR 2020*. [code](https://github.com/Alvin-Zeng/DRN)
- **LGI:** [Local-global video-text interactions for temporal grounding](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mun_Local-Global_Video-Text_Interactions_for_Temporal_Grounding_CVPR_2020_paper.pdf). *in CVPR 2020*. [code](https://github.com/JonghwanMun/LGI4temporalgrounding)
- **CPNet:** [Proposal-free video grounding with contextual pyramid network](https://ojs.aaai.org/index.php/AAAI/article/view/16285). *in AAAI 2021*
- **DeNet:** [Embracing uncertainty: Decoupling and de-bias for robust temporal grounding](https://arxiv.org/abs/2103.16848). *in CVPR 2021*
- **SSMN:** [Single-shot semantic matching network for moment localization in videos](https://dl.acm.org/doi/10.1145/3441577?sid=SCITRUS). *in ACM TOMCCAP 2021*
- **HVTG:** [Hierarchical visual-textual graph for temporal activity localization via language](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650596.pdf). *in ECCV 2020*. [code](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123650596.pdf)
- **PMI:** [Learning modality interaction for temporal sentence localization and event captioning in videos](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123490324.pdf). *in ECCV 2020*
- **DRFT:** [End-to-end multi-modal video temporal grounding](https://arxiv.org/abs/2107.05624). *in NeurIPS 2021*

### **Span-based Method**
*Span-based methods aim to predict the probability of each video snippet/frame being the start and end positions of target moment.*
- **ExCL:** [ExCL: Extractive Clip Localization Using Natural Language Descriptions](https://aclanthology.org/N19-1198.pdf). *in NAACL 2019*
- **L-Net:** [Localizing natural language in videos](https://ojs.aaai.org/index.php/AAAI/article/view/4827). *in AAAI 2019*
- **VSLNet:** [Span-based localizing network for natural language video localization](https://aclanthology.org/2020.acl-main.585.pdf). *in ACL 2020*. [code](https://github.com/IsaacChanghau/VSLNet)
- **VSLNet-L$^*$:** [Natural language video localization: A revisit in span-based question answering framework](https://arxiv.org/abs/2102.13558). *in TPAMI 2021*
- **TMLGA:** [Proposal-free temporal moment localization of a natural-language query in video using guided attention](https://arxiv.org/abs/1908.07236). *in WACV 2020*. [code](https://github.com/crodriguezo/TMLGA)
- **SeqPAN:** [Parallel attention network with sequence matching for video grounding](https://arxiv.org/abs/2105.08481). *in Findings of ACL 2021*. [code](https://github.com/IsaacChanghau/SeqPAN)
- **CPN:** [Cascaded prediction network via segment tree for temporal video grounding](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhao_Cascaded_Prediction_Network_via_Segment_Tree_for_Temporal_Video_Grounding_CVPR_2021_paper.pdf). *in CVPR 2021*
- **IVG:** [Interventional video grounding with dual contrastive learning](https://arxiv.org/abs/2106.11013). *in CVPR 2021*. [code](https://github.com/nanguoshun/IVG)
- [Local-enhanced interaction for temporal moment localization](https://dl.acm.org/doi/abs/10.1145/3460426.3463616). *in ICMR 2021*
- **CI-MHA:** [Cross interaction network for natural language guided video moment retrieval](https://assets.amazon.science/3e/b2/355ae2424b088335d5c0a4085e93/cross-interaction-network-for-natural-language-guided-video-moment-retrieval.pdf). *in SIGIR 2021*
- **MQEI:** [Multi-level query interaction for temporal language grounding](https://ieeexplore.ieee.org/document/9543470). *in TITS 2021*
- **ACRM:** [Frame-wise crossmodal matching for video moment retrieval](https://arxiv.org/abs/2009.10434). *in TMM 2021*
- **ABIN:** [Temporal textual localization in video via adversarial bi-directional interaction networks](https://ieeexplore.ieee.org/document/9194326). *in TMM 2021*
- **CSTI:** [Collaborative spatial-temporal interaction for language-based moment retrieval](https://ieeexplore.ieee.org/document/9613549). *in WCSP 2021*
- **DORi:** [Dori: Discovering object relationships for moment localization of a natural language query in a video](https://openaccess.thecvf.com/content/WACV2021/papers/Rodriguez-Opazo_DORi_Discovering_Object_Relationships_for_Moment_Localization_of_a_Natural_WACV_2021_paper.pdf). *in WACV 2021*
- **CBLN:** [Context-aware biaffine localizing network for temporal sentence grounding](https://openaccess.thecvf.com/content/CVPR2021/papers/Liu_Context-Aware_Biaffine_Localizing_Network_for_Temporal_Sentence_Grounding_CVPR_2021_paper.pdf). *in CVPR 2021*. [code](https://github.com/liudaizong/CBLN)
- **PEARL:** [Natural language video moment localization through query-controlled temporal convolution](https://sites.ecse.rpi.edu/~rjradke/papers/zhang-wacv22.pdf). *in WACV 2022*

### **Reinforcement Leaning-based Method**
*RL-based method formulates TSGV as a sequence decision making problem, and utilizes deep reinforcement learning techniques to solve it.*
- **RWM-RL:** [Read, watch, and move: Reinforcement learning for temporally grounding natural language descriptions in videos](https://arxiv.org/abs/1901.06829). *in AAAI 2019*
- **SM-RL:** [Language-driven temporal activity localization: A semantic matching reinforcement learning model](https://openaccess.thecvf.com/content_CVPR_2019/papers/Wang_Language-Driven_Temporal_Activity_Localization_A_Semantic_Matching_Reinforcement_Learning_Model_CVPR_2019_paper.pdf). *in CVPR 2019*
- **TSP-PRL:** [Tree-structured policy based progressive reinforcement learning for temporally language grounding in video](https://arxiv.org/abs/2001.06680). *in AAAI 2020*
- **AVMR:** [Adversarial video moment retrieval by jointly modeling ranking and localization](https://dl.acm.org/doi/10.1145/3394171.3413841). *in ACM MM 2020*
- **STRONG:** [Strong: Spatio-temporal reinforcement learning for cross-modal video moment localization](https://dl.acm.org/doi/10.1145/3394171.3413840). *in ACM MM 2020*
- **TripNet:** [Tripping through time: Efficient localization of activities in videos](https://www.bmvc2020-conference.com/assets/papers/0549.pdf). *in BMVC 2020*
- **MABAN:** [Maban: Multi-agent boundary-aware network for natural language moment retrieval](https://ieeexplore.ieee.org/document/9451629). *in TIP 2021*

### **Other Supervised Method**

- **FIFO:** [Find and focus: Retrieve and localize video events with natural language queries](https://www.ecva.net/papers/eccv_2018/papers_ECCV/papers/Dian_SHAO_Find_and_Focus_ECCV_2018_paper.pdf). *in ECCV 2018.* [code](http://www.xiongyu.me/projects/find_and_focus/)
- **DPIN:** [Dual path interaction network for video moment localization](https://dl.acm.org/doi/abs/10.1145/3394171.3413975). *in ACM MM 2020*
- **Sscs:** [Support-set based cross-supervision for video grounding](https://openaccess.thecvf.com/content/ICCV2021/papers/Ding_Support-Set_Based_Cross-Supervision_for_Video_Grounding_ICCV_2021_paper.pdf). *in ICCV 2021*
- **DepNet:** [Dense events grounding in video](https://ojs.aaai.org/index.php/AAAI/article/view/16175). *in AAAI 2021*
- **SNEAK:** [Sneak: Synonymous sentences-aware adversarial attack on natural language video localization](https://arxiv.org/abs/2112.04154). *in ArXiv 2021.* [code](https://github.com/shiwen1997/SNEAK-CAF2)
- **BSP:** [Boundary-sensitive pre-training for temporal localization in videos](https://openaccess.thecvf.com/content/ICCV2021/papers/Xu_Boundary-Sensitive_Pre-Training_for_Temporal_Localization_in_Videos_ICCV_2021_paper.pdf). *in ICCV 2021.* [code](https://frostinassiky.github.io/bsp)
- **GTR:** [On pursuit of designing multi-modal transformer for video grounding](https://aclanthology.org/2021.emnlp-main.773.pdf). *in EMNLP 2021.* [code](https://sites.google.com/view/mengcao/publication/gtr)

### **Weakly-supervised TSGV Method**
*Under weakly-supervised setting, TSGV methods only need video-query pairs but not the annotations of starting/end time.*
#### **Multi-Instance Learning Method**
- **TGA:** [Weakly supervised video moment retrieval from text queries](https://openaccess.thecvf.com/content_CVPR_2019/papers/Mithun_Weakly_Supervised_Video_Moment_Retrieval_From_Text_Queries_CVPR_2019_paper.pdf). *in CVPR 2019*
- **WSLLN:** [WSLLN:weakly supervised natural language localization networks](https://aclanthology.org/D19-1157.pdf). *in EMNLP 2019*
- **Coarse-to-Fine:** [Look closer to ground better: Weakly-supervised temporal grounding of sentence in video](https://arxiv.org/abs/2001.09308). *in ArXiv 2020*
- **VLANet:** [Vlanet: Video-language alignment network for weakly-supervised video moment retrieva](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123730154.pdf). *in ECCV 2020*
- **BAR:** [Reinforcement learning for weakly supervised temporal grounding of natural language in untrimmed videos](https://arxiv.org/abs/2009.08614). *in ACM MM 2020*
- **CCL:** [Counterfactual contrastive learning for weakly-supervised vision-language grounding](https://papers.nips.cc/paper/2020/file/d27b95cac4c27feb850aaa4070cc4675-Paper.pdf). *in NeurIPS 2020*
- **AsyNCE:** [Asynce: Disentangling false-positives for weakly-supervised video grounding](https://dl.acm.org/doi/10.1145/3474085.3481539). *in ACM MM 2021*
- [Visual co-occurrence alignment learning for weakly-supervised video moment retrieval](https://dl.acm.org/doi/10.1145/3474085.3475278). *in ACM MM 2021*
- **FSAN:** [Fine-grained semantic alignment network for weakly supervised temporal language grounding](https://aclanthology.org/2021.findings-emnlp.9.pdf). *in Findings of EMNLP 2021*
- **CRM:** [Cross-sentence temporal and semantic relations in video activity localisation](https://openaccess.thecvf.com/content/ICCV2021/papers/Huang_Cross-Sentence_Temporal_and_Semantic_Relations_in_Video_Activity_Localisation_ICCV_2021_paper.pdf). *in ICCV 2021*
- **LCNet:** [Local correspondence network for weakly supervised temporal sentence grounding](https://ieeexplore.ieee.org/document/9356448). *in TIP 2021*
- [Regularized two granularity loss function for weakly supervised video moment retrieval](https://ieeexplore.ieee.org/document/9580967). *in TMM 2021*
- **WSTAN:** [Weakly supervised temporal adjacent network for language grounding](https://arxiv.org/abs/2106.16136). *in TMM 2021*
- **LoGAN:** [Logan: Latent graph co-attention network for weakly-supervised video moment retrieval](https://arxiv.org/abs/1909.13784). *in WACV 2021*

#### **Reconstruction-based Method**
- [Weakly supervised dense event captioning in videos](https://proceedings.neurips.cc/paper/2018/file/49af6c4e558a7569d80eee2e035e2bd7-Paper.pdf). *in NeurIPS 2018*
- **SCN:** [Weakly-supervised video moment retrieval via semantic completion network](https://arxiv.org/abs/1911.08199). *in AAAI 2020*
- **EC-SL:** [Towards bridging event captioner and sentence localizer for weakly supervised dense event captioning](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Towards_Bridging_Event_Captioner_and_Sentence_Localizer_for_Weakly_Supervised_CVPR_2021_paper.pdf). *in CVPR 2021*
- **MARN:** [Weakly-supervised multilevel attentional reconstruction network for grounding textual queries in videos](https://arxiv.org/abs/2003.07048). *in ArXiv 2020*
- [Towards bridging video and language by caption generation and sentence localization](https://dl.acm.org/doi/abs/10.1145/3474085.3481032). *in ACM MM 2021*

#### **Other Weakly-supervised Method**
- **RTBPN:** [Regularized two-branch proposal networks for weakly-supervised moment retrieval in videos](https://arxiv.org/abs/2008.08257). *in ACM MM 2020*. [code](https://github.com/ikuinen/regularized_two-branch_proposal_network)
- **S$^4$TLG:** [Self-supervised learning for semi-supervised temporal language grounding](https://arxiv.org/abs/2109.11475). *in ArXiv 2021*
- **PSVL:** [Zero-shot natural language video localization](https://openaccess.thecvf.com/content/ICCV2021/papers/Nam_Zero-Shot_Natural_Language_Video_Localization_ICCV_2021_paper.pdf). *in ICCV 2021*. [code](https://github.com/gistvision/PSVL)
- [Learning video moment retrieval without a single annotated video](https://ieeexplore.ieee.org/document/9415694). *in TCSVT 2021*


## 参考
[Survey of Zhang *et al*](https://arxiv.org/pdf/2201.08071.pdf)
