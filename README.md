# DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning
Codes for the paper: [DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning](https://arxiv.org/abs/2407.18523).


## Datasets

You can process your datasets with the codes from [DyGLib](https://github.com/yule-BUAA/DyGLib/tree/master/preprocess_data) or download our pre-processed data from [Kaggle](https://www.kaggle.com/datasets/chenxi1228/datasets-for-dtformer).

Please put the processed data in ```processed_data``` folder.

## Training
* Example of training on *Bitcoin-OTC* dataset:
```{bash}
python train_link_prediction.py --dataset_name bitcoinotc --using_snapshot_feat --using_snap_counts --using_intersect_feat --intersect_mode sum --num_patch_size 3 --patch_size 8 --max_input_sequence_length 256 --num_runs 5 --gpu 0
```

You can choose not to use the Neighbor Positional Feature, Neighbor Occurrence Feature, or Neighbor Intersect Feature by removing ``--using_snapshot_feat``, ``--using_snap_counts``, or ``--using_intersect_feat``, respectively.

You can use ``--intersect_mode`` to change the modes for modeling the Neighbor Intersect Feature (you can choose from ``gru``, ``sum``, and ``mlp``).

``--num_patch_size`` is used to control the multi-patching module, please read our paper for more detail.

## Acknowledgments
This repository is based on the code from [DyGLib](https://github.com/yule-BUAA/DyGLib).

We thank the author for making their code open-source.


## Citation

Please consider citing our paper when using this project.

```{bibtex}
@inproceedings{chen2024dtformer,
  title={DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning},
  author={Chen, Xi and Xiong, Yun and Zhang, Siwei and Zhang, Jiawei and Zhang, Yao and Zhou, Shiyang and Wu, Xixi and Zhang, Mingyang and Liu, Tengfei and Wang, Weiqiang},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={301--311},
  year={2024}
}
```
or

```{bibtex}
@article{chen2024dtformer,
  title={DTFormer: A Transformer-Based Method for Discrete-Time Dynamic Graph Representation Learning},
  author={Chen, Xi and Xiong, Yun and Zhang, Siwei and Zhang, Jiawei and Zhang, Yao and Zhou, Shiyang and Wu, Xixi and Zhang, Mingyang and Liu, Tengfei and Wang, Weiqiang},
  journal={arXiv preprint arXiv:2407.18523},
  year={2024}
}
```
