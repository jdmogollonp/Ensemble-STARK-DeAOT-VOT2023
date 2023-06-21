# Ensemble-Mixformer-STARK-VOT2023

## Ensemble Different Trackers to Make a Robust Single and Multi-Object Tracking

The tracker is an ensemble algorithm that adapts to different scenarios based on the number of objects in the video sequences. For scenarios with a small number of objects, it employs the MixFormer tracker. MixFormer predicts the bounding boxes of the objects, providing accurate position estimations. It is coupled with the Segment Anything Model (SAM), which generates segmentation masks based on the predicted bounding boxes, ensuring precise object identification. In situations with large objects present, the ensemble tracker switches to the STARK tracker. STARK predicts the bounding boxes of the objects, providing accurate position estimations. It is coupled with the Segment Anything Model (SAM), which generates segmentation masks based on the predicted bounding boxes, ensuring precise object identification.. By utilizing the ensemble method and incorporating MixFormer and STARK, the tracker ensures robust and accurate object tracking across various scenarios. It provides precise position estimation and segmentation masks for both single and multiple object tracking tasks, enhancing the overall tracking performance.

</br>

## VOT Running

To run it using VOT package, change

```
command = ensemble_mixformer_stark
```
and give path accordingly in 
```
trackers.ini
```

</br>

## :full_moon_with_face:Credits

* MixFormer - [https://github.com/MCG-NJU/MixFormer](https://github.com/MCG-NJU/MixFormer)
* STARK - [https://github.com/researchmm/Stark](https://github.com/researchmm/Stark)
* SAM - [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)

</br>

## Citations
Please consider citing the related paper(s) in your publications if it helps your research.
```
@inproceedings {yan2021stark,
    author = {B. Yan and H. Peng and J. Fu and D. Wang and H. Lu},
    booktitle = {2021 IEEE/CVF International Conference on Computer Vision (ICCV)},
    title = {Learning Spatio-Temporal Transformer for Visual Tracking},
    publisher = {IEEE Computer Society},
    pages = {10428-10437},
    year = {2021}
}
@inproceedings{cui2022mixformer,
  title={Mixformer: End-to-end tracking with iterative mixed attention},
  author={Cui, Yutao and Jiang, Cheng and Wang, Limin and Wu, Gangshan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13608--13618},
  year={2022}
}
@misc{cui2023mixformer,
      title={MixFormer: End-to-End Tracking with Iterative Mixed Attention}, 
      author={Yutao Cui and Cheng Jiang and Gangshan Wu and Limin Wang},
      year={2023},
      eprint={2302.02814},
      archivePrefix={arXiv}
}
@article{kirillov2023segment,
  title={Segment anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C and Lo, Wan-Yen and others},
  journal={arXiv preprint arXiv:2304.02643},
  year={2023}
}

```