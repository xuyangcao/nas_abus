# NAS ABUS
NAS based segmentation method on ABUS tumor segmentation.

# Useage

## search 

python train_search.py --epoch 600 --save `save_path` --batch_size 2 --gpu 1 --growth_rate 16 --num_init_features 64

## train

python train.py --epoch 600 --batch_size 2 --dataset abus3d --save `save_path` --gpu 1 --growth_rate 48 --num_init_features 64 --threshold 0.5


## test 

python test.py --resume `check_poing_path`

# Reference

Y. Ji, R. Zhang, Z. Li, J. Ren, S. Zhang, and P. Luo, “UXNet: Searching Multi-level Feature Aggregation for 3D Medical Image Segmentation,” in Medical Image Computing and Computer Assisted Intervention – MICCAI 2020, Cham, 2020, pp. 346–356, doi: 10.1007/978-3-030-59710-8_34.

代码：无

W. Chen, X. Gong, X. Liu, Q. Zhang, Y. Li, and Z. Wang, “FasterSeg: Searching for Faster Real-time Semantic Segmentation,” arXiv:1912.10917 [cs], Jan. 2020, Accessed: Sep. 25, 2020. [Online]. Available: http://arxiv.org/abs/1912.10917.

代码：https://github.com/VITA-Group/FasterSeg

C. Liu et al., “Auto-DeepLab: Hierarchical Neural Architecture Search for Semantic Image Segmentation,” in 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Long Beach, CA, USA, Jun. 2019, pp. 82–92, doi: 10.1109/CVPR.2019.00017.

代码：https://github.com/MenghaoGuo/AutoDeeplab

H. Liu, K. Simonyan, and Y. Yang, “DARTS: Differentiable Architecture Search,” arXiv:1806.09055 [cs, stat], Apr. 2019, Accessed: Feb. 04, 2021. [Online]. Available: http://arxiv.org/abs/1806.09055.

代码：https://github.com/quark0/darts

H. Cai, L. Zhu, and S. Han, “ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware,” arXiv:1812.00332 [cs, stat], Feb. 2019, Accessed: Oct. 05, 2020. [Online]. Available: http://arxiv.org/abs/1812.00332.

代码：https://github.com/mit-han-lab/proxylessnas 

Weng, Yu et al. “NAS-Unet: Neural Architecture Search for Medical Image Segmentation.” IEEE Access 7 (2019): 44247-44257.

代码：https://github.com/tianbaochou/NasUnet
