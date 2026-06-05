# CADF-Net:Efficient Image Super-Resolution Network Based on Correlation Spatial Attention and Dynamic Fusion

**The official repository with Pytorch**

## Installation

**Clone this repo:**

```bash
git clone git@github.com:ang-plus/CADF-Net.git
cd CDSNet
```

**Dependencies:**

- PyTorch>1.10
- OpenCV
- Matplotlib 3.3.4
- opencv-python
- pyyaml
- tqdm
- numpy
- torchvision

## Preparation

- Download pretrained models, and copy them to `./train_logs/`:

| Settings         | CKPT name           | CKPT url                                                                                                                                                                            |
| ---------------- | ------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DIV2K $\times 2$ | CDSNet_X2_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/1dJhTlhloaiYn9yImk6pa1Q) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/18lSvJq9CGCwDomkas2gh8K6UOq8qRLIw/view?usp=sharing) |
| DF2K $\times 2$  | CDSNet_X2_DF2K.zip  | [baidu cloud](https://pan.baidu.com/s/1IK_bzB5gp2tK67zF-VV4Lg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/12EvHRof0-kA2Wt_BzfFJBK1J0jbzfz-4/view?usp=sharing) |
| DIV2K $\times 3$ | CDSNet_X3_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/19J5uONEOYWxAbEMWIF9qDA) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1Rwg6o-RGC-TEiyVSVT9FS1iHjx5n948h/view?usp=sharing) |
| DF2K $\times 3$  | CDSNet_X3_DF2K.zip  | [baidu cloud](https://pan.baidu.com/s/1mXL7AOUwyC91UDcEWFCh2Q) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/198R2c3nlyhL4FxMJSC_gccyL3O1gH_K6/view?usp=sharing) |
| DIV2K $\times 4$ | CDSNet_X4_DIV2K.zip | [baidu cloud](https://pan.baidu.com/s/1kGasS_wslZy4OyzaHTukvg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1VoPUw0SRnCPAU8_R5Ue15bn2gwSBr97g/view?usp=sharing) |
| DF2K $\times 4$  | CDSNet_X4_DF2K.zip  | [baidu cloud](https://pan.baidu.com/s/1ovxRa4-wOKZLq_nO6hddsg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/17rJXJHBYt4Su8cMDMh-NOWMBdE6ki5em/view?usp=sharing) |

- Download benchmark ([baidu cloud](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), and copy them to `./benchmark/`. If you want to generate the benchmark by yourself, please refer to the official repository of [RCAN](https://github.com/yulunzhang/RCAN).

## Evaluate Pretrained Models

### Example: evaluate the model trained with DF2K@X4:

- Step 1, the following cmd will report a performance evaluated with python script, and generated images are placed in `./SR`

```
python test.py -v "CDSNet_X4_DF2K" -s 1000 -t tester_Matlab --test_dataset_name "Urban100"
```

- Step2, please execute the `Evaluate_PSNR_SSIM.m` script in the root directory to obtain the results reported in the paper. Please modify `Line 8 (Evaluate_PSNR_SSIM.m): methods = {'CDSNet_X4_DF2K'};` and `Line 10 (Evaluate_PSNR_SSIM.m): dataset = {'Urban100'};` to match the model/dataset name evaluated above.

## Training

- Step1, please download training dataset from [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (`Train Data Track 1 bicubic downscaling x? (LR images)` and `Train Data (HR images)`), then set the dataset root path in `./env/env.json: Line 8: "DIV2K":"TO YOUR DIV2K ROOT PATH"`

- Step2, please download benchmark ([baidu cloud](https://pan.baidu.com/s/1HsMtfjEzj4cztaF2sbnOMg) (passwd: sjtu) , [Google driver](https://drive.google.com/file/d/1w-brbpprWHyT4tzCe_MoB2tqEcSOc5OW/view?usp=sharing)), and copy them to `./benchmark/`. If you want to generate the benchmark by yourself, please refer to the official repository of [RCAN](https://github.com/yulunzhang/RCAN).

- Step3, training with DIV2K $\times 4$ dataset:

```
python train.py -v "CDSNet_X4_DIV2K" -p train --train_yaml "train_CDSNet_X4_DIV2K.yaml"
```

## Visualization

![performance](./doc/imgs/对比Donku1.tif)

## Results
Method	Scale	Set5	Set14	BSD100	Urban100	Manga109
		PSNR	SSIM	PSNR	SSIM	PSNR	SSIM	PSNR	SSIM	PSNR	SSIM
RCAN[8]	



X2	38.07	0.9614	34.02	0.9216	32.17	0.9010	32.76	0.9310	39.01	0.9211
TDPN[37]		38.31	0.9621	34.16	0.9225	32.32	0.9025	33.36	0.9386	39.57	0.9784
ELAN[38]		38.17	0.9611	33.94	0.9207	32.30	0.9012	32.76	0.9340	39.11	0.9782
MambaIR[19]		38.24	0.9625	34.38	0.9257	32.42	0.9023	34.15	0.9426	39.88	0.9763
SwinIR[9]		38.27	0.9617	34.26	0.9250	32.40	0.9024	33.16	0.9357	39.44	0.9786
MAT[39]		38.32	0.9610	34.41	0.9255	32.44	0.9011	34.18	0.9451	39.75	0.9769
SRFormer[12]		38.30	0.9622	34.33	0.9258	32.42	0.9028	34.20	0.9440	39.80	0.9785
ATD[11]		38.33	0.9619	34.47	0.9253	32.39	0.9011	34.44	0.9455	39.91	0.9777
Ours		38.37	0.9624	34.50	0.9260	32.44	0.9026	34.50	0.9456	39.98	0.9786
RCAN[8]	



X3


	34.24	0.9209	29.86	0.8371	28.60	0.8022	27.68	0.8536	32.97	0.9423
TDPN[37]		34.74	0.9312	30.71	0.8501	29.34	0.8126	29.26	0.8724	34.48	0.9508
ELAN[38]		34.80	0.9313	34.70	0.8504	29.28	0.8124	29.32	0.8754	34.73	0.9517
MambaIR[19]		34.76	0.9323	30.75	0.8546	29.32	0.8126	29.48	0.8847	35.20	0.9546
SwinIR[9]		34.48	0.9278	30.38	0.8434	29.10	0.8063	28.47	0.8596	33.81	0.9464
MAT[39]		34.78	0.9320	30.69	0.8544	29.22	0.8160	29.80	0.8868	35.14	0.9539
SRFormer[12]		34.70	0.9318	30.70	0.8504	29.28	0.8124	29.59	0.8786	35.15	0.9550
ATD[11]		34.77	0.9330	30.76	0.8556	29.35	0.8136	29.76	0.8897	35.20	0.9551
Ours		34.85	0.9334	30.76	0.8564	29.39	0.8140	29.88	0.8902	35.25	0.9558
RCAN[8]	



X4


	32.63	0.9002	28.87	0.7889	27.77	0.7463	26.82	0.8087	31.22	0.9173
TDPN[37]		32.69	0.9005	29.01	0.7943	27.93	0.7460	27.24	0.8171	31.58	0.9218
ELAN[38]		32.75	0.9022	28.96	0.7914	27.83	0.7459	27.13	0.8167	31.68	0.9226
MambaIR[19]		32.96	0.9044	29.21	0.7961	27.98	0.7503	27.68	0.8287	32.32	0.9272
SwinIR[9]		32.72	0.9001	28.92	0.7918	27.73	0.7456	27.10	0.8185	31.69	0.9326
MAT[39]		32.33	0.8962	29.10	0.7949	27.90	0.7510	27.71	0.8329	32.30	0.9282
SRFormer[12]		32.96	0.9033	29.08	0.7953	27.94	0.7502	27.68	0.8311	32.21	0.9271
ATD[11]		32.86	0.9028	29.07	0.7943	27.81	0.7493	27.84	0.8334	32.29	0.9317
Ours		32.98	0.9051	29.11	0.7964	27.95	0.7520	27.87	0.8344	32.39	0.9327

## Related Projects

## License

This project is released under the Apache 2.0 license.

## To cite our paper

If this work helps your research, please cite the following paper:

```
@inproceedings{cdsnet,
  title      = {Cross-Domain Self-allocation Attention Network},
  author     = {Wang, Hang and Chen, Xuanhong and Ni, Bingbing and Liu, Yutian and Liu jinfan},
  booktitle  = {Conference on Computer Vision and Pattern Recognition},
  year       = {2023}
}
```
