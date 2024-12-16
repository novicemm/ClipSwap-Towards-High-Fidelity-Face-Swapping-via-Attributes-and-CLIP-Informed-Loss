# ClipSwap: Towards High Fidelity Face Swapping via Attributes and CLIP-Informed Loss (FG 2024 Oral)

Phyo Thet Yee, Sudeepta Mishra, Abhinav Dhall
<br><br>

We present ClipSwap, a new framework designed for high-fidelity face swapping. We use a conditional Generative Adversarial Network and a CLIP-based encoder, which extracts rich semantic knowledge to achieve attributes-aware face swapping. Our framework uses CLIP embedding in the face swapping process for improving the transmission of source image’s identity details to the swapped image by refining the high-level semantic attributes obtained from the source image. And source image serves as the input reference image for CLIP and ensures a more accurate and detailed identity representation in the final result. Additionally, we apply Contrastive Loss to guide the transformation of source facial attributes onto the swapped image from various viewpoints. We also introduce Attributes Preservation Loss, which penalizes the network to keep the facial attributes of the target image.

<br><br>
![clipswap](https://github.com/novicemm/ClipSwap-Towards-High-Fidelity-Face-Swapping-via-Attributes-and-CLIP-Informed-Loss-FG-2024-Oral-/assets/42999480/d034e8cb-6ad3-4f09-92cf-c3d4127cc610)

<br><br>
**Model Architecture**

![framework](https://github.com/novicemm/ClipSwap-Towards-High-Fidelity-Face-Swapping-via-Attributes-and-CLIP-Informed-Loss-FG-2024-Oral-/assets/42999480/ab994791-5df1-4b68-a30d-050cdff3d6e3)

<br><br>
**Results**

![results](https://github.com/novicemm/ClipSwap-Towards-High-Fidelity-Face-Swapping-via-Attributes-and-CLIP-Informed-Loss-FG-2024-Oral-/assets/42999480/c237a03c-8ab6-4d08-a1ae-63c51ef2ccf4)

<br><br>
**Wild Image Results**

![wild_images](https://github.com/novicemm/ClipSwap-Towards-High-Fidelity-Face-Swapping-via-Attributes-and-CLIP-Informed-Loss-FG-2024-Oral-/assets/42999480/96859be5-bacb-478a-bf00-f84a21e65629)

<br><br>
**Installation**

Clone the code and set up the environment.

```bash
conda env create -f environment.yaml

```
**Download Pretrained Models**

Download the pretrained checkpoints folder, which include the following parts: RetinaFace-Res50.h5, ArcFace-Res50.h5, and our model ClipSwap.h5, from [here](https://drive.google.com/drive/folders/1WWRTGQy-cx9QxMsF_etrEnWIvqxutlVX?usp=drive_link).  After downloading, place the models in the project folder. 

The pretrained models should be organized as follows:

```
./pretrained_ckpts/
|-- arcface
|  |-- ArcFace-Res50.h5
|-- retinaface
|  |-- RetinaFace-Res50.h5
|-- clipswap
|  |-- ClipSwap.h5

```

**Inference**

To run the inference, you can either execute the provided bash script or run the Python script directly:

1. Using the bash script
   
```bash
sh inference.sh
```
2. Using the python script

```bash
python inference.py --target /path/to/target_image.jpg --source /path/to/source_image.jpg --output ./output/result.jpg
```
Make sure to replace /path/to/target_image.jpg and /path/to/source_image.jpg with the actual paths to your input images.

**Acknowledgements**

We borrow some code from [insightface](https://github.com/deepinsight/insightface) and [FaceDancer](https://github.com/felixrosberg/FaceDancer).

**Citation**

If you find our work useful in your research, please consider citing us:

```bash
@inproceedings{yee2024clipswap,
  title={ClipSwap: Towards High Fidelity Face Swapping via Attributes and CLIP-Informed Loss},
  author={Yee, Phyo Thet and Mishra, Sudeepta and Dhall, Abhinav},
  booktitle={2024 IEEE 18th International Conference on Automatic Face and Gesture Recognition (FG)},
  pages={1--10},
  year={2024},
  organization={IEEE}
}
```

