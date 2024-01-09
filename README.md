# [[NeurIPS 2023] Content-based Unrestricted Adversarial Attack](https://openreview.net/pdf?id=gO60SSGOMy)

Zhaoyu Chen, Bo Li, Shuang Wu, Kaixun Jiang, Shouhong Ding, Wenqiang Zhang

This repository offers Pytorch code to reproduce results from the paper. Please consider citing our paper if you find it interesting or helpful to your research.

```
@inproceedings{
chen2023contentbased,
title={Content-based Unrestricted Adversarial Attack},
author={Zhaoyu Chen and Bo Li and Shuang Wu and Kaixun Jiang and Shouhong Ding and Wenqiang Zhang},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
year={2023},
url={https://openreview.net/forum?id=gO60SSGOMy}
}
```


## Requirements

- Python == 3.8.0
- Pytorch == 1.12.1
- torchvision == 0.13.1
- CUDA == 11.3
- timm == 0.9.5
- TensorFlow == 2.11.0
- diffuseers == 0.3.0
- huggingface-hub == 0.11.1
- pyiqa == 0.1.6.3


## Quick Start

- **Prepare models**

  Download the pretrained models and their checkpoints.

   ```bash
  cd checkpoints
  ./download.sh
  ```

- **Prepare datasets**

  We obtain the datasets from Natural-Color-Fool.

   ```bash
  cd third_party
  ./download.sh
  ```
  
- **Generate the prompts**

  Here, we use BLIP-v2 to automatically generate corresponding prompts.

  ```bash
  TRANSFORMERS_OFFLINE=1 python3 generate_prompts.py
  ```

- **Image latent mapping**

    We use null-text embedding to map images into the latent space.

  ```bash
  python3 image_latent_mapping.py
  ```

- **Adversarial latent optimization**

   After the latent is processed offline, we perform latent optimization to obtain adversarial examples.

  ```bash
  CUDA_VISIBLE_DEVICES=0 python3 adversarial_latent_optimization.py --model mnv2 --beta 0.1 --alpha 0.04 --steps 10 --norm 2 --start 0 --end 1000 --mu 1 --eps 0.1 
  ```

- **Evaluate the accuracy**

   Infer the model with images.

  ```bash
  python3 test_model.py --model MODEL_NAME --img_path IMAGE_SAVE_PATH
  ```

- **Evaluate the image quality**

   Test the image quality with images.

  ```bash
  python3 test_quality.py --metric METRIC --img_path IMAGE_SAVE_PATH
  ```

## License
The project is only free for academic research purposes but has no authorization for commerce. Part of the code is modified from Prompt-to-Prompt.