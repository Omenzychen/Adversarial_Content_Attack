import os
import torch
from PIL import Image
from lavis.models import load_model_and_preprocess


'''
TRANSFORMERS_OFFLINE=1 python3 generate_prompts.py
'''


if __name__ == '__main__':
    image_nums = 1000
    save_path = 'temp/1000'
    all_prompts = [ "" for i in range(1000)]

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device=device)
    # # load sample image
    img_path = 'third_party/Natural-Color-Fool/dataset/images'
    img_list = os.listdir(img_path)
    for i in range(image_nums):
        path = os.path.join(img_path, img_list[i])
        idx = int(img_list[i].split('.')[0]) - 1
        raw_image = Image.open(path).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        ans = model.generate({"image": image, "prompt": "Question: Please give a detailed description of the image. Answer:"})

        print(path, idx, ans)
        all_prompts[idx] = ans[0]
        

    with open(os.path.join(save_path, 'prompts.txt'), 'w') as f:
        for i in range(len(all_prompts)):
            f.write(all_prompts[i]+ '\n')