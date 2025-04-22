from controlnet_aux import OpenposeDetector
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
openpose_pre_train_path = r"lllyasviel/sd-controlnet-openpose"
openpose = OpenposeDetector.from_pretrained(r'lllyasviel/ControlNet').to("cuda")
# o_image = load_image(r"D:\lqh12\a-sd-based-models\sd-controlnet-openpose\images\pose.png")

controlnet = ControlNetModel.from_pretrained(openpose_pre_train_path, torch_dtype=torch.float16).to("cuda")

# pipe_t = StableDiffusionControlNetPipeline.from_pretrained(
#     r"KBlueLeaf/kohaku-v2.1", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
# ).to("cuda")
