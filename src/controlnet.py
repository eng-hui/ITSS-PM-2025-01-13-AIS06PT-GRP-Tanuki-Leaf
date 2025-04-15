from controlnet_aux import OpenposeDetector
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
openpose_pre_train_path = r"lllyasviel/sd-controlnet-openpose"
openpose = OpenposeDetector.from_pretrained(r'lllyasviel/ControlNet').to("cuda")
# o_image = load_image(r"D:\lqh12\a-sd-based-models\sd-controlnet-openpose\images\pose.png")

controlnet = ControlNetModel.from_pretrained(openpose_pre_train_path, torch_dtype=torch.float16).to("cuda")

pipe_t = StableDiffusionControlNetPipeline.from_pretrained(
    r"KBlueLeaf/kohaku-v2.1", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16
).to("cuda")

def get_a_b_control_net(a, b, o_image):
    print(f"hhhhhhhhhhhhhhhhhhhhhhh:{o_image.shape}")
    o_image = o_image[0].permute(1, 2, 0)
    image = pipe_t.prepare_image(    
            image=openpose(o_image.cpu()),
            width=512,
            height=512,
            batch_size=1 * 1,
            num_images_per_prompt=1,
            device="cuda",
            dtype=controlnet.dtype,
            do_classifier_free_guidance=False,
            guess_mode=False,            
        )
    
    down_block_res_samples, mid_block_res_sample = controlnet(
        a,
        801,
        encoder_hidden_states=b,
        controlnet_cond=image,
        conditioning_scale=1.0,
        guess_mode=False,
        return_dict=False,
    )
    return down_block_res_samples, mid_block_res_sample