import gc
import os
import shutil
import random
import folder_paths
import torch

import comfy
import node_helpers
import comfy.model_management as mm
import comfy.sample
import comfy.samplers
import comfy_extras.nodes_model_advanced

from nodes import CLIPVisionEncode, CLIPTextEncode, VAEDecode, VAEDecodeTiled
from comfy.sd import load_text_encoder_state_dicts, CLIPType, VAE
from comfy.utils import common_upscale, load_torch_file
from comfy.clip_vision import load_clipvision_from_sd
from node_helpers import conditioning_set_values
from .latent_preview import prepare_callback, get_previewer
from .model_patcher.teacache import patch_teacache
from .model_patcher.patch import patch_cfg_zero_star, patch_enhance_video, skip_layer_guidance, CFGGuider2
from .model_patcher.optimization import patch_sage_attention, patch_model_order, torch_compile_model
from .dataclass import Config
from comfy_extras.nodes_custom_sampler import Noise_EmptyNoise, Noise_RandomNoise
from .gguf.nodes import load_gguf

from spandrel import ModelLoader, ImageModelDescriptor
from torch.hub import download_url_to_file

REPO_ID_COMFYORG = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
REPO_ID_KIJAI = "Kijai/WanVideo_comfy"
REPO_ID_MODELS = {}

def add_model_list_from_huggingface(repo_id, filter):
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id)
        for file in files:
            if filter in file:
                file_name = file.rsplit("/", 1)[-1]
                REPO_ID_MODELS[file_name] = repo_id
    except Exception as e:
        print(f"Failed to fetch from {repo_id}: {e}")

try:
    add_model_list_from_huggingface("city96/Wan2.1-I2V-14B-480P-gguf", ".gguf")
    add_model_list_from_huggingface("city96/Wan2.1-I2V-14B-720P-gguf", ".gguf")
    add_model_list_from_huggingface("city96/Wan2.1-T2V-14B-gguf", ".gguf")
    add_model_list_from_huggingface(REPO_ID_COMFYORG, "diffusion_models")
    # add_model_list_from_huggingface("city96/Wan2.1-Fun-14B-InP-gguf", ".gguf")
    # add_model_list_from_huggingface("city96/Wan2.1-Fun-14B-Control-gguf", ".gguf")

    if not REPO_ID_MODELS:
        raise ValueError("Model list is empty.")

except Exception as e:
    print(f"Using fallback due to error: {e}")
    REPO_ID_MODELS = folder_paths.get_filename_list("diffusion_models")

CLIP_NAME = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
CLIP_VISION_NAME = "clip_vision_h.safetensors"
VAE_NAME = "wan_2.1_vae.safetensors"
TAEHV_NAME = "taew2_1.safetensors"


intermediate_device = mm.intermediate_device()
offload_device = mm.unet_offload_device()
torch_device = mm.get_torch_device()

def convert_filename_comfyorg(model_type, model_name):
    return f"split_files/{model_type}/{model_name}", model_type

def download_huggingface_model(repo_id, file_name, saved_directory):
    model_name = file_name.rsplit('/', 1)[-1]

    saved_directory = os.path.join(folder_paths.models_dir, saved_directory)
    saved_full_path = os.path.join(saved_directory, model_name)
    if os.path.exists(saved_full_path):
        print(f"exists model: {saved_full_path}")
        return saved_full_path
    
    print(f"Downloading the {model_name} model. Please wait a moment...")

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo_id,
        local_dir=saved_directory,
        allow_patterns=file_name,
    )

    downloaded_path = os.path.join(saved_directory, file_name)
    if downloaded_path != saved_full_path and os.path.exists(downloaded_path):
        shutil.move(downloaded_path, saved_full_path)
        print(f"File moved to: {saved_full_path}, check exists: {os.path.exists(saved_full_path)}")

    return saved_full_path

def download_github_model(repo_id, tag, file_name, saved_directory):
    download_url = f"https://github.com/{repo_id}/releases/download/{tag}/{file_name}/"

    saved_directory = os.path.join(folder_paths.models_dir, saved_directory)
    saved_full_path = os.path.join(saved_directory, file_name)
    if os.path.exists(saved_full_path):
        print(f"exists model: {saved_full_path}")
        return saved_full_path
    else:
        os.makedirs(saved_directory, exist_ok=True)
    
    print(f"Downloading the {file_name} model. Please wait a moment...")

    download_url_to_file(download_url, saved_full_path, hash_prefix=None, progress=True)

    return saved_full_path

class WanVideoModelLoader_F2:
    encoder_models = {
        "clip": None,
        "clip_vision": None,
        "vae": None,
        "taehv": None,
    }
    loaded_model = None
    loaded_loras = {
        "lora_1": None,
        "lora_2": None,
        "lora_3": None,
    }
    def __init__(self):
        self.loaded_model = None
        self.downloaded_taehv = False

    @classmethod
    def INPUT_TYPES(s):
        lora_files = ["disabled"] + folder_paths.get_filename_list("loras")
        return {
            "required":{
                "unet_name": ([
                    k for k in REPO_ID_MODELS.keys()
                ], ),
            "lora_1": (lora_files, {"advanced": True}), "lora_1_strength": ("FLOAT", {"default": 1.00, "min": -10.00, "max": 10.00, "step":0.01, "round": 0.01, "advanced": True}),
            "lora_2": (lora_files, {"advanced": True}), "lora_2_strength": ("FLOAT", {"default": 1.00, "min": -10.00, "max": 10.00, "step":0.01, "round": 0.01, "advanced": True}),
            "lora_3": (lora_files, {"advanced": True}), "lora_3_strength": ("FLOAT", {"default": 1.00, "min": -10.00, "max": 10.00, "step":0.01, "round": 0.01, "advanced": True}),
            },
        }
       
    RETURN_TYPES = ("MODEL", )
    FUNCTION = "load"

    CATEGORY = "Flow2/Wan 2.1"

    @classmethod
    def get_clip(cls):
        if cls.encoder_models["clip"] is None:
            path = download_huggingface_model(REPO_ID_COMFYORG, *convert_filename_comfyorg("text_encoders", CLIP_NAME))
            state_dicts = load_torch_file(path, safe_load=True)
            clip = load_text_encoder_state_dicts(
                state_dicts=[state_dicts],
                clip_type=CLIPType.WAN,
                model_options={"load_device": torch_device, "offload_device": offload_device} #if clip_offload == "cpu" else {}
            )
            del state_dicts
            cls.encoder_models["clip"] = clip

        return cls.encoder_models["clip"]
    
    @classmethod
    def get_clip_vision(cls):
        if cls.encoder_models["clip_vision"] is None:
            path = download_huggingface_model(REPO_ID_COMFYORG, *convert_filename_comfyorg("clip_vision", CLIP_VISION_NAME))
            state_dicts = load_torch_file(path, safe_load=True)
            clip_vision = load_clipvision_from_sd(sd=state_dicts)
            del state_dicts
            cls.encoder_models["clip_vision"] = clip_vision

        return cls.encoder_models["clip_vision"]
    
    @classmethod
    def get_vae(cls):
        if cls.encoder_models["vae"] is None:
            path = download_huggingface_model(REPO_ID_COMFYORG, *convert_filename_comfyorg("vae", VAE_NAME))
            state_dicts = load_torch_file(path, safe_load=True)
            vae = VAE(sd=state_dicts)
            del state_dicts
            cls.encoder_models["vae"] = vae

        return cls.encoder_models["vae"]
    
    @classmethod
    def get_taehv(cls):
        if cls.encoder_models["taehv"] is None:
            path = download_huggingface_model(REPO_ID_KIJAI, TAEHV_NAME, "vae_approx")
            state_dicts = load_torch_file(path, safe_load=True)
            from .taehv.taehv import TAEHV
            taesd = TAEHV(sd=state_dicts)
            del state_dicts
            cls.encoder_models["taehv"] = taesd

        return cls.encoder_models["taehv"]
    
    @classmethod
    def apply_lora_cached(cls, model, slot, name, strength):
        if name == "disabled":
            cls.loaded_loras[slot] = None
            return model

        cached = cls.loaded_loras[slot]
        if cached is None or cached["name"] != name:
            lora_path = folder_paths.get_full_path_or_raise("loras", name)
            sd = load_torch_file(lora_path, safe_load=True)
            cls.loaded_loras[slot] = {"name": name, "state_dict": sd}
            print(f"Loaded Lora: {name} (slot: {slot}, strength: {strength})")

        if strength != 0:
            model = comfy.sd.load_lora_for_models(model, None, cls.loaded_loras[slot]["state_dict"], strength, 0)[0]

        return model
    
    @classmethod
    def load(
            cls,
            unet_name,
            lora_1, lora_1_strength,
            lora_2, lora_2_strength,
            lora_3, lora_3_strength,
        ):
        if cls.loaded_model is None or cls.loaded_model[0] != unet_name:

            repo_id = REPO_ID_MODELS[unet_name]
            if repo_id == REPO_ID_COMFYORG:
                path = download_huggingface_model(REPO_ID_COMFYORG, *convert_filename_comfyorg("diffusion_models", unet_name))
            else:
                path = download_huggingface_model(repo_id, unet_name, "diffusion_models")

            if "gguf" in unet_name:
                print("load gguf model...")
                model = load_gguf(path)
            else:
                print("load diffusion model...")
                model_options = {}
                if "e4m3fn" in unet_name:
                    model_options["dtype"] = torch.float8_e4m3fn
                elif "e4m3fn_fast" in unet_name:
                    model_options["dtype"] = torch.float8_e4m3fn
                    model_options["fp8_optimizations"] = True
                elif "e5m2" in unet_name:
                    model_options["dtype"] = torch.float8_e5m2

                model = comfy.sd.load_diffusion_model(path, model_options=model_options)

            cls.loaded_model = (unet_name, model)

        model = cls.loaded_model[-1]

        WanVideoModelLoader_F2.get_clip()
        WanVideoModelLoader_F2.get_clip_vision()
        WanVideoModelLoader_F2.get_vae()
        WanVideoModelLoader_F2.get_taehv()

        model = cls.apply_lora_cached(model, "lora_1", lora_1, lora_1_strength)
        model = cls.apply_lora_cached(model, "lora_2", lora_2, lora_2_strength)
        model = cls.apply_lora_cached(model, "lora_3", lora_3, lora_3_strength)

        return (model, )
    
class WanVideoConfigure_F2:
    config = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "positive": ("STRING", {"multiline": True, "placeholder": "if positive is empty, the model will decide for itself."}),
                "negative": ("STRING", {"multiline": True, }),
                "width": ("INT", {"default": 512, "min": 16, "max": 1280, "step": 16, "display": "slider"}),
                "height": ("INT", {"default": 512, "min": 16, "max": 1280, "step": 16, "display": "slider"}),
                "duration": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 10.0, "step": 0.1, "round": 0.01}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 10.0, "step":0.1, "round": 0.1}),
                "flow_shift": ("FLOAT", {"default": 6.00, "min": 0.00, "max": 20.00, "step":0.01, "round": 0.01}),
                "sampling_steps": ("INT", {"default": 20, "min": 0, "max": 50}),
                "guidance_percent": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 1.00, "step":0.01, "round": 0.01, "advanced": True}),
                "enhance_strength": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 10.0, "step":0.01, "round": 0.01, "advanced": True}),
                "cfg_zero_steps": (("disabled", "1", "2", "3", "4" ), {"advanced": True}),
                "skip_layer": (("disabled", "9", "10", "9, 10"), {"advanced": True}),
                "skip_start_percent": ("FLOAT", {"default": 0.1, "min": 0.00, "max": 1.00, "step":0.01, "round": 0.0, "advanced": True}),
                "skip_end_percent": ("FLOAT", {"default": 0.9, "min": 0.00, "max": 1.00, "step":0.01, "round": 0.01, "advanced": True}),
            },
        }

    RETURN_TYPES = ("PATCH", "NUMBER", "NUMBER", )
    RETURN_NAMES = ("patch", "width", "height", )
    FUNCTION = "set"

    CATEGORY = "Flow2/Wan 2.1"
    
    @classmethod
    def set(
            cls,
            positive,
            negative,
            width,
            height,
            duration,
            guidance_scale,
            guidance_percent,
            flow_shift,
            sampling_steps,
            enhance_strength,
            cfg_zero_steps,
            skip_layer,
            skip_start_percent,
            skip_end_percent,
        ):

        cls.config = Config(
            positive=positive,
            negative=negative,
            width=width,
            height=height,
            frames=round(duration * 16 + 1),
            guidance_scale=guidance_scale,
            guidance_percent=guidance_percent,
            flow_shift=flow_shift,
            sampling_steps=sampling_steps,
            enhance_strength=enhance_strength,
            cfg_zero_steps=cfg_zero_steps,
            skip_layer=skip_layer,
            skip_start_percent=skip_start_percent,
            skip_end_percent=skip_end_percent,
        )

        return ({"key": "signal", "value": True}, width, height, )
    
class WanVideoModelPatcher_F2:
    def __init__(self):
        self.patched_sage = False
        self.patched_order = False

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": ("MODEL", ),
                "patch": ("PATCH", ),
                "sage_attention": (("disabled", "auto", "triton", ), ),
                "teacache": (("disabled", "normal", "retention", ), ),
                "compile_model": (("disabled", "default", ), ),
            }
        }
    
    RETURN_TYPES = ("MODEL", )
    RETURN_NAMES = ("model", )
    FUNCTION = "patch"

    CATEGORY = "Flow2/Wan 2.1"
    
    def patch(
            self,
            model,
            patch,
            sage_attention,
            teacache,
            compile_model,
        ):
        
        if self.patched_sage != sage_attention:
            patch_sage_attention(sage_attention)
            self.patched_sage = sage_attention

        if not self.patched_order:
            patch_model_order()
            self.patched_order = True

        if compile_model != "disabled":
            compile_settings = {
                "fullgraph": False,
                "dynamic": False,
                "backend": "inductor",
                "mode": compile_model
            }

            setattr(model.model, "compile_settings", compile_settings)

        config = WanVideoConfigure_F2.config
        model = comfy_extras.nodes_model_advanced.ModelSamplingSD3.patch(None, model, config.flow_shift)[0]

        l = ((config.frames - 1) // 4) + 1
        #model = patch_riflex(model, l, k=6)
        if config.frames > 0 and config.enhance_strength > 0:
            model = patch_enhance_video(model, weight=config.enhance_strength, latent_frames=l)

        if config.skip_layer != "disabled":
            model = skip_layer_guidance(model, config.skip_layer, config.skip_start_percent, config.skip_end_percent)

        if config.cfg_zero_steps != "disabled":
            model = patch_cfg_zero_star(model, int(config.cfg_zero_steps))

        if teacache != "disabled":
            model = patch_teacache(model, WanVideoModelLoader_F2.loaded_model[0], teacache)

        if compile_model != "disabled":
            model = torch_compile_model(model, compile_model)


        return (model, )
    
def clear_cuda_cache():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

class WanVideoSampler_F2:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "model": ("MODEL", ),
                "seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff}),
                "sampler": (comfy.samplers.SAMPLER_NAMES, {"default": "uni_pc"}),
                "scheduler": (comfy.samplers.SCHEDULER_NAMES, {"default": "sgm_uniform"}),
                "denoised_output": ("BOOLEAN", ),
                "vae_decode_type": (("default", "tiled"), {"advanced": True}),
                "vae_tile_size": ("INT", {"default": 192, "min": 64, "max": 4096, "step": 32, "advanced": True}),
                "preview_resolution": ("INT", {"default": 256, "min": 64, "max": 1280, "step": 32, "advanced": True}),
                "unload_all_models": ("BOOLEAN", {"advanced": True}),
            },
            "optional":{
                "start_image": ("IMAGE", ),
                "end_image": ("IMAGE", ),
            },
        }

    #RETURN_TYPES = ("IMAGE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", )
    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("images", )
    FUNCTION = "sampling"

    CATEGORY = "Flow2/Wan 2.1"

    def create_empty_latent(self, width, height, frames, batch_size):
        latent = torch.zeros([batch_size, 16, ((frames - 1) // 4) + 1, height // 8, width // 8]).cpu()
        return {"samples": latent}
    
    def encode_text(self, clip, text):
        cond = CLIPTextEncode.encode(None, clip, text)[0]
        return cond
    
    def encode_image(self, clip_vision, start_image):
        vision_output = CLIPVisionEncode.encode(None, clip_vision, image=start_image, crop="center")[0]
        return vision_output
    
    def encode_image_to_video(self, width, height, length, latent, vae, start_image, end_image=None):
        start_image = common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
        empty_frames = torch.ones((length, height, width, start_image.shape[-1])).cpu() * 0.5
        empty_frames[:start_image.shape[0]] = start_image
        concat_latent_image = vae.encode(empty_frames[:, :, :, :3]) # vae load

        mask = torch.ones((1, 1, latent.shape[2], concat_latent_image.shape[-2], concat_latent_image.shape[-1])).cpu()
        mask[:, :, :((start_image.shape[0] - 1) // 4) + 1] = 0.0

        if end_image is not None:
            end_image = common_upscale(start_image[:length].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

        return (concat_latent_image, mask, )
    
    def encode_start_end_frames(self, width, height, length, latent, vae, start_image, end_image):
        pass # todo

    def encode_wan_i2v_inpaint(self, width, height, length, latent, vae, start_image, end_image):
        pass # todo

    def encode_wan_i2v_control(self, width, height, length, latent, vae, start_image, end_image):
        pass # todo
    
    def get_split_sigmas(model, scheduler, step1, step2):
        total_steps = step1 + step2

        full_sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, total_steps).cpu()

        high_sigmas = full_sigmas[:step1 + 1]
        low_sigmas = full_sigmas[step1:]
        return (full_sigmas, high_sigmas, low_sigmas)

    def sampling(
            self,
            model,
            seed,
            sampler,
            scheduler,
            denoised_output,
            vae_decode_type,
            vae_tile_size,
            preview_resolution,
            unload_all_models,
            start_image=None,
            end_image=None,
        ):

        i2v = "i2v" in WanVideoModelLoader_F2.loaded_model[0].lower() #model.model.model_config.unet_config.get("model_type", "")

        config = WanVideoConfigure_F2.config

        width = config.width
        height = config.height

        if i2v and start_image is not None:
            width = start_image.shape[2]
            height = start_image.shape[1]

        print(f"final resolution: {width} x {height}")

        clip = WanVideoModelLoader_F2.get_clip()
        positive = self.encode_text(clip, config.positive)
        negative = self.encode_text(clip, config.negative)
        clip.cond_stage_model.to(offload_device)
        clear_cuda_cache()

        batch_size = 1
        empty_latent = self.create_empty_latent(width, height, config.frames, batch_size)
        
        vae = WanVideoModelLoader_F2.get_vae()

        if i2v and start_image is not None:
            clip_vision = WanVideoModelLoader_F2.get_clip_vision()
            clip_vision_output = self.encode_image(clip_vision, start_image)
            clip_vision.model.to(offload_device)
            clear_cuda_cache()
        
            concat_latent_image, concat_mask = self.encode_image_to_video(width, height, config.frames, empty_latent["samples"], vae, start_image)
            vae.first_stage_model.to(offload_device)
            clear_cuda_cache()

            positive = conditioning_set_values(positive, {
                "concat_latent_image": concat_latent_image, "concat_mask": concat_mask, "clip_vision_output": clip_vision_output})
            negative = conditioning_set_values(negative, {
                "concat_latent_image": concat_latent_image, "concat_mask": concat_mask, "clip_vision_output": clip_vision_output})
            
        latent_image = empty_latent["samples"]
        latent = empty_latent.copy()

        latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        full_sigmas = comfy.samplers.calculate_sigmas(model.get_model_object("model_sampling"), scheduler, config.sampling_steps).cpu()
        sampler = comfy.samplers.sampler_object(sampler)

        x0_output = {}
        previewer = get_previewer(WanVideoModelLoader_F2.get_taehv(), model.load_device, preview_resolution)
        pbar = comfy.utils.ProgressBar(config.sampling_steps)
        preview_callback = prepare_callback(previewer, pbar, x0_output)

        def get_random_noise():
            return Noise_RandomNoise(seed=seed).generate_noise(latent)
        
        def get_empty_noise():
            return Noise_EmptyNoise().generate_noise(latent)
        
        def get_cfg_guider2():
            guider = CFGGuider2(model)
            guider.set_conds(positive, negative)
            guider.set_cfg(config.guidance_scale, config.guidance_percent)
            return guider
        
        guider = get_cfg_guider2()
        
        samples = guider.sample(get_random_noise(), latent_image, sampler, full_sigmas, denoise_mask=noise_mask, callback=preview_callback, disable_pbar=False, seed=seed)

        if denoised_output:
            print("denoising...")

            steps = 5
            pbar = comfy.utils.ProgressBar(steps)
            preview_callback = prepare_callback(previewer, pbar, x0_output)

            sampler = comfy.samplers.KSampler(model, steps=steps, device=model.load_device, sampler="dpmpp_2m", scheduler=scheduler, denoise=0.49, model_options=model.model_options)
            samples = sampler.sample(get_random_noise(), positive, negative, cfg=1.0, latent_image=latent_image, force_full_denoise=True, denoise_mask=noise_mask, callback=preview_callback, seed=seed)
            
        model.model.to(offload_device)
        clear_cuda_cache()

        out = latent.copy()
        out["samples"] = samples
        if "x0" in x0_output:
            out_denoised = latent.copy()
            out_denoised["samples"] = model.model.process_latent_out(x0_output["x0"].cpu())
            print("x0 is valid")
        else:
            out_denoised = out
            print("x0 is not valid")

        if vae_decode_type == "tiled":
            print("processing in tiled vae decode...")
            images = VAEDecodeTiled.decode(None, vae, samples=out_denoised, tile_size=vae_tile_size)[0]
        else:
            print("processing in default vae decode...")
            images = VAEDecode.decode(None, vae, samples=out_denoised)[0]

        vae.first_stage_model.to(offload_device)
        clear_cuda_cache()

        if unload_all_models:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()

        return (images, )
    
class WanVideoEnhancer_F2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "images": ("IMAGE", ),
                "upscale_model": (["disabled"] + folder_paths.get_filename_list("upscale_models"), ),
                "interpolate_model": (("disabled", "rife47.pth", "rife48.pth", "rife49.pth"), ), 
                "upscale_factor": ("FLOAT", {"default": 1.00, "min": 1.00, "max": 8.00, "step": 0.01}),
                "interpolate_frame": ("INT", {"default": 30, "min": 30, "max": 60, "step": 30}),
                "order": (("upscale_first", "interpolate_first"), ),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", )
    RETURN_NAMES = ("images", "framerate", )
    FUNCTION = "process"

    CATEGORY = "Flow2/Wan 2.1"

    def process(
            self,
            images,
            upscale_model,
            interpolate_model,
            upscale_factor,
            interpolate_frame,
            order,
        ):

        def upscale(images):
            return self.upscale(upscale_model, images, upscale_factor)

        def interpolate(images):
            return self.interpolate(interpolate_model, images, interpolate_frame)
        
        if order == "upscale_first":
            order = [upscale, interpolate]
        else:
            order = [interpolate, upscale]

        for k in order:
            images = k(images)

        framerate = interpolate_frame if interpolate_model != "disabled" else 16

        return (images, framerate, )

    def interpolate(self, model_name, images, framerate):
        if model_name == "disabled":
            print("No interpolation model specified. Skipping interpolation.")
            return images

        from .frame_interpolation.rife_arch import IFNet
        from .frame_interpolation.utils import preprocess_frames, postprocess_frames, generic_frame_loop
        model_path = download_github_model("styler00dollar/VSGAN-tensorrt-docker", "models", model_name, "vfi_models")
        model = IFNet(arch_ver="4.7")
        sd =  torch.load(model_path)
        model.load_state_dict(sd)
        del sd
        model.eval().to(torch_device)
        frames = preprocess_frames(images)
        
        print("interpolating...")

        def return_middle_frame(frame_0, frame_1, timestep, model, scale_list, in_fast_mode, in_ensemble):
            return model(frame_0, frame_1, timestep, scale_list, in_fast_mode, in_ensemble)
        
        clear_cache_after_n_frames = 10
        multiplier = int(framerate / 15)
        
        args = [model, [8, 4, 2, 1], True, True]
        images = postprocess_frames(
            generic_frame_loop(
                model_name.replace(".pth", ""),
                frames,
                clear_cache_after_n_frames,
                multiplier,
                return_middle_frame,
                *args,
                interpolation_states=None,
                dtype=torch.float32,
            )
        )

        clear_cuda_cache()
        
        return images

    def upscale(self, model_name, images, factor):
        if model_name == "disabled":
            print("No upscale model specified. Skipping upscale.")
            return images

        model_path = folder_paths.get_full_path_or_raise("upscale_models", model_name)
        sd = load_torch_file(model_path, safe_load=True)
        if "module.layers.0.residual_group.blocks.0.norm1.weight" in sd:
            sd = comfy.utils.state_dict_prefix_replace(sd, {"module.":""})
        model = ModelLoader().load_from_state_dict(sd).eval()
        del sd

        if not isinstance(model, ImageModelDescriptor):
            del model
            raise Exception("Upscale model must be a single-image model.")
        
        print("upscaling...")

        scale = model.scale

        memory_required = mm.module_size(model.model)
        memory_required += (512 * 512 * 3) * images.element_size() * max(scale, 1.0) * 384.0
        memory_required += images.nelement() * images.element_size()
        mm.free_memory(memory_required, torch_device)

        model.to(torch_device)
        in_img = images.movedim(-1, -3).to(torch_device)

        tile = 512
        overlap = 32

        oom = True
        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(in_img, lambda a: model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=scale, pbar=pbar)
                oom = False
            except mm.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e
                
        del model

        clear_cuda_cache()

        s = torch.clamp(s.movedim(-3, -1), min=0, max=1.0)

        scale_by = factor / scale

        samples = s.movedim(-1, 1)
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)
        s = common_upscale(samples, width, height, "lanczos", "disabled")
        s = s.movedim(1, -1)
        return s

    
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageSequence
class ResizeImage_F2:
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))

    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required":{
                "width": ("NUMBER", ),
                "height": ("NUMBER", ),
                "image": (sorted(files), {"image_upload": True}),
                "keep_proportion": (("none", "shortest", "longest"), ),
                "image_quality": ("INT", {"default": 90, "min": 0, "max": 100}),
                "saturate": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step":0.1}),
                "blur": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 5.00, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 0.015, "min": 0.000, "max": 0.100, "step":0.001, "round": 0.001}),
                "noise_seed": ("INT", {"default": 1234, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": False}),
                "stop_iteration": ("BOOLEAN", {"default": False}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "NUMBER", "NUMBER", )
    RETURN_NAMES = ("image", "width", "height", )
    FUNCTION = "resize"

    CATEGORY = "Flow2/Wan 2.1"

    def resize(
            self,
            width,
            height,
            image,
            keep_proportion,
            image_quality,
            saturate,
            blur,
            noise_strength,
            noise_seed,
            stop_iteration
        ):
        image = self.load_image(image=image)[0]

        if keep_proportion != "none":
            _, H, W, _ = image.shape

            if keep_proportion == "longest":

                ratio = W / H

                if W > H:
                    width = width
                    height = round(width * ratio)
                elif W < H:
                    height = height
                    width = round(height * ratio)
                else:  # W == H
                    if width > height:
                        height = width
                    else:
                        width = height

            elif keep_proportion == "shortest":

                ratio = H / W

                if W > H:
                    height = height
                    width = round(height * ratio)
                elif W < H:
                    width = width
                    height = round(width * ratio)
                else:  # W == H
                    if width < height:
                        height = width
                    else:
                        width = height

            width = width - (width % 16)
            height = height - (height % 16)

        image = common_upscale(image.movedim(-1, 1), width, height, "lanczos", "center").movedim(1, -1)

        if noise_strength > 0.0:
            torch.manual_seed(noise_seed)
            sigma = torch.ones((image.shape[0], )).to(image.device, image.dtype) * noise_strength
            image_noise = torch.randn_like(image) * sigma[:, None, None, None]
            image_noise = torch.where(image==-1, torch.zeros_like(image), image_noise)
            image = image + image_noise

        out = self.save_images(image, blur=blur, quality=image_quality, saturate=saturate)
        image = self.load_image(out["ui"]["images"][0]["filename"], folder_paths.get_temp_directory())[0]

        if stop_iteration:
            from comfy_execution.graph import ExecutionBlocker
            result = ExecutionBlocker(None), ExecutionBlocker(None), ExecutionBlocker(None), 
        else:
            result = image, width, height,
            
        out["result"] = result
        return out
    
    def save_images(self, images, blur, quality, saturate, filename_prefix="ComfyUI"):
        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()

            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img = img.filter(ImageFilter.GaussianBlur(radius=blur))
            enhancer = ImageEnhance.Color(img)
            img = enhancer.enhance(saturate)

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.jpg"

            saved_path = os.path.join(full_output_folder, file)
            img.save(saved_path, format="JPEG", quality=quality)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
    
    def load_image(self, image, base_directory=None):
        image_path = folder_paths.get_annotated_filepath(image, base_directory)

        img = node_helpers.pillow(Image.open, image_path)

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)