# https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite/blob/main/videohelpersuite/latent_preview.py

from PIL import Image
import time
import io
from threading import Thread
import torch.nn.functional as F
import torch

import latent_preview
import server
serv = server.PromptServer.instance

#from .utils import hook

# rates_table = {'Mochi': 24//6, 'LTXV': 24//8, 'HunyuanVideo': 24//4,
#                'Cosmos1CV8x8x8': 24//8, 'Wan21': 16//4}

class WrappedPreviewer(latent_preview.LatentPreviewer):
    def __init__(self, previewer, rate=8, resolution=128):
        self.first_preview = True
        self.last_time = 0
        self.last_time2 = 0
        self.c_index = 0
        self.rate = rate
        self.resolution = resolution

        if hasattr(previewer, 'taesd'):
            self.taesd = previewer.taesd
        elif hasattr(previewer, 'latent_rgb_factors'):
            self.latent_rgb_factors = previewer.latent_rgb_factors
            self.latent_rgb_factors_bias = previewer.latent_rgb_factors_bias
        else:
            raise Exception('Unsupported preview type for VHS animated previews')
        
        self.count = 0

    def decode_latent_to_preview_image(self, x0):
        if x0.ndim == 5:
            #Keep batch major
            x0 = x0.movedim(2,1)
            x0 = x0.reshape((-1,)+x0.shape[-3:])
        num_images = x0.size(0)

        new_time = time.time()

        if new_time - self.last_time2 < 1.0: # for teacache
            return None

        self.last_time2 = new_time

        num_previews = int((new_time - self.last_time) * self.rate)
        self.last_time = self.last_time + num_previews/self.rate
        if num_previews > num_images:
            num_previews = num_images
        elif num_previews <= 0:
            return None
        if self.first_preview:
            self.first_preview = False
            serv.send_sync('VHS_latentpreview', {'length': num_images, 'rate': self.rate})
            self.last_time = new_time + 1/self.rate
        if self.c_index + num_previews > num_images:
            x0 = x0.roll(-self.c_index, 0)[:num_previews]
        else:
            x0 = x0[self.c_index:self.c_index + num_previews]

        Thread(target=self.process_previews, args=(x0, self.c_index,
                                            num_images)).run()

        self.c_index = (self.c_index + num_previews) % num_images
        return None
    
    def process_previews(self, image_tensor, ind, leng):
        r = self.resolution
        image_tensor = self.decode_latent_to_preview(image_tensor)
        if image_tensor.size(1) > r or image_tensor.size(2) > r:
            image_tensor = image_tensor.movedim(-1,0)
            if image_tensor.size(2) < image_tensor.size(3):
                height = (r * image_tensor.size(2)) // image_tensor.size(3)
                image_tensor = F.interpolate(image_tensor, (height,r), mode='bilinear')
            else:
                width = (r * image_tensor.size(3)) // image_tensor.size(2)
                image_tensor = F.interpolate(image_tensor, (r, width), mode='bilinear')
            image_tensor = image_tensor.movedim(0,-1)
        previews_ubyte = (image_tensor.clamp(0, 1)
                        .mul(0xFF)  # to 0..255
                        ).to(device="cpu", dtype=torch.uint8)
        for preview in previews_ubyte:
            i = Image.fromarray(preview.numpy())
            message = io.BytesIO()
            message.write((1).to_bytes(length=4, byteorder='big')*2)
            message.write(ind.to_bytes(length=4, byteorder='big'))
            i.save(message, format="JPEG", quality=90, compress_level=1)
            #NOTE: send sync already uses call_soon_threadsafe
            serv.send_sync(server.BinaryEventTypes.PREVIEW_IMAGE,
                        message.getvalue(), serv.client_id)
            if self.rate == 16:
                ind = (ind + 1) % ((leng-1) * 4 - 1)
            else:
                ind = (ind + 1) % leng
                
    def decode_latent_to_preview(self, x0):
        if hasattr(self, 'taesd'):
            #x_sample = self.taesd.decode_video(x0).movedim(1, 3)
            x0 = x0.unsqueeze(0)
            x_sample = self.taesd.decode_video(x0, parallel=False, show_progress_bar=False)[0].permute(0, 2, 3, 1)
            return x_sample
        else:
            self.latent_rgb_factors = self.latent_rgb_factors.to(dtype=x0.dtype, device=x0.device)
            if self.latent_rgb_factors_bias is not None:
                self.latent_rgb_factors_bias = self.latent_rgb_factors_bias.to(dtype=x0.dtype, device=x0.device)
            latent_image = F.linear(x0.movedim(1, -1), self.latent_rgb_factors,
                                    bias=self.latent_rgb_factors_bias)
            latent_image = (latent_image + 1.0) / 2.0
            return latent_image