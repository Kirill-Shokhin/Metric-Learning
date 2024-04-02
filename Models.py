from PIL import Image
import torch
from tqdm.notebook import tqdm
import numpy as np
import cv2
from torch import nn
from torch.utils.data import Dataset, DataLoader


class DINOv2(nn.Module):
    def __init__(self, head=None, device="cuda"):
        super().__init__()
        self.device = device
        self.model, self.preprocess, self.head = self.init_model(head, device)

    @staticmethod
    def init_model(head, device):
        from transformers import AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = torch.hub.load('facebookresearch/dinov2', "dinov2_vitb14").eval().to(device)
        if head is not None:
            head = head.eval().to(device)

        return model, processor, head

    def forward(self, x):
        x = self.model(x)
        if self.head is not None:
            x = self.head(x)
        return x

    @torch.no_grad()
    def _inference(self, path):
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to open the image: {path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)["pixel_values"]
        outputs = self(inputs)
        return outputs

    @torch.no_grad()
    def _inference_batch(self, batch_images):
        return self(batch_images.to(self.device)).cpu().detach()

    def inference(self, paths, batch_size=1):
        if batch_size == 1:
            outputs = []
            for path in tqdm(paths):
                out = self._inference(path)
                if out is not None:
                    outputs.append(out.cpu().detach())
        else:
            ds = BaseDataset(paths, self.preprocess)
            dataloader = DataLoader(ds, batch_size, collate_fn=lambda batch: torch.cat(
                [item for item in batch if item is not None]))
            outputs = [self._inference_batch(batch_images) for batch_images in tqdm(dataloader)]
        return torch.cat(outputs) if outputs else None


class BaseDataset(Dataset):
    def __init__(self, paths, transform=None):
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        image = cv2.imread(path)
        if image is None:
            print(f"Failed to open the image: {path}")
            return None

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(images=image, return_tensors="pt")["pixel_values"]
        return image


# class BaseModel:
#     def __init__(self, path, device="cuda"):
#         self.device = device
#         self.model, self.preprocess = self.init_model(path)
#
#     def init_model(self, path):
#         raise NotImplementedError
#
#     def inference(self, img_paths):
#         raise NotImplementedError
#
#
# class CLIP(BaseModel):
#     def __init__(self, device="cuda"):
#         super().__init__(path="openai/clip-vit-base-patch32", device=device)
#
#     def init_model(self, path):
#         from transformers import AutoProcessor, CLIPVisionModel
#
#         model = CLIPVisionModel.from_pretrained(path).eval().to(self.device)
#         processor = AutoProcessor.from_pretrained(path)
#         return model, processor
#
#     @torch.no_grad()
#     def _inference(self, path):
#         image = Image.open(path)
#         inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs)
#         # last_hidden_state = outputs.last_hidden_state
#         pooled_output = outputs.pooler_output
#         return pooled_output
#
#     def inference(self, paths):
#         return torch.concat([self._inference(path).cpu().detach() for path in tqdm(paths)])
#
#
# class DINOv2_HF(BaseModel):
#     def __init__(self, device="cuda"):
#         super().__init__(path="facebook/dinov2-base", device=device)
#
#     def init_model(self, path):
#         from transformers import AutoImageProcessor, Dinov2Model
#
#         model = Dinov2Model.from_pretrained(path).eval().to(self.device)
#         processor = AutoImageProcessor.from_pretrained(path)
#
#         return model, processor
#
#     @torch.no_grad()
#     def _inference(self, path):
#         image = cv2.imread(path)[:, :, ::-1]
#         # image = pad_to_square(image)
#         inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
#         outputs = self.model(**inputs)
#         # last_hidden_state = outputs.last_hidden_state
#         pooled_output = outputs.pooler_output
#         return pooled_output
#
#     def inference(self, paths):
#         return torch.concat([self._inference(path).cpu().detach() for path in tqdm(paths)])
#
#
# def pad_to_square(image):
#     height, width = image.shape[:2]
#     max_dim = max(height, width)
#
#     pad_height = max_dim - height
#     pad_width = max_dim - width
#
#     pad_top = pad_height // 2
#     pad_bottom = pad_height - pad_top
#     pad_left = pad_width // 2
#     pad_right = pad_width - pad_left
#
#     padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant',
#                           constant_values=120)
#
#     return padded_image
