from transformers import AutoImageProcessor, AutoModel
from torch.utils.data import Dataset, DataLoader
from tqdm.notebook import tqdm
from torch import nn
import torch
import cv2


class DINOv2(nn.Module):
    def __init__(self, head="head.ckpt", device="cuda"):
        super().__init__()
        self.device = device
        self.model, self.preprocess, self.head = self.init_model(head, device)

    @staticmethod
    def init_model(head_path, device):
        processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        model = AutoModel.from_pretrained('facebook/dinov2-base').eval().to(device)
        head = None
        if head_path is not None:
            head = HeadLayer()
            head.load_state_dict(torch.load(head_path, map_location=device)['state_dict'])
            head.eval().to(device)

        return model, processor, head

    def forward(self, x):
        x = self.model(pixel_values=x).pooler_output
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

    def inference(self, paths, batch_size=1, workers=0):
        if batch_size == 1:
            outputs = []
            for path in tqdm(paths):
                out = self._inference(path)
                if out is not None:
                    outputs.append(out.cpu().detach())
        else:
            ds = BaseDataset(paths, self.preprocess)
            dataloader = DataLoader(ds, batch_size, num_workers=workers,
                                    collate_fn=lambda batch: torch.cat(
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


class HeadLayer(nn.Module):
    def __init__(self, d_model=768, extend=4):
        super().__init__()
        self.linear1 = nn.Linear(d_model, extend * d_model)
        self.linear2 = nn.Linear(extend * d_model, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm(x)
        return x

