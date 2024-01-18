from PIL import Image
import torch
from tqdm.notebook import tqdm


class BaseModel:
    def __init__(self, path, device="cuda"):
        self.device = device
        self.model, self.preprocess = self.init_model(path)

    def init_model(self, path):
        raise NotImplementedError

    def inference(self, img_paths):
        raise NotImplementedError


class CLIP(BaseModel):
    def __init__(self, device="cuda"):
        super().__init__(path="openai/clip-vit-base-patch32", device=device)

    def init_model(self, path):
        from transformers import AutoProcessor, CLIPVisionModel

        model = CLIPVisionModel.from_pretrained(path).eval().to(self.device)
        processor = AutoProcessor.from_pretrained(path)
        return model, processor

    @torch.no_grad()
    def _inference(self, path):
        image = Image.open(path)
        inputs = self.preprocess(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        # last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        return pooled_output

    def inference(self, paths):
        return torch.concat([self._inference(path).cpu().detach() for path in tqdm(paths)])
