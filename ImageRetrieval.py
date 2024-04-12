from torchmetrics.functional.pairwise import pairwise_cosine_similarity
from tqdm.notebook import tqdm
import numpy as np
import torch
import glob
import cv2
import os


class ImageRetrieval:
    def __init__(self, Model, img_paths):
        self.model = Model
        self.img_paths = img_paths
        self.embeddings = None

    def get_embeddings(self, save, batch_size=1, workers=0):
        self.embeddings = self.model.inference(self.img_paths, batch_size, workers)
        torch.save(self.embeddings, save)
        print(f"Features saves to {save}")

    def load_embeddings(self, path, n_components=-1, use_head=True):
        self.embeddings = torch.load(path, map_location=self.model.device)
        if use_head:
            with torch.no_grad():
                self.embeddings = self.model.head(self.embeddings)

        if n_components > 0:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=n_components)
            self.embeddings = torch.FloatTensor(self.pca.fit_transform(self.embeddings))

    @staticmethod
    def metric(features_queries, features_galleries, topk=3, cosine=True):

        if cosine:
            dist = pairwise_cosine_similarity(features_queries, features_galleries)
        else:
            dist = torch.cdist(features_queries, features_galleries)

        return torch.topk(dist, dim=1, k=topk, largest=cosine)

    def query(self, query_paths, topk=3, cosine=True):
        assert self.embeddings is not None, "You need to get or load embeddings"
        queries = self.model.inference(query_paths)
        if queries.shape[1] != self.embeddings.shape[1]:
            queries = torch.FloatTensor(self.pca.transform(queries))
        return self.metric(queries, self.embeddings, topk, cosine)

    def search(self, query_folder="query_images", topk=3, cosine=True, seed=None):
        if query_folder is None:
            result_folder = "results_inside"
            rng = np.random.default_rng(seed)
            query_paths = rng.choice(self.img_paths, 10, replace=False)
        else:
            result_folder = "results"
            query_paths = glob.glob(f"{query_folder}/*")

        postfix = "_default" if self.model.head is None else "_finetuned"
        idxs = self.query(query_paths, topk, cosine)[1]
        for i, path in enumerate(query_paths):
            name = f"{i}{postfix}.jpg" #if query_folder is None else os.path.basename(path)
            result = self.draw(path, idxs[i][query_folder is None:].tolist(), self.img_paths)
            result = cv2.putText(result, postfix[1:], (result.shape[1] // 2 - 145, 50),
                                 cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.imwrite(f"{result_folder}/{name}", result)

    @staticmethod
    def draw(query_path, k_idxs, galleries, w=6, height=500):
        # Load and resize the query image
        query = cv2.imread(query_path)
        query_width = int(query.shape[1] * (height / query.shape[0]))
        query = cv2.resize(query, (query_width, height))
        query = cv2.rectangle(query, (0, 0), (query_width - 1, height - 1), (0, 255, 0), 10)

        # Load and resize gallery images
        gallery_images = []
        for idx in k_idxs:
            img = cv2.imread(galleries[idx])
            img_width = int(img.shape[1] * (height / img.shape[0]))
            img = cv2.resize(img, (img_width, height))
            gallery_images.append(img)

        img_lsit = [query] + gallery_images
        rows = [np.hstack(img_lsit[i:i + w]) for i in range(0, len(img_lsit), w)]
        max_width = np.max([row.shape[1] for row in rows])

        # Create the result image with correct layout
        result = np.ones((len(rows)*height, max_width, 3), dtype=np.uint8) * 255

        # Paste query image at the top
        for i, row in enumerate(rows):
            result[i*height:(i+1)*height, :row.shape[1], :] = row

        return result


def delete_failed_images(paths):
    for path in tqdm(paths):
        if cv2.imread(path) is None:
            os.remove(path)
            print(f"failed image {path} was deleted")
