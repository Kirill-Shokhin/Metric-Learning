from torchmetrics.functional.pairwise import pairwise_cosine_similarity
import numpy as np
import torch
import glob
import cv2
import os


class ImageRetrieval:
    def __init__(self, Model, Dataset):
        self.model = Model()
        self.dataset = Dataset()
        self.features = None

    def get_embeddings(self, save):
        self.features = self.model.inference(self.dataset.img_paths)
        torch.save(self.features, save)
        print(f"Features saves to {save}")

    def load_embeddings(self, path, n_components=-1):
        self.features = torch.load(path)
        if n_components > 0:
            from sklearn.decomposition import PCA
            self.pca = PCA(n_components=n_components)
            self.features = torch.FloatTensor(self.pca.fit_transform(self.features))

    @staticmethod
    def metric(features_queries, features_galleries, topk=3, cosine=True):

        if cosine:
            dist = pairwise_cosine_similarity(features_queries, features_galleries)
        else:
            dist = torch.cdist(features_queries, features_galleries)

        return torch.topk(dist, dim=1, k=topk, largest=cosine)

    def query(self, query_paths, topk=3, cosine=False):
        assert self.features is not None, "You need to get or load embeddings"
        features_queries = self.model.inference(query_paths)
        if hasattr(self, "pca"):
            features_queries = torch.FloatTensor(self.pca.transform(features_queries))
        return self.metric(features_queries, self.features, topk, cosine)

    def search(self, query_folder="tests", result_folder="results", topk=3, cosine=False):
        query_paths = glob.glob(f"{query_folder}/*")
        dists, idxs = self.query(query_paths, topk, cosine)
        for i, path in enumerate(query_paths):
            result = self.draw(path, idxs[i].tolist(), self.dataset.img_paths)
            cv2.imwrite(f"{result_folder}/{os.path.basename(path)}", result)

    @staticmethod
    def draw(query_path, k_idxs, galleries):
        result = [cv2.imread(galleries[idx]) for idx in k_idxs]
        query = cv2.resize(cv2.imread(query_path), result[0].shape[:2][::-1])

        res = np.concatenate((query, 255*np.ones((result[0].shape[0], 70, 3)), *result), axis=1)
        res = cv2.resize(res, dsize=(0, 0), fx=2, fy=2)
        return res
