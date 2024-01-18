import pandas as pd


class BaseDataset:
    def __init__(self, path):
        self.img_paths = self.init_dataset(path)

    def init_dataset(self, path):
        raise NotImplementedError


class InShop(BaseDataset):
    def __init__(self):
        super().__init__(path="df.csv")

    def init_dataset(self, path):
        df = pd.read_csv(path)
        df['path'] = df['path'].str.replace('img_highres', 'img')
        return df['path'].tolist()