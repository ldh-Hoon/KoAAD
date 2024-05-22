from src.datasets.base_dataset import SimpleAudioFakeDataset
import pandas as pd
from pathlib import Path

class MAILABS(SimpleAudioFakeDataset):                                            
    def __init__(self, root_path, subset=None, **kwargs):
        super().__init__(root_path, subset, **kwargs)
        self.root_path = Path(f'{root_path}')
        self.subset = subset
        self.samples = self.load_samples()

    def load_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        folders_1 = list(self.root_path.glob("en_US/by_book/*"))
        for f1 in folders_1:
            if not os.path.isdir(f1):
                continue
            folders_2 = list(f1.glob("*"))
            for f2 in folders_2:
                path = f1 / f2.name

                if not path.exists():
                    print(f"{path} 경로를 찾을 수 없습니다.")

                samples_list = list(path.rglob("*.wav"))
                if self.subset == 'train':
                    samples_list = samples_list[:int(len(samples_list)*split[0])]
                elif self.subset == 'test':
                    samples_list = samples_list[int(len(samples_list)*(split[0]):]
                for sample in samples_list:
                    if sample.stem[0]==".":
                        continue
                    if os.path.exists(sample):
                        samples["user_id"].append(None)
                        samples["path"].append(sample)
                        samples["sample_name"].append(sample.stem)
                        samples["attack_type"].append("-")
                        samples["label"].append("bonafide")

        return pd.DataFrame(samples)

