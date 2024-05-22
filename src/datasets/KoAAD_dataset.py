from src.datasets.base_dataset import SimpleAudioFakeDataset
import pandas as pd
from pathlib import Path
import os

class KoAAD(SimpleAudioFakeDataset):                                            
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

        folders_1 = list(self.root_path.glob("*"))
        for f1 in folders_1:
            if not os.path.isdir(f1):
                continue
        
            if not f1.exists():
                print(f"{path} 경로를 찾을 수 없습니다.")
            
            samples_list = list(f1.rglob("*.[wm][ap][v3]"))
            if self.subset == 'train':
                samples_list = samples_list[:int(len(samples_list)*0.7)]
            else:
                samples_list = samples_list[int(len(samples_list)*0.7):]
            for sample in samples_list:
                samples["user_id"].append(None)
                samples["path"].append(sample)
                samples["sample_name"].append(sample.stem)
                samples["attack_type"].append("-")
                samples["label"].append("spoof")

        print(f"KoAAD_{self.subset}:{len(samples['label'])}")
        return pd.DataFrame(samples)

