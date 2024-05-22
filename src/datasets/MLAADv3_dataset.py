from src.datasets.base_dataset import SimpleAudioFakeDataset
import pandas as pd
from pathlib import Path

class MLAADv3(SimpleAudioFakeDataset):
    languages=['fr', 'et', 'ar', 'hu', 'bg', 'es', 'el', 'da', 'ga', 'ru', 'fi', 
               'uk', 'pl', 'en', 'sw', 'mt', 'sk', 'ro', 'hi', 'cs', 'nl', 'it', 'de']
                                            
    def __init__(self, root_path, subset=None, **kwargs):
        super().__init__(root_path, subset, **kwargs)
        self.root_path = Path(f'{root_path}')
        self.subset = subset
        self.samples = self.load_samples()

    def load_samples(self):
        samples = {
            "user_id": [],
            "language" : [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        for lang in self.languages:
            r_path = self.root_path / f"fake/{lang}"
            folders = list(r_path.glob("*"))
            for folder in folders:
                path = r_path / folder.name

                if not path.exists():
                    print(f"{path} 경로를 찾을 수 없습니다.")
                    continue
                samples_list = list(path.rglob("*.wav"))
                if self.subset == 'train':
                    samples_list = samples_list[:int(len(samples_list)*0.7)]
                else:
                    samples_list = samples_list[int(len(samples_list)*0.7):]
                for sample in samples_list:
                    samples["user_id"].append(None)
                    samples["language"].append(lang)
                    samples["path"].append(sample)
                    samples["sample_name"].append(sample.stem)
                    samples["attack_type"].append("-")
                    samples["label"].append("spoof")
        print(f"__MLAADv3_{self.subset}:{len(sample['label'])}") 
        return pd.DataFrame(samples)
