from src.datasets.base_dataset import SimpleAudioFakeDataset
import pandas as pd
from pathlib import Path

class MAILABS(SimpleAudioFakeDataset):                                            
    def __init__(self, root_path, **kwargs):
        super().__init__(root_path, **kwargs)
        self.root_path = Path(f'{root_path}/MAILABS')
        self.samples = self.load_samples()

    def load_samples(self):
        samples = {
            "user_id": [],
            "sample_name": [],
            "attack_type": [],
            "label": [],
            "path": []
        }

        path = self.root_path / ""
        
        # 해당 언어의 디렉토리가 존재하는지 확인
        if not path.exists():
            print(f"{path} 경로를 찾을 수 없습니다.")
        
        samples_list = list(path.rglob("*.wav"))
        samples_list = self.split_samples(samples_list)
        for sample in samples_list:
            samples["user_id"].append(None)
            samples["path"].append(sample)
            samples["sample_name"].append(sample.stem)
            samples["attack_type"].append("-")
            samples["label"].append("bonafide")

        return pd.DataFrame(samples)

