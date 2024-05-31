import logging
from typing import List, Optional


import pandas as pd

from src.datasets.base_dataset import SimpleAudioFakeDataset
from src.datasets.deepfake_asvspoof_dataset import DeepFakeASVSpoofDataset
from src.datasets.fakeavceleb_dataset import FakeAVCelebDataset
from src.datasets.wavefake_dataset import WaveFakeDataset
from src.datasets.asvspoof_dataset import ASVSpoof2019DatasetOriginal
from src.datasets.MLAADv3_dataset import MLAADv3
from src.datasets.MAILABS_dataset import MAILABS
from src.datasets.aihub_dataset import AIHUB
from src.datasets.KoAAD_dataset import KoAAD


LOGGER = logging.getLogger()


class DetectionDataset(SimpleAudioFakeDataset):
    def __init__(
        self,
        asvspoof_path=None,
        wavefake_path=None,
        fakeavceleb_path=None,
        asvspoof2019_path=None,
        MLAADv3_path=None,
        MAILABS_path=None,
        AIHUB_path=None,
        KoAAD_path=None,
        subset: str = "val",
        augmentation = False,
        transform=None,
        oversample: bool = True,
        undersample: bool = False,
        return_label: bool = True,
        reduced_number: Optional[int] = None,
        return_meta: bool = False,
    ):
        super().__init__(
            subset=subset,
            transform=transform,
            return_label=return_label,
            return_meta=return_meta,
            augmentation = augmentation,
        )
        datasets = self._init_datasets(
            asvspoof_path=asvspoof_path,
            wavefake_path=wavefake_path,
            fakeavceleb_path=fakeavceleb_path,
            asvspoof2019_path=asvspoof2019_path,
            MLAADv3_path=MLAADv3_path,
            MAILABS_path=MAILABS_path,
            AIHUB_path=AIHUB_path,
            KoAAD_path=KoAAD_path,
            subset=subset,
            augmentation = augmentation,
        )
        self.samples = pd.concat([ds.samples for ds in datasets], ignore_index=True)

        if oversample:
            self.oversample_dataset()
        elif undersample:
            self.undersample_dataset()

        if reduced_number:
            LOGGER.info(f"Using reduced number of samples - {reduced_number}!")
            self.samples = self.samples.sample(
                min(len(self.samples), reduced_number),
                random_state=42,
            )

    def _init_datasets(
        self,
        subset: str,
        augmentation: bool,
        asvspoof_path: Optional[str],
        wavefake_path: Optional[str],
        fakeavceleb_path: Optional[str],
        asvspoof2019_path: Optional[str],
        MLAADv3_path=Optional[str],
        MAILABS_path=Optional[str],
        AIHUB_path=Optional[str],
        KoAAD_path=Optional[str],
    ) -> List[SimpleAudioFakeDataset]:
        datasets = []

        if asvspoof_path is not None:
            asvspoof_dataset = DeepFakeASVSpoofDataset(asvspoof_path, subset=subset)
            datasets.append(asvspoof_dataset)

        if wavefake_path is not None:
            wavefake_dataset = WaveFakeDataset(wavefake_path, subset=subset)
            datasets.append(wavefake_dataset)

        if fakeavceleb_path is not None:
            fakeavceleb_dataset = FakeAVCelebDataset(fakeavceleb_path, subset=subset)
            datasets.append(fakeavceleb_dataset)

        if asvspoof2019_path is not None:
            la_dataset = ASVSpoof2019DatasetOriginal(
                asvspoof2019_path, fold_subset=subset
            )
            datasets.append(la_dataset)

        if MLAADv3_path is not None:
            MLAADv3_dataset = MLAADv3(MLAADv3_path, subset=subset)
            datasets.append(MLAADv3_dataset)

        if MAILABS_path is not None:
            MAILABS_dataset = MAILABS(MAILABS_path, subset=subset)
            datasets.append(MAILABS_dataset)

        if AIHUB_path is not None:
            aihub_dataset = AIHUB(AIHUB_path, subset=subset)
            datasets.append(aihub_dataset)

        if KoAAD_path is not None:
            KoAAD_dataset = KoAAD(KoAAD_path, subset=subset)
            datasets.append(KoAAD_dataset)
        return datasets

    def oversample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        diff_length = spoof_length - bona_length

        if diff_length < 0:
            raise NotImplementedError

        if diff_length > 0:
            bonafide = samples.get_group("bonafide").sample(diff_length, replace=True)
            self.samples = pd.concat([self.samples, bonafide], ignore_index=True)

    def undersample_dataset(self):
        samples = self.samples.groupby(by=["label"])
        bona_length = len(samples.groups["bonafide"])
        spoof_length = len(samples.groups["spoof"])

        if spoof_length < bona_length:
            raise NotImplementedError

        if spoof_length > bona_length:
            spoofs = samples.get_group("spoof").sample(bona_length, replace=True)
            self.samples = pd.concat(
                [samples.get_group("bonafide"), spoofs], ignore_index=True
            )

    def get_bonafide_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("bonafide")
        return self.samples

    def get_spoof_only(self):
        samples = self.samples.groupby(by=["label"])
        self.samples = samples.get_group("spoof")
        return self.samples
