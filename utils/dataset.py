from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image


class CubCTrainDataset(Dataset):
    """
    CUB-C 训练集加载：
    根目录结构示例：
        CUB-C/
          ├── annotations/
          │     ├── images.txt
          │     ├── image_class_labels.txt
          │     ├── train_test_split.txt
          │     └── ...
          ├── origin/
          │     └── 001.Black_footed_Albatross/xxx.jpg
          ├── fog/
          ├── contrast/
          ├── brightness/
          ├── motion_blur/
          └── snow/

    所有 corruption 目录下的相对路径与 origin 完全一致。
    本 Dataset 返回 (degraded_img, clean_img, label)
    """
    def __init__(self, root, corruption="fog", split="train", transform=None):
        """
        Args:
            root: CUB-C 根目录路径，例如 /data/.../CUB-C
            corruption: 使用哪种降质类型目录
                可选：["fog", "contrast", "brightness", "motion_blur", "snow", "origin"]
            split: "train" 或 "test"，根据 annotations/train_test_split.txt 过滤样本
            transform: 对图像同时应用的 transform
        """
        self.root = Path(root)
        self.transform = transform
        self.corruptions = self._normalize_corruptions(corruption)
        self.split = split

        allowed_corruptions = {
            "fog", "contrast", "brightness", "motion_blur", "snow", "origin"
        }
        for corr in self.corruptions:
            if corr not in allowed_corruptions:
                raise ValueError(
                    f"Unsupported corruption '{corr}'. "
                    f"Supported: {sorted(allowed_corruptions)}"
                )

        ann_root = self.root / "annotations"
        images_txt = ann_root / "images.txt"
        labels_txt = ann_root / "image_class_labels.txt"
        split_txt = ann_root / "train_test_split.txt"

        if not images_txt.exists():
            raise FileNotFoundError(f"Expected file not found: {images_txt}")
        if not labels_txt.exists():
            raise FileNotFoundError(f"Expected file not found: {labels_txt}")
        if not split_txt.exists():
            raise FileNotFoundError(f"Expected file not found: {split_txt}")

        # 1) img_id -> 相对路径（相对于 origin/ 的路径）
        id2relpath = {}
        with open(images_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id_str, rel_path = line.split()
                id2relpath[int(img_id_str)] = rel_path

        # 2) img_id -> 类别 label（与 CUB_200_2011 格式一致，通常是 1~200）
        id2label = {}
        with open(labels_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id_str, label_str = line.split()
                id2label[int(img_id_str)] = int(label_str)

        # 3) 根据 train_test_split.txt 过滤出 train 或 test
        base_samples = []
        with open(split_txt, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                img_id_str, is_train_str = line.split()
                img_id = int(img_id_str)
                is_train = bool(int(is_train_str))

                if self.split == "train" and not is_train:
                    continue
                if self.split == "test" and is_train:
                    continue

                rel_path = id2relpath[img_id]
                label = id2label[img_id]
                base_samples.append((rel_path, label))

        if len(base_samples) == 0:
            raise RuntimeError(
                f"No samples found for split='{self.split}' "
                f"in {self.root}. Please check annotations/*.txt."
            )
        self.samples = []
        for rel_path, label in base_samples:
            for corr in self.corruptions:
                self.samples.append((rel_path, label, corr))

    def _normalize_corruptions(self, corruption):
        allowed = ["fog", "contrast", "brightness", "motion_blur", "snow", "origin"]
        if isinstance(corruption, str):
            value = corruption.strip()
            if value == "all":
                return allowed  # Include origin for identity mapping learning
            items = [c.strip() for c in value.split(",") if c.strip()]
            if not items:
                raise ValueError("Empty corruption list")
            return items
        if isinstance(corruption, (list, tuple, set)):
            return list(corruption)
        raise TypeError("corruption must be str, list, tuple, or set")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, label, corruption = self.samples[idx]

        # 干净图像
        clean_path = self.root / "origin" / rel_path
        # 降质图像（如果 corruption="origin"，那就是使用干净图本身）
        degraded_root = self.root / corruption
        degraded_path = degraded_root / rel_path

        clean = Image.open(clean_path).convert("RGB")
        degraded = Image.open(degraded_path).convert("RGB")

        if self.transform is not None:
            clean_t = self.transform(clean)
            degraded_t = self.transform(degraded)
        else:
            clean_t, degraded_t = clean, degraded

        return degraded_t, clean_t, label


class ImageNetValCDataset(Dataset):
    def __init__(self, root, transform=None):
        """
        root: data/imagenet_val_c
        假定：
        - root/images/<id>.jpg
        - root/labels.txt  (id label)
        """
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        with open(self.root / "labels.txt") as f:
            for line in f:
                img_id, label = line.strip().split()
                self.samples.append((img_id, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_id, label = self.samples[idx]
        img = Image.open(self.root / "images" / f"{img_id}.jpg").convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(label)
