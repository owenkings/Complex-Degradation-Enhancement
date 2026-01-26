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
                
                # CubCTrainDataset 设计逻辑：
                # root/corruption/rel_path
                # 如果 corruption="origin", 则 root/origin/rel_path
                
                base_samples.append((img_id, rel_path, label))
        
        # 4) 生成最终 samples 列表：(degraded_path, clean_path, label)
        self.samples = []
        for (img_id, rel_path, label) in base_samples:
            clean_path = self.root / "origin" / rel_path
            
            # 如果 corruption 是 origin，则 degraded == clean
            if "origin" in self.corruptions:
                 self.samples.append((clean_path, clean_path, label))
            
            # 其他 corruptions
            for corr in self.corruptions:
                if corr == "origin":
                    continue
                # CUB-C 是 root/corruption/rel_path
                degraded_path = self.root / corr / rel_path
                if degraded_path.exists():
                    self.samples.append((degraded_path, clean_path, label))
                else:
                    # 某些图片可能没有对应的 corruption 版本或者路径不对
                    pass

    def _normalize_corruptions(self, corruption):
        if corruption == "all":
            return ["fog", "contrast", "brightness", "motion_blur", "snow"]
        return [c.strip() for c in corruption.split(",")]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        degraded_path, clean_path, label = self.samples[idx]
        
        degraded_img = Image.open(degraded_path).convert("RGB")
        clean_img = Image.open(clean_path).convert("RGB")
        
        if self.transform:
            degraded_img = self.transform(degraded_img)
            clean_img = self.transform(clean_img)
            
        return degraded_img, clean_img, label


class ImageNetCDataset(Dataset):
    def __init__(self, root, corruption, severity, transform, synset_mapping, max_samples=0):
        self.root = Path(root)
        self.corruption = corruption
        self.severity = severity
        self.transform = transform
        self.synset_to_idx = synset_mapping
        self.max_samples = max_samples
        self.samples = self._collect_samples()

    def _collect_samples(self):
        if self.corruption == "origin":
            base = self.root / "origin"
        else:
            # Check if severity subdirectory exists
            base_corr = self.root / self.corruption
            base_sev = base_corr / str(self.severity)
            if base_sev.exists():
                base = base_sev
            elif base_corr.exists():
                # Fallback to corruption root if severity folder not found
                # This assumes the corruption folder itself contains the synsets (e.g. flat structure or specific severity download)
                base = base_corr
            else:
                base = base_sev # Let it fail in the next check if neither exists
                
        if not base.exists():
            raise FileNotFoundError(f"ImageNet-C path not found: {base}")

        samples = []
        for synset_dir in base.iterdir():
            if not synset_dir.is_dir():
                continue
            synset = synset_dir.name
            if synset not in self.synset_to_idx:
                continue
            label = self.synset_to_idx[synset]
            for img_path in synset_dir.iterdir():
                if img_path.suffix.lower() not in {".jpeg", ".jpg", ".png"}:
                    continue
                samples.append((img_path, label))
                if self.max_samples and len(samples) >= self.max_samples:
                    return samples

        if len(samples) == 0:
            raise RuntimeError(f"No samples found under {base}")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        
        # Load clean image
        # Assuming structure: root/<corruption>/<severity>/<synset>/<image>
        # Clean image: root/origin/<synset>/<image>
        synset = img_path.parent.name
        filename = img_path.name
        clean_path = self.root / "origin" / synset / filename
        
        if clean_path.exists():
            clean_img = Image.open(clean_path).convert("RGB")
        else:
            # Strict mode: raise error if clean image is missing
            raise FileNotFoundError(f"[ERROR] Clean image not found at {clean_path}. PSNR/SSIM calculation requires paired clean images.")

        if self.transform is not None:
            img = self.transform(img)
            clean_img = self.transform(clean_img)
            
        return img, clean_img, label
