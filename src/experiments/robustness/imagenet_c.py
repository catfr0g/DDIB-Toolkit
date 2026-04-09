"""
ImageNet-C dataset loader for robustness evaluation.

ImageNet-C contains 15 corruption types at 5 severity levels.
This module provides utilities to load and evaluate on ImageNet-C.
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from loguru import logger
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ImageNet-C corruption types
CORRUPTION_TYPES = [
	'gaussian_noise',
	'shot_noise',
	'impulse_noise',
	'defocus_blur',
	'glass_blur',
	'motion_blur',
	'zoom_blur',
	'snow',
	'frost',
	'fog',
	'brightness',
	'contrast',
	'elastic_transform',
	'pixelate',
	'jpeg_compression',
]

# Severity levels
SEVERITY_LEVELS = [1, 2, 3, 4, 5]

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class ImageNetCDataset(Dataset):
	"""
	ImageNet-C dataset for robustness evaluation.

	The dataset should be organized as:
	data_dir/
	    corruption_type/
	        severity_level/
	            class_1/
	                image1.JPEG
	                ...
	            class_2/
	                ...

	Args:
	    data_dir: Root directory containing ImageNet-C data
	    corruption_types: List of corruption types to include (default: all)
	    severity_levels: List of severity levels to include (default: all)
	    transform: Optional transform to apply to images
	"""

	def __init__(
		self,
		data_dir: Path,
		corruption_types: Optional[List[str]] = None,
		severity_levels: Optional[List[int]] = None,
		transform: Optional[Callable] = None,
	):
		self.data_dir = Path(data_dir)
		self.corruption_types = corruption_types or CORRUPTION_TYPES
		self.severity_levels = severity_levels or SEVERITY_LEVELS
		self.transform = transform or transforms.Compose(
			[
				transforms.Resize(256),
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
			]
		)

		self.samples: List[Tuple[Path, int, str, int]] = []
		self._load_samples()

		logger.info(
			f'Loaded {len(self.samples)} samples from ImageNet-C '
			f'({len(self.corruption_types)} corruptions × {len(self.severity_levels)} severity levels)'
		)

	def _load_samples(self):
		"""Load all image paths with their labels and metadata."""
		for corruption_type in self.corruption_types:
			for severity in self.severity_levels:
				corruption_dir = self.data_dir / corruption_type / str(severity)

				if not corruption_dir.exists():
					logger.warning(
						f'Directory not found: {corruption_dir}. '
						'Skipping this corruption/severity combination.'
					)
					continue

				# Iterate through class directories
				for class_dir in corruption_dir.iterdir():
					if not class_dir.is_dir():
						continue

					# Extract class index from directory name
					try:
						class_idx = int(class_dir.name)
					except ValueError:
						logger.warning(f'Invalid class directory: {class_dir.name}')
						continue

					# Load all images in the class directory
					for img_path in class_dir.glob('*.JPEG'):
						self.samples.append((img_path, class_idx, corruption_type, severity))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
		img_path, label, corruption_type, severity = self.samples[idx]

		# Load image
		image = Image.open(img_path).convert('RGB')

		if self.transform:
			image = self.transform(image)

		return image, label, corruption_type, severity


class ImageNetCCIFAR10Dataset(Dataset):
	"""
	CIFAR-10-C dataset for robustness evaluation on CIFAR-10 models.

	CIFAR-10-C applies the same corruption types as ImageNet-C to CIFAR-10 test images.
	The dataset should be organized as:
	data_dir/
	    corruption_type/
	        severity_level/
	            images.npy
	            labels.npy

	Args:
	    data_dir: Root directory containing CIFAR-10-C data
	    corruption_types: List of corruption types to include (default: all)
	    severity_levels: List of severity levels to include (default: all)
	    transform: Optional transform to apply to images
	"""

	def __init__(
		self,
		data_dir: Path,
		corruption_types: Optional[List[str]] = None,
		severity_levels: Optional[List[int]] = None,
		transform: Optional[Callable] = None,
	):
		self.data_dir = Path(data_dir)
		self.corruption_types = corruption_types or CORRUPTION_TYPES
		self.severity_levels = severity_levels or SEVERITY_LEVELS

		# CIFAR-10 normalization
		self.transform = transform or transforms.Compose(
			[
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.4914, 0.4822, 0.4465),
					std=(0.2023, 0.1994, 0.2010),
				),
			]
		)

		self.samples: List[Tuple[np.ndarray, int, str, int]] = []
		self._load_samples()

		logger.info(
			f'Loaded {len(self.samples)} samples from CIFAR-10-C '
			f'({len(self.corruption_types)} corruptions × {len(self.severity_levels)} severity levels)'
		)

	def _load_samples(self):
		"""Load all corrupted images with their labels and metadata."""
		for corruption_type in self.corruption_types:
			for severity in self.severity_levels:
				corruption_dir = self.data_dir / corruption_type / str(severity)

				if not corruption_dir.exists():
					logger.warning(
						f'Directory not found: {corruption_dir}. '
						'Skipping this corruption/severity combination.'
					)
					continue

				images_path = corruption_dir / 'images.npy'
				labels_path = corruption_dir / 'labels.npy'

				if not images_path.exists() or not labels_path.exists():
					# Try alternative format: single .npy file per corruption
					alt_path = self.data_dir / f'{corruption_type}.npy'
					if alt_path.exists():
						# Load and extract severity level data
						data = np.load(alt_path)
						# CIFAR-10-C format: [10000, 32, 32, 3] for each severity
						# 5 severities stacked: [50000, 32, 32, 3]
						start_idx = (severity - 1) * 10000
						end_idx = severity * 10000
						images = data[start_idx:end_idx]

						# Load labels
						labels_path = self.data_dir / 'labels.npy'
						if labels_path.exists():
							labels = np.load(labels_path)
							labels = labels[start_idx:end_idx]
						else:
							logger.warning(f'Labels not found for {corruption_type}')
							continue
					else:
						logger.warning(f'Data not found: {corruption_dir}')
						continue
				else:
					images = np.load(images_path)
					labels = np.load(labels_path)

				# Add samples
				for i in range(len(images)):
					self.samples.append((images[i], int(labels[i]), corruption_type, severity))

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str, int]:
		image, label, corruption_type, severity = self.samples[idx]

		# Convert numpy array to PIL Image for consistent transform application
		image = Image.fromarray(image)

		if self.transform:
			image = self.transform(image)

		return image, label, corruption_type, severity


def create_imagenet_c_dataloader(
	data_dir: Path,
	corruption_types: Optional[List[str]] = None,
	severity_levels: Optional[List[int]] = None,
	batch_size: int = 64,
	num_workers: int = 4,
	transform: Optional[Callable] = None,
) -> DataLoader:
	"""
	Create a DataLoader for ImageNet-C dataset.

	Args:
	    data_dir: Root directory containing ImageNet-C data
	    corruption_types: List of corruption types to include
	    severity_levels: List of severity levels to include
	    batch_size: Batch size for data loading
	    num_workers: Number of workers for data loading
	    transform: Optional transform to apply to images

	Returns:
	    DataLoader for ImageNet-C
	"""
	dataset = ImageNetCDataset(
		data_dir=data_dir,
		corruption_types=corruption_types,
		severity_levels=severity_levels,
		transform=transform,
	)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)


def create_cifar10_c_dataloader(
	data_dir: Path,
	corruption_types: Optional[List[str]] = None,
	severity_levels: Optional[List[int]] = None,
	batch_size: int = 64,
	num_workers: int = 4,
	transform: Optional[Callable] = None,
) -> DataLoader:
	"""
	Create a DataLoader for CIFAR-10-C dataset.

	Args:
	    data_dir: Root directory containing CIFAR-10-C data
	    corruption_types: List of corruption types to include
	    severity_levels: List of severity levels to include
	    batch_size: Batch size for data loading
	    num_workers: Number of workers for data loading
	    transform: Optional transform to apply to images

	Returns:
	    DataLoader for CIFAR-10-C
	"""
	dataset = ImageNetCCIFAR10Dataset(
		data_dir=data_dir,
		corruption_types=corruption_types,
		severity_levels=severity_levels,
		transform=transform,
	)

	return DataLoader(
		dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)


def get_default_corruptions() -> List[str]:
	"""Return list of all ImageNet-C corruption types."""
	return CORRUPTION_TYPES.copy()


def get_default_severities() -> List[int]:
	"""Return list of all severity levels."""
	return SEVERITY_LEVELS.copy()
