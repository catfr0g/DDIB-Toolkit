"""
CIFAR-10-C data preparation script.

Downloads and prepares CIFAR-10-C dataset for robustness evaluation.
CIFAR-10-C contains CIFAR-10 test images corrupted with 15 different
corruption types at 5 severity levels each.

The dataset is downloaded from the official source and organized into
the expected directory structure:

data/processed/CIFAR-10-C/
    gaussian_noise/
        1/
            images.npy
            labels.npy
        2/
            ...
    ...
"""

from pathlib import Path
import shutil
from typing import Optional

from loguru import logger
import numpy as np
import typer
from typer import Context

# CIFAR-10-C download URLs (from official GitHub: github.com/hendrycks/robustness)
CIFAR_10_C_URL = 'https://zenodo.org/records/2535967/files/CIFAR-10-C.tar'

# CIFAR-10-C labels are the same as CIFAR-10 test set labels
# They can be downloaded from the CIFAR-10 Python version dataset
CIFAR_10_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

# Corruption type names (in order as they appear in the data file)
CORRUPTION_NAMES = [
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

# Images per severity level
IMAGES_PER_SEVERITY = 10000  # CIFAR-10 test set size

# Expected file size for CIFAR-10-C.tar (approximately 360 MB)
EXPECTED_TAR_SIZE = 377487360  # bytes (approximate)


def download_file(
	url: str, dest_path: Path, chunk_size: int = 8192, expected_size: Optional[int] = None
) -> Path:
	"""
	Download a file from URL to destination path.

	Args:
	    url: URL to download from
	    dest_path: Destination file path
	    chunk_size: Size of chunks for streaming download
	    expected_size: Expected file size for verification

	Returns:
	    Path to downloaded file
	"""
	from urllib.error import URLError
	import urllib.request

	dest_path.parent.mkdir(parents=True, exist_ok=True)

	logger.info(f'Downloading {url} to {dest_path}')

	try:
		with urllib.request.urlopen(url) as response:
			total_size = response.getheader('Content-Length')
			total_size = int(total_size) if total_size else expected_size

			downloaded = 0
			with open(dest_path, 'wb') as f:
				while True:
					chunk = response.read(chunk_size)
					if not chunk:
						break
					f.write(chunk)
					downloaded += len(chunk)

					if total_size:
						progress = (downloaded / total_size) * 100
						if downloaded % (chunk_size * 100) == 0:
							logger.debug(f'Download progress: {progress:.1f}%')

		# Verify file size
		actual_size = dest_path.stat().st_size
		if (
			expected_size and abs(actual_size - expected_size) > expected_size * 0.01
		):  # 1% tolerance
			logger.warning(
				f'Downloaded file size ({actual_size}) differs from expected ({expected_size}). File may be corrupted.'
			)
			return dest_path

		logger.success(f'Download completed: {dest_path} ({actual_size / 1024 / 1024:.1f} MB)')
		return dest_path

	except URLError as e:
		logger.error(f'Download failed: {e}')
		raise


def extract_tar(tar_path: Path, extract_dir: Path) -> Path:
	"""
	Extract a .tar file to the specified directory.

	Args:
	    tar_path: Path to .tar file
	    extract_dir: Directory to extract to

	Returns:
	    Path to extraction directory
	"""
	import tarfile

	logger.info(f'Extracting {tar_path} to {extract_dir}')

	extract_dir.mkdir(parents=True, exist_ok=True)

	with tarfile.open(tar_path, 'r') as tar:
		tar.extractall(path=extract_dir)

	logger.success(f'Extraction completed: {extract_dir}')
	return extract_dir


def extract_cifar10_labels(data_dir: Path, force: bool = False) -> Path:
	"""
	Extract CIFAR-10 test labels from the CIFAR-10 dataset.

	CIFAR-10-C uses the same labels as CIFAR-10 test set.
	This function downloads CIFAR-10 Python dataset and extracts the test labels.

	Args:
	    data_dir: Directory to store downloaded data
	    force: Force re-download

	Returns:
	    Path to labels.npy file
	"""
	import pickle
	import tarfile

	labels_path = data_dir / 'labels.npy'
	cifar10_tar = data_dir / 'cifar-10-python.tar.gz'

	# Download CIFAR-10 if not exists
	if not cifar10_tar.exists() or force:
		download_file(CIFAR_10_URL, cifar10_tar)

	# Extract CIFAR-10
	extract_dir = data_dir / 'cifar-10-batches-py'
	if not extract_dir.exists():
		logger.info(f'Extracting CIFAR-10 to {extract_dir}')
		with tarfile.open(cifar10_tar, 'r:gz') as tar:
			tar.extractall(path=data_dir)

	# Load test labels
	test_batch = extract_dir / 'test_batch'
	if not test_batch.exists():
		# Try with .bin extension
		test_batch = extract_dir / 'test_batch.bin'

	if test_batch.exists():
		# Python version (pickle)
		with open(test_batch, 'rb') as f:
			data = pickle.load(f, encoding='bytes')
		test_labels = np.array(data[b'labels'])
	else:
		# Binary version
		test_batch_bin = extract_dir / 'test_batch'
		if not test_batch_bin.exists():
			raise FileNotFoundError(f'Cannot find CIFAR-10 test batch in {extract_dir}')

		with open(test_batch_bin, 'rb') as f:
			# Binary format: 1 byte label + 3072 bytes image, repeated 10000 times
			data = np.frombuffer(f.read(), dtype=np.uint8)
			data = data.reshape(10000, 3073)
			test_labels = data[:, 0]

	# Save labels
	np.save(labels_path, test_labels)
	logger.info(f'Saved CIFAR-10 test labels to {labels_path}')

	# Cleanup
	if cifar10_tar.exists():
		cifar10_tar.unlink()
	if extract_dir.exists():
		shutil.rmtree(extract_dir)

	return labels_path


def download_cifar10c(output_dir: Path, cleanup: bool = True, force: bool = False) -> Path:
	"""
	Download CIFAR-10-C dataset.

	Args:
	    output_dir: Directory to store downloaded data
	    cleanup: Whether to remove temporary files after extraction
	    force: Force re-download even if file exists

	Returns:
	    Path to extracted data directory
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	# Download the main tar file
	tar_path = output_dir / 'CIFAR-10-C.tar'

	# Check if tar file exists and is valid
	if tar_path.exists() and not force:
		# Verify file size (should be ~360 MB)
		actual_size = tar_path.stat().st_size
		min_expected = EXPECTED_TAR_SIZE * 0.95  # 95% of expected
		if actual_size < min_expected:
			logger.warning(
				f'Tar file size ({actual_size / 1024 / 1024:.1f} MB) is smaller than expected '
				f'({EXPECTED_TAR_SIZE / 1024 / 1024:.1f} MB). Re-downloading...'
			)
			tar_path.unlink()
		else:
			logger.info(
				f'Using existing tar file: {tar_path} ({actual_size / 1024 / 1024:.1f} MB)'
			)

	if not tar_path.exists() or force:
		download_file(CIFAR_10_C_URL, tar_path, expected_size=EXPECTED_TAR_SIZE)

	# Extract
	extract_dir = output_dir / 'CIFAR-10-C-raw'
	if not extract_dir.exists():
		extract_tar(tar_path, extract_dir)
		# Check if files are in a subdirectory (CIFAR-10-C/CIFAR-10-C/)
		nested_dir = extract_dir / 'CIFAR-10-C'
		if nested_dir.exists() and nested_dir.is_dir():
			# Move all .npy files up one level
			for npy_file in nested_dir.glob('*.npy'):
				shutil.move(str(npy_file), str(extract_dir / npy_file.name))
			# Remove nested directory
			shutil.rmtree(nested_dir)
	else:
		logger.info(f'Using existing extraction: {extract_dir}')

	# Download/extract labels
	labels_path = output_dir / 'labels.npy'
	if not labels_path.exists() or force:
		try:
			labels_path = extract_cifar10_labels(output_dir, force=force)
		except Exception as e:
			logger.error(f'Failed to extract labels from CIFAR-10: {e}')
			raise
	else:
		logger.info(f'Using existing labels file: {labels_path}')

	# Cleanup tar file if requested
	if cleanup and tar_path.exists():
		tar_path.unlink()
		logger.info(f'Removed temporary tar file: {tar_path}')

	return extract_dir


def organize_cifar10c(
	raw_data_dir: Path,
	labels_path: Path,
	output_dir: Path,
	cleanup: bool = False,
) -> Path:
	"""
	Organize CIFAR-10-C data into the expected directory structure.

	The raw data contains 15 .npy files, each with shape [50000, 32, 32, 3]
	(5 severities × 10000 images). This function splits them into separate
	files per corruption type and severity level.

	Args:
	    raw_data_dir: Directory containing raw extracted data
	    labels_path: Path to labels.npy file
	    output_dir: Output directory for organized data
	    cleanup: Whether to remove raw data after organizing

	Returns:
	    Path to organized data directory
	"""
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	logger.info('Organizing CIFAR-10-C data...')

	# Load labels
	if not labels_path.exists():
		raise FileNotFoundError(f'Labels file not found: {labels_path}')

	base_labels = np.load(labels_path)
	logger.info(f'Loaded base labels with shape: {base_labels.shape}')

	# CIFAR-10-C has 5 severity levels, each with the same 10000 images
	# So we need to replicate labels for all severity levels
	if len(base_labels) == IMAGES_PER_SEVERITY:
		# Replicate labels for all severity levels
		all_labels = np.tile(base_labels, len(SEVERITY_LEVELS))
		logger.info(
			f'Replicated labels to shape: {all_labels.shape} for {len(SEVERITY_LEVELS)} severity levels'
		)
	else:
		all_labels = base_labels
		logger.info(f'Using labels as-is with shape: {all_labels.shape}')

	# Process each corruption type
	total_files = 0
	for corruption_idx, corruption_name in enumerate(CORRUPTION_NAMES):
		logger.info(f'Processing corruption {corruption_idx + 1}/15: {corruption_name}')

		# Find the corruption file
		corruption_file = raw_data_dir / f'{corruption_name}.npy'
		if not corruption_file.exists():
			logger.warning(f'Corruption file not found: {corruption_file}')
			continue

		# Load corruption data
		data = np.load(corruption_file)
		logger.debug(f'  Data shape: {data.shape}')

		# Split by severity and save
		for severity in SEVERITY_LEVELS:
			severity_dir = output_dir / corruption_name / str(severity)
			severity_dir.mkdir(parents=True, exist_ok=True)

			# Calculate indices
			start_idx = (severity - 1) * IMAGES_PER_SEVERITY
			end_idx = severity * IMAGES_PER_SEVERITY

			# Extract images and labels for this severity
			severity_images = data[start_idx:end_idx]
			severity_labels = all_labels[start_idx:end_idx]

			# Save
			images_path = severity_dir / 'images.npy'
			labels_path_sev = severity_dir / 'labels.npy'

			np.save(images_path, severity_images)
			np.save(labels_path_sev, severity_labels)

			total_files += 2
			logger.debug(f'  Severity {severity}: Saved {len(severity_images)} images')

		# Cleanup raw file if requested
		if cleanup and corruption_file.exists():
			corruption_file.unlink()

	logger.success(f'Organization completed: {total_files} files created')

	# Cleanup raw directory if requested
	if cleanup and raw_data_dir.exists():
		shutil.rmtree(raw_data_dir)
		logger.info(f'Removed raw data directory: {raw_data_dir}')

	return output_dir


def verify_cifar10c(data_dir: Path) -> bool:
	"""
	Verify that CIFAR-10-C data is correctly organized.

	Args:
	    data_dir: Directory containing organized CIFAR-10-C data

	Returns:
	    True if verification passes, False otherwise
	"""
	logger.info(f'Verifying CIFAR-10-C data in {data_dir}')

	if not data_dir.exists():
		logger.error(f'Data directory does not exist: {data_dir}')
		return False

	errors = []

	for corruption_name in CORRUPTION_NAMES:
		corruption_dir = data_dir / corruption_name

		if not corruption_dir.exists():
			errors.append(f'Missing corruption directory: {corruption_dir}')
			continue

		for severity in SEVERITY_LEVELS:
			severity_dir = corruption_dir / str(severity)

			if not severity_dir.exists():
				errors.append(f'Missing severity directory: {severity_dir}')
				continue

			images_path = severity_dir / 'images.npy'
			labels_path = severity_dir / 'labels.npy'

			if not images_path.exists():
				errors.append(f'Missing images file: {images_path}')

			if not labels_path.exists():
				errors.append(f'Missing labels file: {labels_path}')

			# Verify shapes
			if images_path.exists():
				images = np.load(images_path)
				expected_shape = (IMAGES_PER_SEVERITY, 32, 32, 3)
				if images.shape != expected_shape:
					errors.append(
						f'Unexpected images shape at {images_path}: '
						f'{images.shape} (expected {expected_shape})'
					)

			if labels_path.exists():
				labels = np.load(labels_path)
				if len(labels) != IMAGES_PER_SEVERITY:
					errors.append(
						f'Unexpected labels length at {labels_path}: '
						f'{len(labels)} (expected {IMAGES_PER_SEVERITY})'
					)

	if errors:
		logger.error(f'Verification failed with {len(errors)} errors:')
		for error in errors[:10]:  # Show first 10 errors
			logger.error(f'  - {error}')
		if len(errors) > 10:
			logger.error(f'  ... and {len(errors) - 10} more errors')
		return False

	logger.success('Verification passed! CIFAR-10-C data is correctly organized.')
	return True


def prepare_cifar10c(
	data_dir: Optional[Path] = None,
	output_dir: Optional[Path] = None,
	cleanup: bool = True,
	skip_download: bool = False,
	skip_verify: bool = False,
	force: bool = False,
) -> Path:
	"""
	Complete CIFAR-10-C data preparation pipeline.

	Args:
	    data_dir: Directory for storing data (default: data/processed)
	    output_dir: Output directory for organized data (default: data_dir/CIFAR-10-C)
	    cleanup: Whether to remove temporary files
	    skip_download: Skip download (use existing raw data)
	    skip_verify: Skip verification after organizing
	    force: Force re-download and re-preparation

	Returns:
	    Path to organized CIFAR-10-C data directory
	"""
	if data_dir is None:
		data_dir = Path('data/processed')

	if output_dir is None:
		output_dir = data_dir / 'CIFAR-10-C'

	data_dir = Path(data_dir).resolve()
	output_dir = Path(output_dir).resolve()

	logger.info('=' * 60)
	logger.info('CIFAR-10-C Data Preparation')
	logger.info('=' * 60)
	logger.info(f'Data directory: {data_dir}')
	logger.info(f'Output directory: {output_dir}')

	# Check if already prepared (skip if not force)
	if output_dir.exists() and verify_cifar10c(output_dir) and not force:
		logger.info('CIFAR-10-C data already exists and is valid. Skipping preparation.')
		return output_dir

	# Create directories
	data_dir.mkdir(parents=True, exist_ok=True)
	raw_dir = data_dir / 'CIFAR-10-C-raw'
	labels_path = data_dir / 'labels.npy'

	# Download
	if not skip_download:
		raw_data_dir = download_cifar10c(data_dir, cleanup=False, force=force)
	else:
		raw_data_dir = raw_dir
		if not raw_data_dir.exists():
			raise FileNotFoundError(
				f'Raw data directory not found: {raw_data_dir}. '
				'Set skip_download=False to download.'
			)

	# Organize
	organize_cifar10c(
		raw_data_dir=raw_data_dir,
		labels_path=labels_path,
		output_dir=output_dir,
		cleanup=cleanup,
	)

	# Verify
	if not skip_verify:
		if not verify_cifar10c(output_dir):
			raise RuntimeError('Verification failed after organization')

	logger.success('=' * 60)
	logger.success('CIFAR-10-C preparation completed successfully!')
	logger.success('=' * 60)

	return output_dir


app = typer.Typer()


@app.callback(invoke_without_command=True)
def main(
	ctx: Context,
	data_dir: Path = typer.Option(
		Path('data/processed'),
		'--data-dir',
		'-d',
		help='Directory to store data',
	),
	output_dir: Optional[Path] = typer.Option(
		None,
		'--output-dir',
		'-o',
		help='Output directory (default: data_dir/CIFAR-10-C)',
	),
	cleanup: bool = typer.Option(
		True,
		'--cleanup/--no-cleanup',
		help='Remove temporary files after processing',
	),
	skip_download: bool = typer.Option(
		False,
		'--skip-download',
		help='Skip download (use existing raw data)',
	),
	skip_verify: bool = typer.Option(
		False,
		'--skip-verify',
		help='Skip verification after organizing',
	),
	force: bool = typer.Option(
		False,
		'--force',
		'-f',
		help='Force re-preparation even if data exists',
	),
):
	"""
	Download and prepare CIFAR-10-C dataset for robustness evaluation.

	This script downloads CIFAR-10-C from the official source and organizes
	it into the expected directory structure for robustness evaluation.
	"""
	if ctx.invoked_subcommand is not None:
		return

	output_path = Path(output_dir) if output_dir else data_dir / 'CIFAR-10-C'

	# Remove existing data if force is set
	if force and output_path.exists():
		logger.warning(f'Removing existing data: {output_path}')
		import shutil

		shutil.rmtree(output_path)

	prepare_cifar10c(
		data_dir=data_dir,
		output_dir=output_path,
		cleanup=cleanup,
		skip_download=skip_download,
		skip_verify=skip_verify,
	)


@app.command()
def verify(
	data_dir: Path = typer.Option(
		Path('data/processed/CIFAR-10-C'),
		'--data-dir',
		'-d',
		help='Directory containing CIFAR-10-C data',
	),
):
	"""
	Verify CIFAR-10-C data integrity.
	"""
	success = verify_cifar10c(data_dir)
	if not success:
		raise typer.Exit(1)


@app.command()
def download(
	output_dir: Path = typer.Option(
		Path('data/processed'),
		'--output-dir',
		'-o',
		help='Directory to store downloaded data',
	),
	cleanup: bool = typer.Option(
		True,
		'--cleanup/--no-cleanup',
		help='Remove tar file after extraction',
	),
):
	"""
	Download CIFAR-10-C without organizing.
	"""
	download_cifar10c(output_dir, cleanup=cleanup)


if __name__ == '__main__':
	app()
