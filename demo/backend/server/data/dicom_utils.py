
import os
from PIL import Image
import numpy as np
import glob
import os
import zipfile
import glob
import numpy as np
import pydicom
from PIL import Image
import tempfile
import shutil
import imageio
from strawberry.file_uploads import Upload

def convert_dicoms_to_video(file, video_images_dir="jpgs", ww=None, wl=None):
    tempdir = tempfile.mkdtemp()

    zip_file_path = f"{tempdir}/in.zip"
    # out_path = f"{tempdir}/out.mp4"
    with open(zip_file_path, "wb") as in_f:
        in_f.write(file.read())

    # Check if the file exists and has a non-zero size
    if not os.path.isfile(zip_file_path):
        raise ValueError("Invalid file")
    
    if os.path.getsize(zip_file_path) == 0:
        raise ValueError("file size is 0!")

    # Create temporary directories for extracted DICOMs
    temp_dir = tempfile.mkdtemp()
    dcm_dir = os.path.join(temp_dir, 'dicoms')
    os.makedirs(dcm_dir, exist_ok=True)
    os.makedirs(video_images_dir, exist_ok=True)

    # Extract the zip file
    with zipfile.ZipFile(zip_file_path) as zip_ref:
        zip_ref.extractall(dcm_dir)

    # Load DICOM files
    dicom_filenames = glob.glob(os.path.join(dcm_dir, '*.dcm'))
    files = [pydicom.dcmread(fname) for fname in dicom_filenames if fname.endswith('.dcm')]

    # Sort slices based on DICOM metadata
    def sort_key(s):
        if hasattr(s, 'SliceLocation'):
            return s.SliceLocation
        elif hasattr(s, 'InstanceNumber'):
            return s.InstanceNumber
        elif hasattr(s, 'ImagePositionPatient'):
            return s.ImagePositionPatient[2]
        else:
            return float('inf')  # Put files with no relevant tag at the end

    slices = sorted(files, key=sort_key)

    # Prepare 3D array of pixel data
    img_shape = list(slices[0].pixel_array.shape) + [len(slices)]
    img3d = np.zeros(img_shape)

    # Fill the 3D array with pixel data, applying normalization and WW/WL adjustments
    for i, s in enumerate(slices):
        img2d = s.pixel_array.astype(np.float32)
        slope = getattr(s, 'RescaleSlope', 1)
        intercept = getattr(s, 'RescaleIntercept', 0)
        img2d = img2d * slope + intercept

        if ww and wl:
            ww = float(ww)
            wl = float(wl)
            min_val = wl - ww / 2
            max_val = wl + ww / 2
            img2d = np.clip(img2d, min_val, max_val)
            img2d = (img2d - min_val) / (max_val - min_val) * 255

        img3d[:, :, i] = img2d

    # Normalize the 3D array
    #non_zero_values = img3d[img3d != 0]
    #min_val = int(np.min(non_zero_values)) + 100
    #max_val = int(0.67 * np.max(non_zero_values))
    #img3d_normalized = np.clip(img3d, min_val, max_val)
    #img3d_normalized = 255 * (img3d_normalized - min_val) / (max_val - min_val)
    #img3d_normalized = img3d_normalized.astype(np.uint8)

    # Convert slices to JPG and save in video_images_dir
    for idx in range(img3d.shape[2]):
        image_array = img3d[:, :, idx]
        image = Image.fromarray(image_array).convert("RGB")
        image.save(os.path.join(video_images_dir, f"{idx}.jpg"), quality=100)

    out_path = os.path.join(video_images_dir, "output.mp4")
    image_files = sorted(glob.glob(os.path.join(video_images_dir, '*.jpg')), key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
    images = [imageio.imread(img_file) for img_file in image_files]
    imageio.mimsave(out_path, images, fps=16, codec='libx264')

    f = open(out_path, "rb")
    upload_file = Upload(f)
    shutil.rmtree(tempdir)
    # Cleanup temporary directories
    print(f"Converted DICOM files to JPGs in {video_images_dir}")
    return upload_file

