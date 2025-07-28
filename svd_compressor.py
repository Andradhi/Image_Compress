import numpy as np
from PIL import Image
import time
import os

def compress_image_svd(image_path, compression_rate_percentage, output_dir="static/uploads/compressed/"):
    """
    Mengkompresi gambar menggunakan Singular Value Decomposition (SVD).

    Args:
        image_path (str): Jalur lengkap ke file gambar input.
        compression_rate_percentage (int): Tingkat kompresi dalam persentase (1-100%).
                                            Ini akan digunakan untuk menentukan jumlah singular values.
        output_dir (str): Direktori untuk menyimpan gambar hasil kompresi.

    Returns:
        tuple: (compressed_image_path, runtime_ms, pixel_difference_percentage)
               - compressed_image_path (str): Jalur ke gambar hasil kompresi.
               - runtime_ms (float): Waktu eksekusi kompresi dalam milidetik.
               - pixel_difference_percentage (float): Persentase perbedaan pixel.
                                                     (Didefinisikan sebagai 100 * (1 - (jumlah singular value / min(width, height))))
    """
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)

    try:
        img = Image.open(image_path)
        img_array = np.array(img)
    except FileNotFoundError:
        print(f"Error: File not found at {image_path}")
        return None, 0, 0
    except Exception as e:
        print(f"Error opening image: {e}")
        return None, 0, 0

    # Pastikan gambar adalah RGB (3 channel)
    if img_array.ndim == 2: # Grayscale
        img_array = np.stack([img_array, img_array, img_array], axis=-1)
    elif img_array.ndim == 3 and img_array.shape[2] == 4: # RGBA
        # Konversi RGBA ke RGB (buang channel alpha)
        img_array = img_array[:, :, :3]

    if img_array.ndim != 3 or img_array.shape[2] != 3:
        print(f"Warning: Unexpected image array shape: {img_array.shape}. Attempting to proceed.")
        # Fallback for unexpected shapes, e.g., already compressed images or unusual formats
        # We assume it's a 3-channel image even if the shape suggests otherwise, for SVD
        if img_array.ndim == 2:
            img_array = np.stack([img_array, img_array, img_array], axis=-1)
        elif img_array.ndim == 3 and img_array.shape[2] > 3:
            img_array = img_array[:,:,:3] # Take first 3 channels

    height, width, channels = img_array.shape
    
    
    max_k = min(height, width)
    k = int(max_k * (compression_rate_percentage / 100.0))
    

    k = max(1, k)
    k = min(max_k, k)

    compressed_channels = []
    for i in range(channels):
        channel = img_array[:, :, i]
        U, s, V = np.linalg.svd(channel, full_matrices=False)
        
        # Rekonstruksi channel dengan k singular values
        reconstructed_channel = U[:, :k] @ np.diag(s[:k]) @ V[:k, :]
        compressed_channels.append(reconstructed_channel)

    compressed_img_array = np.stack(compressed_channels, axis=-1)
    
    # Pastikan nilai piksel dalam rentang 0-255 dan tipe data uint8
    compressed_img_array = np.clip(compressed_img_array, 0, 255).astype(np.uint8)

    compressed_img = Image.fromarray(compressed_img_array)

    # Buat nama file output
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    output_filename = f"{name}_compressed_{compression_rate_percentage}{ext}"
    compressed_image_path = os.path.join(output_dir, output_filename)
    
    # Simpan gambar terkompresi
    compressed_img.save(compressed_image_path)

    end_time = time.time()
    runtime_ms = (end_time - start_time) * 1000

    pixel_difference_percentage = 100.0 * (1.0 - (k / max_k))
    
    return compressed_image_path, runtime_ms, pixel_difference_percentage

if __name__ == '__main__':

    dummy_img_array = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)
    dummy_img = Image.fromarray(dummy_img_array)
    dummy_img_path = "static/uploads/original/dummy_test_image.png"
    os.makedirs(os.path.dirname(dummy_img_path), exist_ok=True)
    dummy_img.save(dummy_img_path)
    print(f"Dummy image saved to {dummy_img_path}")

    # Uji kompresi
    output_path, runtime, diff_perc = compress_image_svd(dummy_img_path, 50) # Kompresi 50%
    if output_path:
        print(f"Compressed image saved to: {output_path}")
        print(f"Compression Runtime: {runtime:.2f} ms")
        print(f"Pixel Difference Percentage: {diff_perc:.2f}%")

    output_path_high, runtime_high, diff_perc_high = compress_image_svd(dummy_img_path, 90) # Kompresi 90%
    if output_path_high:
        print(f"Compressed image (high quality) saved to: {output_path_high}")
        print(f"Compression Runtime (high quality): {runtime_high:.2f} ms")
        print(f"Pixel Difference Percentage (high quality): {diff_perc_high:.2f}%")
