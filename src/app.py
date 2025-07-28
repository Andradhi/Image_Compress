"""
SVD Image Compression Web Application
Mata Kuliah: Aljabar Linear
Program Studi: Informatika
Universitas Sebelas Maret

Aplikasi web untuk kompresi gambar menggunakan algoritma Singular Value Decomposition (SVD)
"""

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import cv2
from PIL import Image
import io
import base64
import time
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Membuat folder uploads jika belum ada
# dari (Andra) nanti kalo di laptop kalian belum ada folder upload bakal buat yang isinya image yang kalian masukin
if not os.path.exists('uploads'):
    os.makedirs('uploads')

class SVDImageCompressor:
    """
    Kelas untuk melakukan kompresi gambar menggunakan algoritma SVD
    """
    
    def __init__(self):
        self.original_image = None
        self.compressed_image = None
        self.compression_time = 0
        
    def load_image(self, image_path):
        """
        Memuat gambar dari file path
        
        Args:
            image_path (str): Path ke file gambar
            
        Returns:
            numpy.ndarray: Array gambar yang dimuat
        """
        try:
            # Membaca gambar menggunakan PIL untuk menjalankan berbagai format intinya smeua format bisa
            pil_image = Image.open(image_path)
            
            # Konversi ke RGB jika gambar dalam mode lain
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
                
            # Konversi ke numpy array
            image_array = np.array(pil_image)
            self.original_image = image_array
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Error loading image: {str(e)}")
    
    def compress_image_svd(self, image_array, compression_rate):
        """
        Melakukan kompresi gambar menggunakan SVD
        
        Args:
            image_array (numpy.ndarray): Array gambar yang akan dikompresi
            compression_rate (float): Tingkat kompresi (0.01 - 1.0)
            
        Returns:
            numpy.ndarray: Array gambar yang telah dikompresi
        """
        start_time = time.time()
        
        try:
            # Mendapatkan dimensi gambar
            height, width, channels = image_array.shape
            max_rank = min(height, width)
            k = int(max_rank * compression_rate)
            k = max(1, k)  #\
            compressed_channels = []
            
            for channel in range(channels):
                # Mengambil matrix untuk channel saat ini
                channel_matrix = image_array[:, :, channel].astype(np.float64)
                
                # Melakukan SVD decomposition
                # A = U * S * V^T
                U, S, Vt = np.linalg.svd(channel_matrix, full_matrices=False)
                
                # Mengambil k singular values terbesar
                U_k = U[:, :k]
                S_k = S[:k]
                Vt_k = Vt[:k, :]
                
                # Merekonstruksi matrix dengan rank k
                compressed_channel = np.dot(U_k, np.dot(np.diag(S_k), Vt_k))
                
                # Memastikan nilai pixel dalam range [0, 255]
                compressed_channel = np.clip(compressed_channel, 0, 255)
                compressed_channels.append(compressed_channel)
            compressed_image = np.stack(compressed_channels, axis=2).astype(np.uint8)
            
            self.compression_time = time.time() - start_time
            self.compressed_image = compressed_image
            
            return compressed_image
            
        except Exception as e:
            raise Exception(f"Error during SVD compression: {str(e)}")
    
    def calculate_compression_ratio(self):
        """
        Menghitung rasio kompresi berdasarkan perubahan ukuran data
        
        Returns:
            dict: Informasi statistik kompresi
        """
        if self.original_image is None or self.compressed_image is None:
            return None
            
        original_size = self.original_image.size
        compressed_size = self.compressed_image.size

        return {
            'original_pixels': original_size,
            'compressed_pixels': compressed_size,
            'compression_time': round(self.compression_time, 4)
        }
    
    def save_compressed_image(self, output_path):
        """
        Menyimpan gambar yang telah dikompresi
        
        Args:
            output_path (str): Path untuk menyimpan gambar hasil kompresi
        """
        if self.compressed_image is None:
            raise Exception("No compressed image available")
            
      
        pil_image = Image.fromarray(self.compressed_image)
        pil_image.save(output_path, quality=95)
compressor = SVDImageCompressor()

def array_to_base64(image_array):
    """
    Konversi numpy array ke base64 string untuk ditampilkan di web
    
    Args:
        image_array (numpy.ndarray): Array gambar
        
    Returns:
        str: Base64 encoded string
    """
    pil_image = Image.fromarray(image_array)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    encoded_string = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded_string}"

@app.route('/')
def index():
    """Route untuk halaman utama"""
    return render_template('index.html')

@app.route('/compress', methods=['POST'])
def compress_image():
    """Route untuk melakukan kompresi gambar"""
    try:
        # Validasi input
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
            
        if 'compression_rate' not in request.form:
            return jsonify({'error': 'No compression rate provided'}), 400
        
        image_file = request.files['image']
        compression_rate = float(request.form['compression_rate']) / 100.0  # Convert percentage to decimal
        
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
            
        if compression_rate < 0.01 or compression_rate > 1.0:
            return jsonify({'error': 'Compression rate must be between 1% and 100%'}), 400
        
        # Simpan file upload sementara
        filename = secure_filename(image_file.filename)
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        image_file.save(temp_path)
        
        # Load dan kompresi gambar
        original_image = compressor.load_image(temp_path)
        compressed_image = compressor.compress_image_svd(original_image, compression_rate)
    
        stats = compressor.calculate_compression_ratio()
        
        # Konversi gambar ke base64 untuk ditampilkan
        original_b64 = array_to_base64(original_image)
        compressed_b64 = array_to_base64(compressed_image)
        
        # Simpan hasil kompresi untuk download
        output_filename = f"compressed_{filename}"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
        compressor.save_compressed_image(output_path)
   
        os.remove(temp_path)
        
        response_data = {
            'success': True,
            'original_image': original_b64,
            'compressed_image': compressed_b64,
            'compression_time': stats['compression_time'],
            'compression_rate_used': compression_rate * 100,
            'download_filename': output_filename
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Route untuk download file hasil kompresi"""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting SVD Image Compression Application...")
    print("Aplikasi dapat diakses di: http://localhost:5000")
    print("Tekan Ctrl+C untuk menghentikan aplikasi")
    
    app.run(debug=True, host='0.0.0.0', port=5000)