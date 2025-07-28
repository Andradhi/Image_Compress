from flask import Flask, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename
import os
import time
from svd_compressor import compress_image_svd # Import fungsi dari svd_compressor.py

app = Flask(__name__)

UPLOAD_FOLDER_ORIGINAL = 'static/uploads/original'
UPLOAD_FOLDER_COMPRESSED = 'static/uploads/compressed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

app.config['UPLOAD_FOLDER_ORIGINAL'] = UPLOAD_FOLDER_ORIGINAL
app.config['UPLOAD_FOLDER_COMPRESSED'] = UPLOAD_FOLDER_COMPRESSED
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB file size

# Pastikan direktori upload ada
os.makedirs(UPLOAD_FOLDER_ORIGINAL, exist_ok=True)
os.makedirs(UPLOAD_FOLDER_COMPRESSED, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    original_image_url = None
    compressed_image_url = None
    runtime = None
    pixel_diff = None

    if request.method == 'POST':
        # Cek apakah ada file di request
        if 'image_file' not in request.files:
            return render_template('index.html', message='No image file part')
        
        file = request.files['image_file']
        
        # Jika user tidak memilih file
        if file.filename == '':
            return render_template('index.html', message='No selected file')
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            original_image_path = os.path.join(app.config['UPLOAD_FOLDER_ORIGINAL'], filename)
            file.save(original_image_path)
            original_image_url = url_for('static', filename=f'uploads/original/{filename}')

            try:
                compression_rate = int(request.form['compression_rate'])
                if not (1 <= compression_rate <= 100):
                    return render_template('index.html', message='Compression rate must be between 1 and 100.')
            except ValueError:
                return render_template('index.html', message='Invalid compression rate. Please enter a number.')
            
            # Panggil fungsi kompresi SVD
            compressed_path, runtime, pixel_diff = compress_image_svd(original_image_path, compression_rate)

            if compressed_path:
                compressed_filename = os.path.basename(compressed_path)
                compressed_image_url = url_for('static', filename=f'uploads/compressed/{compressed_filename}')
            else:
                return render_template('index.html', message='Image compression failed.')

    return render_template('index.html', 
                           original_image_url=original_image_url,
                           compressed_image_url=compressed_image_url,
                           runtime=runtime,
                           pixel_diff=pixel_diff)

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER_COMPRESSED'], filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    else:
        return "File not found!", 404

if __name__ == '__main__':
    app.run(debug=True) # debug=True untuk pengembangan, nonaktifkan untuk produksi