# Smart Market Analytics 📊

Aplikasi analisis data pasar tradisional yang komprehensif untuk Kabupaten Sumedang, dilengkapi dengan fitur Exploratory Data Analysis (EDA) dan K-Means Clustering Analysis.

## 🚀 Fitur Utama

### 📊 Exploratory Data Analysis (EDA)
- **Visualisasi Distribusi**: Analisis distribusi jumlah stok dan kebutuhan barang
- **Analisis Korelasi**: Scatter plot untuk melihat hubungan antara stok dan kebutuhan
- **Top Items Analysis**: Identifikasi barang dengan kebutuhan tertinggi
- **Market Comparison**: Perbandingan performa antar pasar tradisional
- **Insight Otomatis**: Rekomendasi berdasarkan pola data

### 🎯 K-Means Clustering Analysis
- **Clustering Otomatis**: Pengelompokan data berdasarkan pola jumlah dan kebutuhan
- **Visualisasi Interaktif**: Scatter plot dengan Plotly untuk eksplorasi cluster
- **Analisis Cluster**: Insight mendalam untuk setiap cluster yang terbentuk
- **Parameter Tuning**: Slider untuk mengatur jumlah cluster (K = 2-10)

### 📄 Professional PDF Reports
- **Format Pemerintahan**: Header resmi Pemerintah Kabupaten Sumedang
- **Laporan Komprehensif**: Ringkasan eksekutif, analisis detail, dan rekomendasi
- **Export Otomatis**: Download laporan dalam format PDF profesional

## 🛠️ Teknologi yang Digunakan

- **Streamlit**: Framework aplikasi web interaktif
- **Pandas**: Manipulasi dan analisis data
- **Scikit-learn**: Machine learning (K-Means clustering)
- **Matplotlib & Seaborn**: Visualisasi data statis
- **Plotly**: Visualisasi data interaktif
- **FPDF**: Generasi laporan PDF
- **NumPy**: Operasi numerik

## 📋 Persyaratan Sistem

### Dependencies
```
streamlit
pandas
seaborn
matplotlib
plotly
scikit-learn
fpdf2
numpy
```

### Python Version
- Python 3.7 atau lebih baru

## 🚀 Quick Start

### ⚡ Fastest Way (No Installation Required)
1. **Akses langsung**: [Smart Market Analytics](https://smart-market-analytics-lai25-sm037.streamlit.app/)
2. **Upload CSV**: Drag & drop file CSV Anda
3. **Mulai Analisis**: Pilih EDA atau Clustering
4. **Download Report**: Export hasil analisis dalam PDF

## 🔧 Instalasi (Opsional - Untuk Pengembangan Lokal)

1. **Clone repository**
```bash
git clone <repository-url>
cd smart-market-analytics
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Jalankan aplikasi**
```bash
streamlit run app.py
```

4. **Akses aplikasi**
   - Buka browser dan kunjungi: `http://localhost:8501`

## 🌐 Demo Online

Aplikasi ini telah di-deploy dan dapat diakses secara online:

### 🔗 Live Demo
**[Smart Market Analytics - Live Application](https://smart-market-analytics-lai25-sm037.streamlit.app/)**

> ⚡ **Quick Access**: Langsung gunakan aplikasi tanpa instalasi lokal  
> 🔒 **Secure**: Data diproses secara aman di cloud Streamlit  
> 📱 **Responsive**: Dapat diakses dari desktop, tablet, atau mobile  

### Akses Cepat
- **URL**: `https://smart-market-analytics-lai25-sm037.streamlit.app/`
- **Status**: ✅ Online & Ready to Use
- **Uptime**: 24/7 Available
- **Support**: Multi-device compatible

## 📊 Format Data

### Struktur CSV yang Diperlukan
File CSV harus memiliki kolom-kolom berikut:

| Kolom | Tipe Data | Deskripsi |
|-------|-----------|-----------|
| `nama_pasar` | String | Nama pasar tradisional |
| `item_barang` | String | Nama/jenis barang |
| `jumlah` | Integer | Jumlah stok barang (dalam kg) |
| `kebutuhan` | Integer | Kebutuhan barang (dalam kg) |
| `tanggal` | Date | Tanggal pencatatan (YYYY-MM-DD) |
| `satuan_item` | String | Satuan barang (harus 'kg') |

### Contoh Data
```csv
nama_pasar,item_barang,jumlah,kebutuhan,tanggal,satuan_item
Pasar Sumedang,Beras,1000,1200,2024-01-15,kg
Pasar Tanjungsari,Gula,500,450,2024-01-15,kg
Pasar Situraja,Minyak Goreng,300,350,2024-01-15,kg
```

## 🎨 Panduan Penggunaan

### 🌐 Akses Online (Recommended)
1. **Kunjungi aplikasi online**: [Smart Market Analytics Live Demo](https://smart-market-analytics-lai25-sm037.streamlit.app/)
2. **Upload file CSV** langsung di browser
3. **Mulai analisis** tanpa instalasi apapun

### 💻 Local Installation (Optional)
Jika ingin menjalankan secara lokal, ikuti [panduan instalasi](#🔧-instalasi) di atas.

### 1. Upload Data
- Klik tombol "📁 Unggah file CSV Anda"
- Pilih file CSV dengan format yang sesuai
- Sistem akan otomatis memvalidasi dan membersihkan data

### 2. Exploratory Data Analysis
- Gunakan sidebar untuk memfilter data berdasarkan:
  - Pasar yang ingin dianalisis
  - Jenis barang
  - Periode tahun
  - Periode bulan
- Lihat visualisasi dan insight yang dihasilkan secara otomatis

### 3. K-Means Clustering
- Pilih menu "🎯 K-Means Clustering"
- Atur parameter clustering menggunakan slider
- Analisis hasil clustering dan interpretasi bisnis
- Download laporan PDF jika diperlukan

### 4. Filter Data
Aplikasi menyediakan filter komprehensif:
- **Multi-select filter** untuk pasar, barang, tahun, dan bulan
- **Default selection** yang cerdas untuk performa optimal
- **Real-time filtering** yang langsung memperbarui visualisasi

## 📈 Interpretasi Hasil

### EDA Insights
- **Status Stok**: Indikator apakah terjadi overstock atau kekurangan
- **Korelasi**: Mengukur seberapa baik stok mengikuti pola kebutuhan
- **Top Items**: Identifikasi barang dengan kebutuhan tertinggi
- **Market Performance**: Perbandingan performa antar pasar

### Clustering Insights
- **Cluster Tinggi**: Barang/pasar dengan kebutuhan tinggi
- **Cluster Rendah**: Barang/pasar dengan kebutuhan rendah  
- **Pattern Recognition**: Identifikasi pola tersembunyi dalam data
- **Strategic Grouping**: Pengelompokan untuk strategi bisnis

## 🏛️ Konfigurasi Pemerintahan

Aplikasi ini dikonfigurasi khusus untuk:
- **Instansi**: Dinas Koperasi, UKM, Perdagangan dan Perindustrian
- **Wilayah**: Kabupaten Sumedang, Jawa Barat
- **Format Laporan**: Sesuai standar dokumen pemerintahan

### Customization
Untuk menyesuaikan dengan instansi lain, edit bagian berikut di `app.py`:
```python
# Di dalam class ReportPDF, method header()
self.cell(0, 6, "PEMERINTAH KABUPATEN [NAMA_KABUPATEN]", ln=True, align='L')
self.cell(0, 5, "DINAS [NAMA_DINAS]", ln=True, align='L')
```

## 🔒 Keamanan Data

- **Pemrosesan Lokal**: Semua data diproses secara lokal di server
- **No Data Storage**: Aplikasi tidak menyimpan data secara permanen
- **Session-based**: Data hanya tersedia selama sesi aktif

## 🚨 Troubleshooting

### Error Loading Data
- Pastikan file CSV memiliki kolom yang diperlukan
- Periksa format tanggal (YYYY-MM-DD)
- Pastikan kolom numerik tidak mengandung karakter non-angka

### Performance Issues
- Batasi jumlah data untuk dataset besar (> 100k records)
- Gunakan filter untuk mengurangi data yang diproses
- Tutup tab browser yang tidak diperlukan

### PDF Generation Failed
- Pastikan memori sistem mencukupi
- Reduce jumlah cluster jika data terlalu besar
- Periksa koneksi internet untuk dependencies

## 🤝 Kontribusi

Untuk berkontribusi pada proyek ini:
1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

## 📝 Lisensi

Proyek ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 📞 Dukungan

### 🌐 Akses Aplikasi
- **Live Application**: [https://your-app-url.streamlit.app](https://smart-market-analytics-lai25-sm037.streamlit.app/)
- **Status Page**: Monitor uptime dan performa aplikasi
- **Mobile Access**: Akses dari smartphone atau tablet

### 💬 Dukungan Teknis
Untuk dukungan teknis atau pertanyaan:
- **Email**: sendhymaula@gmail.com

## 🔄 Changelog

### Version 1.0.0 (Current)
- ✅ Implementasi EDA lengkap
- ✅ K-Means clustering dengan visualisasi
- ✅ PDF report generation
- ✅ Professional government formatting
- ✅ Comprehensive filtering system

### Planned Features
- 🔄 Time series analysis
- 🔄 Predictive modeling
- 🔄 Dashboard real-time
- 🔄 API integration
- 🔄 Multi-user support

---

**Dikembangkan untuk Pemerintah Kabupaten Sumedang** 🏛️

*Smart Market Analytics - Transforming Market Data into Strategic Insights*
