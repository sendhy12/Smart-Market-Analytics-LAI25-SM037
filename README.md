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
| `id` | Integer | ID unik untuk setiap record |
| `satuan` | String | Satuan pengukuran (kosong/optional) |
| `pasar` | Integer | Kode numerik pasar |
| `tanggal` | Date | Tanggal pencatatan (YYYY-MM-DD) |
| `nama_item` | Integer | Kode item barang |
| `keterangan` | String | Status ketersediaan (cukup/kurang/berlebih) |
| `harga` | Integer | Harga barang per satuan |
| `jumlah` | Integer | Jumlah stok barang tersedia |
| `kebutuhan` | Integer | Kebutuhan barang yang dibutuhkan |
| `item_barang` | String | Nama lengkap barang |
| `satuan_item` | String | Satuan item (kg/liter/dll) |
| `nama_pasar` | String | Nama lengkap pasar tradisional |

### Contoh Data
```csv
id,satuan,pasar,tanggal,nama_item,keterangan,harga,jumlah,kebutuhan,item_barang,satuan_item,nama_pasar
26766,,7,2022-01-01,1,cukup,12000,150,200,Beras Premium,kg,Pasar Parakanmuncang
26767,,7,2022-01-01,2,cukup,11500,100,120,Beras Medium,kg,Pasar Parakanmuncang
26768,,7,2022-01-01,3,kurang,12000,50,100,Beras Termahal,kg,Pasar Parakanmuncang
26769,,7,2022-01-01,4,cukup,14000,80,75,Gula Pasir,kg,Pasar Parakanmuncang
26770,,7,2022-01-01,8,berlebih,15000,200,150,Minyak Goreng Bimoli,liter,Pasar Parakanmuncang
```

### ⚠️ Catatan Penting tentang Data
- **ID**: Setiap record harus memiliki ID unik
- **Tanggal**: Format YYYY-MM-DD wajib diikuti
- **Harga**: Dalam format integer (tanpa desimal)
- **Jumlah & Kebutuhan**: Nilai 0 menunjukkan tidak ada data/tidak tersedia
- **Keterangan**: Indikator status stok (cukup/kurang/berlebih)
- **Satuan**: Kolom ini bisa kosong, satuan sebenarnya ada di `satuan_item`

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
- **Handling Missing Data**: Aplikasi akan otomatis menangani nilai 0 atau data kosong

### 2. Exploratory Data Analysis
- Gunakan sidebar untuk memfilter data berdasarkan:
  - Pasar yang ingin dianalisis (`nama_pasar`)
  - Jenis barang (`item_barang`)
  - Periode tahun (dari kolom `tanggal`)
  - Periode bulan (dari kolom `tanggal`)
  - Status keterangan (`cukup`, `kurang`, `berlebih`)
- Lihat visualisasi dan insight yang dihasilkan secara otomatis

### 3. K-Means Clustering
- Pilih menu "🎯 K-Means Clustering"
- Atur parameter clustering menggunakan slider
- Analisis hasil clustering berdasarkan:
  - **Jumlah stok** vs **Kebutuhan**
  - **Harga** vs **Status ketersediaan**
- Download laporan PDF jika diperlukan

### 4. Filter Data
Aplikasi menyediakan filter komprehensif:
- **Multi-select filter** untuk pasar, barang, tahun, dan bulan
- **Status filter** berdasarkan keterangan stok
- **Default selection** yang cerdas untuk performa optimal
- **Real-time filtering** yang langsung memperbarui visualisasi

## 📈 Interpretasi Hasil

### EDA Insights
- **Status Stok**: Berdasarkan kolom `keterangan` dan perbandingan `jumlah` vs `kebutuhan`
- **Analisis Harga**: Korelasi antara harga dengan ketersediaan stok
- **Top Items**: Identifikasi barang dengan kebutuhan tertinggi dari kolom `kebutuhan`
- **Market Performance**: Perbandingan performa antar `nama_pasar`
- **Trend Analysis**: Pola musiman berdasarkan `tanggal`

### Clustering Insights
- **Cluster Harga Tinggi**: Barang premium dengan harga di atas rata-rata
- **Cluster Kebutuhan Tinggi**: Barang dengan permintaan tinggi
- **Cluster Stok Berlebih**: Identifikasi overstock untuk optimasi
- **Pattern Recognition**: Identifikasi pola tersembunyi dalam hubungan harga-stok-kebutuhan

### Status Keterangan
- **"cukup"**: Stok memadai sesuai kebutuhan
- **"kurang"**: Stok di bawah kebutuhan (perlu restocking)
- **"berlebih"**: Stok melebihi kebutuhan (risk of waste)

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
- **Privacy First**: ID dan informasi sensitif tidak disimpan atau dibagikan

## 🚨 Troubleshooting

### Error Loading Data
- Pastikan file CSV memiliki semua kolom yang diperlukan (12 kolom)
- Periksa format tanggal (YYYY-MM-DD)
- Pastikan kolom `id`, `pasar`, `nama_item`, `harga`, `jumlah`, `kebutuhan` berisi angka
- Kolom `satuan` boleh kosong, tapi kolom lain harus terisi

### Performance Issues dengan Data Besar
- **Zero Values**: Data dengan nilai 0 di `jumlah` dan `kebutuhan` akan difilter otomatis
- **Date Range**: Gunakan filter tanggal untuk membatasi data yang dianalisis
- **Market Selection**: Pilih pasar spesifik untuk analisis mendalam
- **Memory Optimization**: Tutup tab browser yang tidak diperlukan

### PDF Generation Failed
- Pastikan memori sistem mencukupi
- Reduce jumlah cluster jika data terlalu besar
- Periksa apakah data telah difilter dengan benar
- Pastikan tidak ada karakter khusus dalam nama pasar/barang

## 🤝 Kontribusi

Untuk berkontribusi pada proyek ini:
1. Fork repository
2. Buat branch fitur baru (`git checkout -b feature/AmazingFeature`)
3. Commit perubahan (`git commit -m 'Add some AmazingFeature'`)
4. Push ke branch (`git push origin feature/AmazingFeature`)
5. Buat Pull Request

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
- ✅ Implementasi EDA lengkap dengan 12 kolom data
- ✅ K-Means clustering dengan visualisasi harga-stok-kebutuhan
- ✅ PDF report generation dengan format pemerintahan
- ✅ Handling untuk data dengan nilai 0 (missing data)
- ✅ Comprehensive filtering system termasuk status keterangan
- ✅ Price analysis dan correlation insights

### Planned Features
- 🔄 Time series analysis untuk trend bulanan/tahunan
- 🔄 Predictive modeling untuk forecasting kebutuhan
- 🔄 Dashboard real-time dengan auto-refresh
- 🔄 API integration untuk data real-time
- 🔄 Multi-user support dengan role management
- 🔄 Advanced price analytics dan market comparison

---

**Dikembangkan untuk Pemerintah Kabupaten Sumedang** 🏛️

*Smart Market Analytics - Transforming Market Data into Strategic Insights*
