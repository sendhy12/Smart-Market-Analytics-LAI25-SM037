import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from fpdf import FPDF
from io import BytesIO
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
class Config:
    """Configuration constants for the application"""
    PAGE_TITLE = "Smart Market Analytics"
    PAGE_ICON = "📊"
    LAYOUT = "wide"
    
    # Chart settings
    FIGSIZE_LARGE = (12, 8)
    FIGSIZE_MEDIUM = (10, 6)
    FIGSIZE_SMALL = (8, 5)
    
    # Colors
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'warning': '#d62728',
        'info': '#17becf'
    }
    
    # PDF settings
    PDF_MARGIN = 10
    PDF_FONT_SIZE_TITLE = 12
    PDF_FONT_SIZE_NORMAL = 11
    PDF_FONT_SIZE_SMALL = 8

# ==================== DATA PROCESSING ====================
class DataProcessor:
    """Handles all data processing operations"""
    
    @staticmethod
    @st.cache_data
    def load_and_clean_data(uploaded_file) -> pd.DataFrame:
        """Load and clean the uploaded CSV file"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Data cleaning pipeline
            df = df[df['satuan_item'] == 'kg'].copy()
            df['tanggal'] = pd.to_datetime(df['tanggal'], errors='coerce')
            df['tahun'] = df['tanggal'].dt.year
            df['bulan'] = df['tanggal'].dt.month
            df = df.dropna(subset=['jumlah', 'kebutuhan', 'tanggal'])
            
            # Convert to appropriate data types
            df['jumlah'] = pd.to_numeric(df['jumlah'], errors='coerce').astype('Int64')
            df['kebutuhan'] = pd.to_numeric(df['kebutuhan'], errors='coerce').astype('Int64')
            
            # Remove any remaining NaN values
            df = df.dropna(subset=['jumlah', 'kebutuhan'])
            
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return pd.DataFrame()
    
    @staticmethod
    def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
        """Apply filters to the dataframe"""
        filtered_df = df.copy()
        
        for column, values in filters.items():
            if values and column in df.columns:
                filtered_df = filtered_df[filtered_df[column].isin(values)]
        
        return filtered_df
    
    @staticmethod
    def get_statistics(df: pd.DataFrame) -> Dict:
        """Calculate basic statistics for the dataset"""
        return {
            'total_records': len(df),
            'avg_jumlah': df['jumlah'].mean(),
            'avg_kebutuhan': df['kebutuhan'].mean(),
            'median_jumlah': df['jumlah'].median(),
            'median_kebutuhan': df['kebutuhan'].median(),
            'correlation': df['jumlah'].corr(df['kebutuhan'])
        }

# ==================== VISUALIZATION ====================
class Visualizer:
    """Handles all visualization operations"""
    
    @staticmethod
    def create_distribution_plot(df: pd.DataFrame) -> plt.Figure:
        """Create distribution plot for jumlah and kebutuhan"""
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        
        sns.histplot(df['jumlah'], bins=50, color=Config.COLORS['primary'], 
                    label='Jumlah', kde=True, alpha=0.7, ax=ax)
        sns.histplot(df['kebutuhan'], bins=50, color=Config.COLORS['secondary'], 
                    label='Kebutuhan', kde=True, alpha=0.7, ax=ax)
        
        ax.set_xlabel("Jumlah / Kebutuhan")
        ax.set_ylabel("Frekuensi")
        ax.set_title("Distribusi Jumlah dan Kebutuhan Barang")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_scatter_plot(df: pd.DataFrame) -> plt.Figure:
        """Create scatter plot for jumlah vs kebutuhan"""
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_SMALL)
        
        sns.scatterplot(data=df, x='jumlah', y='kebutuhan', 
                       alpha=0.6, color=Config.COLORS['primary'], ax=ax)
        ax.set_title("Hubungan Jumlah vs Kebutuhan")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_top_items_plot(df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
        """Create bar plot for top items by kebutuhan"""
        top_items = (df.groupby('item_barang')[['jumlah', 'kebutuhan']]
                    .mean()
                    .sort_values(by='kebutuhan', ascending=False)
                    .head(top_n))
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        top_items.plot(kind='bar', ax=ax, 
                      color=[Config.COLORS['info'], Config.COLORS['warning']])
        ax.set_title(f"Top {top_n} Barang dengan Kebutuhan Tertinggi")
        ax.set_xlabel("Item Barang")
        ax.set_ylabel("Rata-rata")
        plt.xticks(rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_market_comparison_plot(df: pd.DataFrame, top_n: int = 5) -> plt.Figure:
        """Create horizontal bar plot for market comparison"""
        pasar_stats = (df.groupby('nama_pasar')[['jumlah', 'kebutuhan']]
                      .mean()
                      .sort_values(by='kebutuhan', ascending=False)
                      .head(top_n))
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_MEDIUM)
        pasar_stats.plot(kind='barh', ax=ax, 
                        color=[Config.COLORS['success'], Config.COLORS['warning']])
        ax.set_title(f"Top {top_n} Pasar - Rata-rata Jumlah dan Kebutuhan")
        ax.set_xlabel("Rata-rata")
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

# ==================== ANALYSIS ====================
class Analyzer:
    """Handles analysis and insights generation"""
    
    @staticmethod
    def generate_distribution_insights(stats: Dict) -> List[str]:
        """Generate insights from distribution statistics"""
        insights = []
        insights.append(f"Rata-rata stok: **{stats['avg_jumlah']:.0f}**, Median: **{stats['median_jumlah']:.0f}**")
        insights.append(f"Rata-rata kebutuhan: **{stats['avg_kebutuhan']:.0f}**, Median: **{stats['median_kebutuhan']:.0f}**")
        
        return insights
    
    @staticmethod
    def get_stock_status(avg_jumlah: float, avg_kebutuhan: float) -> Tuple[str, str]:
        """Determine stock status based on averages"""
        if avg_jumlah > avg_kebutuhan:
            return "info", "📦 Rata-rata stok lebih tinggi dari kebutuhan. Potensi overstock."
        elif avg_jumlah < avg_kebutuhan:
            return "warning", "⚠️ Rata-rata kebutuhan lebih tinggi dari stok. Potensi kekurangan barang."
        else:
            return "success", "✅ Stok dan kebutuhan seimbang."
    
    @staticmethod
    def get_correlation_insight(correlation: float) -> Tuple[str, str]:
        """Generate correlation insights"""
        if correlation > 0.7:
            return "success", f"📈 Hubungan kuat (r={correlation:.2f}): stok cenderung sesuai kebutuhan."
        elif correlation > 0.4:
            return "info", f"📉 Korelasi sedang (r={correlation:.2f})."
        else:
            return "warning", f"❗ Korelasi lemah (r={correlation:.2f}): stok tidak berbasis permintaan."

# ==================== TENSORFLOW K-MEANS CLUSTERING ====================
class TensorFlowKMeans:
    """TensorFlow implementation of K-Means clustering"""
    
    def __init__(self, n_clusters: int, max_iterations: int = 100, tolerance: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.history = []
    
    def _initialize_centroids(self, X: tf.Tensor) -> tf.Tensor:
        """Initialize centroids using random data points"""
        n_samples = tf.shape(X)[0]
        random_indices = tf.random.shuffle(tf.range(n_samples))[:self.n_clusters]
        centroids = tf.gather(X, random_indices)
        return tf.Variable(centroids, dtype=tf.float32)
    
    def _assign_clusters(self, X: tf.Tensor, centroids: tf.Variable) -> tf.Tensor:
        """Assign each point to the nearest centroid"""
        # Calculate distances from each point to each centroid
        # X shape: (n_samples, n_features)
        # centroids shape: (n_clusters, n_features)
        
        # Expand dimensions for broadcasting
        X_expanded = tf.expand_dims(X, axis=1)  # (n_samples, 1, n_features)
        centroids_expanded = tf.expand_dims(centroids, axis=0)  # (1, n_clusters, n_features)
        
        # Calculate squared distances
        distances = tf.reduce_sum(tf.square(X_expanded - centroids_expanded), axis=2)
        
        # Return cluster assignments (indices of nearest centroids)
        return tf.argmin(distances, axis=1)
    
    def _update_centroids(self, X: tf.Tensor, assignments: tf.Tensor) -> tf.Tensor:
        """Update centroids based on current assignments"""
        new_centroids = []
        
        for k in range(self.n_clusters):
            # Find points assigned to cluster k
            mask = tf.equal(assignments, k)
            cluster_points = tf.boolean_mask(X, mask)
            
            # Calculate new centroid (mean of assigned points)
            if tf.shape(cluster_points)[0] > 0:
                new_centroid = tf.reduce_mean(cluster_points, axis=0)
            else:
                # If no points assigned, keep the old centroid
                new_centroid = self.centroids[k]
            
            new_centroids.append(new_centroid)
        
        return tf.stack(new_centroids)
    
    def fit(self, X: np.ndarray) -> 'TensorFlowKMeans':
        """Fit the K-Means model to the data"""
        # Convert to TensorFlow tensor
        X_tf = tf.constant(X, dtype=tf.float32)
        
        # Initialize centroids
        self.centroids = self._initialize_centroids(X_tf)
        
        prev_centroids = None
        
        with st.spinner(f"Training TensorFlow K-Means dengan {self.n_clusters} clusters..."):
            progress_bar = st.progress(0)
            
            for iteration in range(self.max_iterations):
                # Store previous centroids for convergence check
                if prev_centroids is not None:
                    prev_centroids = tf.identity(self.centroids)
                else:
                    prev_centroids = tf.identity(self.centroids)
                
                # Assign points to clusters
                assignments = self._assign_clusters(X_tf, self.centroids)
                
                # Update centroids
                new_centroids = self._update_centroids(X_tf, assignments)
                self.centroids.assign(new_centroids)
                
                # Calculate inertia (within-cluster sum of squares)
                inertia = self._calculate_inertia(X_tf, assignments)
                self.history.append(inertia.numpy())
                
                # Check for convergence
                centroid_shift = tf.reduce_mean(tf.norm(self.centroids - prev_centroids, axis=1))
                
                # Update progress
                progress_bar.progress((iteration + 1) / self.max_iterations)
                
                if centroid_shift < self.tolerance:
                    st.success(f"✅ Konvergen pada iterasi {iteration + 1}")
                    break
            
            progress_bar.empty()
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster assignments for new data"""
        if self.centroids is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_tf = tf.constant(X, dtype=tf.float32)
        assignments = self._assign_clusters(X_tf, self.centroids)
        return assignments.numpy()
    
    def _calculate_inertia(self, X: tf.Tensor, assignments: tf.Tensor) -> tf.Tensor:
        """Calculate within-cluster sum of squares (inertia)"""
        total_inertia = 0.0
        
        for k in range(self.n_clusters):
            mask = tf.equal(assignments, k)
            cluster_points = tf.boolean_mask(X, mask)
            
            if tf.shape(cluster_points)[0] > 0:
                centroid = self.centroids[k]
                distances_squared = tf.reduce_sum(tf.square(cluster_points - centroid), axis=1)
                cluster_inertia = tf.reduce_sum(distances_squared)
                total_inertia += cluster_inertia
        
        return total_inertia
    
    def get_inertia(self, X: np.ndarray) -> float:
        """Get the final inertia of the fitted model"""
        if self.centroids is None:
            raise ValueError("Model must be fitted before calculating inertia")
        
        X_tf = tf.constant(X, dtype=tf.float32)
        assignments = self._assign_clusters(X_tf, self.centroids)
        inertia = self._calculate_inertia(X_tf, assignments)
        return inertia.numpy()
    
    def get_centroids(self) -> np.ndarray:
        """Get the final centroids"""
        if self.centroids is None:
            raise ValueError("Model must be fitted before getting centroids")
        return self.centroids.numpy()

# ==================== CLUSTERING ANALYZER WITH TENSORFLOW ====================
class ClusterAnalyzer:
    """Handles K-Means clustering operations using TensorFlow"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.scaler = StandardScaler()
        self.tf_kmeans = None
        self.X_scaled = None
        self.feature_names = ['jumlah', 'kebutuhan']
    
    def prepare_data(self) -> bool:
        """Prepare data for clustering"""
        try:
            X = self.df[self.feature_names].values
            self.X_scaled = self.scaler.fit_transform(X)
            return True
        except Exception as e:
            st.error(f"Error preparing data for clustering: {str(e)}")
            return False
    
    def perform_clustering(self, n_clusters: int) -> Tuple[np.ndarray, Dict]:
        """Perform K-Means clustering using TensorFlow"""
        # Initialize TensorFlow K-Means
        self.tf_kmeans = TensorFlowKMeans(n_clusters=n_clusters, max_iterations=100)
        
        # Fit the model
        self.tf_kmeans.fit(self.X_scaled)
        
        # Get cluster assignments
        clusters = self.tf_kmeans.predict(self.X_scaled)
        
        # Get clustering metrics
        metrics = {
            'inertia': self.tf_kmeans.get_inertia(self.X_scaled),
            'centroids': self.tf_kmeans.get_centroids(),
            'convergence_history': self.tf_kmeans.history
        }
        
        return clusters, metrics
    
    def generate_cluster_insights(self, df_with_clusters: pd.DataFrame) -> List[Dict]:
        """Generate insights for each cluster"""
        insights = []
        
        for cluster_id in sorted(df_with_clusters['cluster'].unique()):
            subset = df_with_clusters[df_with_clusters['cluster'] == cluster_id]
            
            insight = {
                'cluster_id': cluster_id,
                'avg_jumlah': int(subset['jumlah'].mean()),
                'avg_kebutuhan': int(subset['kebutuhan'].mean()),
                'total_items': len(subset),
                'top_items': subset['item_barang'].value_counts().head(3).index.tolist(),
                'interpretation': 'tinggi' if subset['kebutuhan'].mean() > subset['kebutuhan'].median() else 'rendah'
            }
            insights.append(insight)
        
        return insights
    
    def plot_convergence(self) -> plt.Figure:
        """Plot the convergence history of the TensorFlow K-Means algorithm"""
        if self.tf_kmeans is None or not self.tf_kmeans.history:
            return None
        
        fig, ax = plt.subplots(figsize=Config.FIGSIZE_SMALL)
        ax.plot(self.tf_kmeans.history, marker='o', linestyle='-', color=Config.COLORS['primary'])
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Inertia (Within-cluster sum of squares)')
        ax.set_title('TensorFlow K-Means Convergence')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def display_centroids_info(self):
        """Display information about cluster centroids"""
        if self.tf_kmeans is None:
            return
        
        centroids = self.tf_kmeans.get_centroids()
        
        st.subheader("🎯 Informasi Centroid Cluster")
        
        # Create a DataFrame for centroids (in original scale)
        centroids_original = self.scaler.inverse_transform(centroids)
        centroids_df = pd.DataFrame(
            centroids_original, 
            columns=self.feature_names,
            index=[f'Cluster {i}' for i in range(len(centroids_original))]
        )
        
        # Display as a nice table
        st.dataframe(
            centroids_df.round(2),
            use_container_width=True
        )
        
        # Display additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Inertia", f"{self.tf_kmeans.get_inertia(self.X_scaled):.2f}")
        with col2:
            st.metric("Iterations to Converge", len(self.tf_kmeans.history))

# ==================== PDF GENERATOR ====================
class PDFGenerator:
    """Handles PDF report generation with professional formatting"""
    
    class ReportPDF(FPDF):
        def __init__(self):
            super().__init__()
            self.set_auto_page_break(auto=True, margin=20)
        
        def header(self):
            if self.page_no() > 0:
                # Background color for header
                self.set_fill_color(240, 248, 255)
                self.rect(10, 10, 190, 45, 'F')
                
                # Logo placeholder
                self.set_fill_color(70, 130, 180)
                self.rect(15, 15, 20, 20, 'F')
                self.set_font('Arial', 'B', 8)
                self.set_text_color(255, 255, 255)
                self.text(20, 27, 'LOGO')
                
                # Reset text color
                self.set_text_color(0, 0, 0)
                
                # Government header
                self.set_font('Arial', 'B', 14)
                self.set_xy(40, 15)
                self.cell(0, 6, "PEMERINTAH KABUPATEN SUMEDANG", ln=True, align='L')
                
                self.set_font('Arial', 'B', 12)
                self.set_x(40)
                self.cell(0, 5, "DINAS KOPERASI, UKM, PERDAGANGAN DAN PERINDUSTRIAN", ln=True, align='L')
                
                # Contact information
                self.set_font('Arial', '', 9)
                self.set_x(40)
                self.cell(0, 4, "Jl. Raya Sumedang No.123, Sumedang 45300, Jawa Barat", ln=True, align='L')
                self.set_x(40)
                self.cell(0, 4, "Telp: (0261) 123456 | Fax: (0261) 123457", ln=True, align='L')
                self.set_x(40)
                self.cell(0, 4, "Email: dinaskopdag@sumedang.go.id | Website: www.sumedang.go.id", ln=True, align='L')
                
                # Decorative line
                self.set_line_width(0.5)
                self.set_draw_color(70, 130, 180)
                self.line(10, 58, 200, 58)
                self.set_line_width(0.2)
                self.line(10, 60, 200, 60)
                
                self.ln(20)
        
        def footer(self):
            # Background for footer
            self.set_fill_color(240, 248, 255)
            self.rect(10, -25, 190, 15, 'F')
            
            self.set_y(-20)
            self.set_font('Arial', 'I', 8)
            self.set_text_color(100, 100, 100)
            
            # Left side - document info
            self.set_x(15)
            self.cell(0, 5, f"Dokumen dibuat secara otomatis oleh Smart Market Analytics (TensorFlow K-Means)", ln=True, align='L')
            
            # Right side - page and date
            self.set_y(-15)
            self.set_x(15)
            self.cell(0, 5, f"Halaman {self.page_no()} | Dicetak pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", 
                     ln=True, align='R')
            
            self.set_text_color(0, 0, 0)
        
        def add_title_page(self, report_title: str, filters: Dict):
            """Add a professional title page"""
            self.add_page()
            
            # Document title
            self.ln(10)
            self.set_font('Arial', 'B', 18)
            self.set_text_color(70, 130, 180)
            self.cell(0, 12, report_title.upper(), ln=True, align='C')
            
            # Subtitle with TensorFlow mention
            self.set_font('Arial', '', 12)
            self.set_text_color(100, 100, 100)
            self.cell(0, 8, "Analisis Data Pasar Tradisional Kabupaten Sumedang", ln=True, align='C')
            self.cell(0, 6, "Menggunakan TensorFlow K-Means Clustering", ln=True, align='C')
            
            self.ln(15)
            
            # Information box
            self.set_fill_color(248, 249, 250)
            self.set_draw_color(200, 200, 200)
            self.rect(20, self.get_y(), 170, 60, 'DF')
            
            self.set_text_color(0, 0, 0)
            self.ln(5)
            
            # Filter information
            self.set_font('Arial', 'B', 11)
            self.cell(0, 8, "PARAMETER ANALISIS", ln=True, align='C')
            self.ln(3)
            
            self.set_font('Arial', '', 10)
            
            filter_labels = {
                'nama_pasar': 'Pasar yang Dianalisis',
                'tahun': 'Periode Tahun',
                'bulan': 'Periode Bulan',
                'item_barang': 'Jenis Barang'
            }
            
            for key, value in filters.items():
                if value and key in filter_labels:
                    self.set_x(25)
                    self.set_font('Arial', 'B', 10)
                    self.cell(50, 6, f"{filter_labels[key]}:", ln=False, align='L')
                    
                    self.set_font('Arial', '', 10)
                    value_str = ', '.join(map(str, value))
                    if len(value_str) > 80:
                        value_str = value_str[:77] + "..."
                    self.cell(0, 6, value_str, ln=True, align='L')
            
            # Generation info
            self.ln(10)
            self.set_font('Arial', 'I', 9)
            self.set_text_color(120, 120, 120)
            self.cell(0, 6, f"Laporan dibuat pada: {datetime.now().strftime('%d %B %Y, %H:%M WIB')}", 
                     ln=True, align='C')
            self.cell(0, 6, f"Algoritma: TensorFlow K-Means Clustering", ln=True, align='C')
            
            self.add_page()
        
        def add_tensorflow_section(self, metrics: Dict):
            """Add TensorFlow-specific metrics section"""
            self.ln(5)
            
            # Background for section header
            self.set_fill_color(255, 140, 0)  # Orange for TensorFlow
            self.rect(10, self.get_y(), 190, 12, 'F')
            
            # Section title
            self.set_text_color(255, 255, 255)
            self.set_font('Arial', 'B', 12)
            self.cell(0, 12, "  TENSORFLOW K-MEANS METRICS", ln=True, align='L')
            
            self.set_text_color(0, 0, 0)
            self.ln(3)
            
            # Metrics
            self.set_font('Arial', '', 10)
            self.cell(0, 6, f"Total Inertia (Within-cluster sum of squares): {metrics.get('inertia', 0):.2f}", ln=True)
            self.cell(0, 6, f"Iterations to Convergence: {len(metrics.get('convergence_history', []))}", ln=True)
            self.cell(0, 6, f"Number of Centroids: {len(metrics.get('centroids', []))}", ln=True)
            
            self.ln(5)
    
    @staticmethod
    def generate_clustering_report(filters: Dict, insights: List[Dict], stats: Dict = None, tf_metrics: Dict = None) -> BytesIO:
        """Generate comprehensive PDF report for TensorFlow clustering analysis"""
        pdf = PDFGenerator.ReportPDF()
        
        # Title page
        pdf.add_title_page("Laporan Analisis TensorFlow K-Means Clustering", filters)
        
        # TensorFlow metrics section
        if tf_metrics:
            pdf.add_tensorflow_section(tf_metrics)
        
        # Executive Summary
        if stats:
            pdf.ln(5)
            pdf.set_fill_color(70, 130, 180)
            pdf.rect(10, pdf.get_y(), 190, 12, 'F')
            
            pdf.set_text_color(255, 255, 255)
            pdf.set_font('Arial', 'B', 12)
            pdf.cell(0, 12, "  RINGKASAN EKSEKUTIF", ln=True, align='L')
            
            pdf.set_text_color(0, 0, 0)
            pdf.ln(3)
            
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f"Total data yang dianalisis: {stats.get('total_records', 0):,} record", ln=True)
            pdf.cell(0, 6, f"Korelasi jumlah vs kebutuhan: {stats.get('correlation', 0):.3f}", ln=True)
            pdf.cell(0, 6, f"Rata-rata stok keseluruhan: {stats.get('avg_jumlah', 0):.0f} unit", ln=True)
            pdf.cell(0, 6, f"Rata-rata kebutuhan keseluruhan: {stats.get('avg_kebutuhan', 0):.0f} unit", ln=True)
            pdf.ln(5)
        
        # Save to BytesIO
        pdf_output = BytesIO()
        pdf_bytes = pdf.output(dest='S').encode('latin-1')
        pdf_output.write(pdf_bytes)
        pdf_output.seek(0)
        
        return pdf_output

# ==================== UI COMPONENTS ====================
class UIComponents:
    """Reusable UI components"""
    
    @staticmethod
    def create_filters_sidebar(df: pd.DataFrame) -> Dict:
        """Create filter sidebar and return filter values"""
        st.sidebar.header("🔍 Filter Data")
        
        filters = {}
        
        if 'nama_pasar' in df.columns:
            filters['nama_pasar'] = st.sidebar.multiselect(
                "Pilih Pasar:", 
                options=sorted(df['nama_pasar'].unique()),
                default=sorted(df['nama_pasar'].unique())
            )
        
        if 'item_barang' in df.columns:
            filters['item_barang'] = st.sidebar.multiselect(
                "Pilih Barang:", 
                options=sorted(df['item_barang'].unique()),
                default=sorted(df['item_barang'].unique())[:10]  # Limit default selection
            )
        
        if 'tahun' in df.columns:
            filters['tahun'] = st.sidebar.multiselect(
                "Pilih Tahun:", 
                options=sorted(df['tahun'].unique()),
                default=sorted(df['tahun'].unique())
            )
        
        if 'bulan' in df.columns:
            filters['bulan'] = st.sidebar.multiselect(
                "Pilih Bulan:", 
                options=list(range(1, 13)),
                default=sorted(df['bulan'].unique())
            )
        
        return filters
    
    @staticmethod
    def display_metrics(stats: Dict):
        """Display key metrics in columns"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{stats['total_records']:,}")
        with col2:
            st.metric("Avg Jumlah", f"{stats['avg_jumlah']:.0f}")
        with col3:
            st.metric("Avg Kebutuhan", f"{stats['avg_kebutuhan']:.0f}")
        with col4:
            st.metric("Correlation", f"{stats['correlation']:.2f}")

# ==================== MAIN APPLICATION ====================
class MarketAnalysisApp:
    """Main application class"""
    
    def __init__(self):
        self.setup_page()
        self.data_processor = DataProcessor()
        self.visualizer = Visualizer()
        self.analyzer = Analyzer()
        self.ui = UIComponents()
    
    def setup_page(self):
        """Setup Streamlit page configuration"""
        st.set_page_config(
            page_title=Config.PAGE_TITLE,
            page_icon=Config.PAGE_ICON,
            layout=Config.LAYOUT,
            initial_sidebar_state="expanded"
        )
    
    def run(self):
        """Run the main application"""
        st.title(f"{Config.PAGE_ICON} {Config.PAGE_TITLE}")
        
        # Navigation
        page = st.sidebar.radio(
            "📍 Navigasi", 
            ["📊 Exploratory Data Analysis", "🎯 K-Means Clustering"],
            index=0
        )
        
        # File upload
        uploaded_file = st.file_uploader(
            "📁 Unggah file CSV Anda", 
            type=["csv"],
            help="Pastikan file CSV berisi kolom: nama_pasar, item_barang, jumlah, kebutuhan, tanggal, satuan_item"
        )
        
        if uploaded_file is not None:
            df = self.data_processor.load_and_clean_data(uploaded_file)
            
            if df.empty:
                st.error("❌ Data tidak dapat dimuat atau tidak valid.")
                return
            
            st.success(f"✅ Data berhasil dimuat: {len(df):,} records")
            
            if page == "📊 Exploratory Data Analysis":
                self.render_eda_page(df)
            else:
                self.render_clustering_page(df)
        else:
            st.info("👆 Silakan unggah file CSV untuk memulai analisis.")
    
    def render_eda_page(self, df: pd.DataFrame):
        """Render EDA page"""
        st.header("📊 Exploratory Data Analysis")
        
        # Filters
        filters = self.ui.create_filters_sidebar(df)
        filtered_df = self.data_processor.apply_filters(df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter yang dipilih.")
            return
        
        # Statistics
        stats = self.data_processor.get_statistics(filtered_df)
        self.ui.display_metrics(stats)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Distribusi Data")
            fig1 = self.visualizer.create_distribution_plot(filtered_df)
            st.pyplot(fig1)
            
            # Insights
            with st.expander("🔍 Insight Distribusi"):
                insights = self.analyzer.generate_distribution_insights(stats)
                for insight in insights:
                    st.markdown(f"• {insight}")
                
                status_type, status_msg = self.analyzer.get_stock_status(
                    stats['avg_jumlah'], stats['avg_kebutuhan']
                )
                getattr(st, status_type)(status_msg)
        
        with col2:
            st.subheader("🔄 Korelasi Jumlah vs Kebutuhan")
            fig2 = self.visualizer.create_scatter_plot(filtered_df)
            st.pyplot(fig2)
            
            # Correlation insight
            with st.expander("🔍 Insight Korelasi"):
                corr_type, corr_msg = self.analyzer.get_correlation_insight(stats['correlation'])
                getattr(st, corr_type)(corr_msg)
        
        # Additional charts
        st.subheader("🏆 Top Barang dengan Kebutuhan Tertinggi")
        fig3 = self.visualizer.create_top_items_plot(filtered_df)
        st.pyplot(fig3)
        
        st.subheader("🏪 Perbandingan Pasar")
        fig4 = self.visualizer.create_market_comparison_plot(filtered_df)
        st.pyplot(fig4)
    
    def render_clustering_page(self, df: pd.DataFrame):
        """Render clustering page"""
        st.header("🎯 K-Means Clustering Analysis")
        
        # Filters (simplified for clustering)
        filters = {
            'nama_pasar': st.sidebar.multiselect(
                "Pilih Pasar:", 
                options=sorted(df['nama_pasar'].unique()),
                default=sorted(df['nama_pasar'].unique())
            ),
            'tahun': st.sidebar.multiselect(
                "Pilih Tahun:", 
                options=sorted(df['tahun'].unique()),
                default=sorted(df['tahun'].unique())
            ),
            'bulan': st.sidebar.multiselect(
                "Pilih Bulan:", 
                options=sorted(df['bulan'].unique()),
                default=sorted(df['bulan'].unique())
            )
        }
        
        filtered_df = self.data_processor.apply_filters(df, filters)
        
        if filtered_df.empty:
            st.warning("⚠️ Tidak ada data yang cocok dengan filter yang dipilih.")
            return
        
        # Clustering parameters
        k = st.slider("🎛️ Pilih jumlah cluster (K):", min_value=2, max_value=10, value=3)
        
        # Perform clustering
        cluster_analyzer = ClusterAnalyzer(filtered_df)
        if not cluster_analyzer.prepare_data():
            return
        
        clusters = cluster_analyzer.perform_clustering(k)
        filtered_df_with_clusters = filtered_df.copy()
        filtered_df_with_clusters['cluster'] = clusters
        
        # Visualization
        st.subheader("📊 Visualisasi Cluster")
        fig = px.scatter(
            filtered_df_with_clusters, 
            x='jumlah', 
            y='kebutuhan', 
            color=filtered_df_with_clusters['cluster'].astype(str),
            title='Hasil Clustering: Jumlah vs Kebutuhan',
            labels={
                'jumlah': 'Jumlah Barang', 
                'kebutuhan': 'Kebutuhan Barang', 
                'color': 'Cluster'
            },
            hover_data=['item_barang', 'nama_pasar']
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        insights = cluster_analyzer.generate_cluster_insights(filtered_df_with_clusters)
        
        st.subheader("🔍 Insight Clustering")
        for insight in insights:
            with st.expander(f"Cluster {insight['cluster_id']} ({insight['total_items']} items)"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Rata-rata Jumlah", f"{insight['avg_jumlah']:,}")
                    st.metric("Rata-rata Kebutuhan", f"{insight['avg_kebutuhan']:,}")
                
                with col2:
                    st.write("**Top 3 Barang:**")
                    for i, item in enumerate(insight['top_items'], 1):
                        st.write(f"{i}. {item}")
                
                interpretation_color = "green" if insight['interpretation'] == 'tinggi' else "blue"
                st.markdown(f"**Interpretasi:** <span style='color:{interpretation_color}'>Pola kebutuhan {insight['interpretation']}</span>", 
                           unsafe_allow_html=True)
        
        # PDF Export
        st.subheader("📄 Export Laporan")
        if st.button("📥 Generate PDF Report", type="primary"):
            with st.spinner("Membuat laporan PDF profesional..."):
                stats = self.data_processor.get_statistics(filtered_df_with_clusters)
                pdf_output = PDFGenerator.generate_clustering_report(filters, insights, stats)
                
                st.success("✅ Laporan PDF berhasil dibuat!")
                st.download_button(
                    label="📄 Download Laporan PDF",
                    data=pdf_output,
                    file_name=f"laporan_clustering_pasar_sumedang_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    help="Klik untuk mengunduh laporan dalam format PDF"
                )

# ==================== RUN APPLICATION ====================
if __name__ == "__main__":
    app = MarketAnalysisApp()
    app.run()