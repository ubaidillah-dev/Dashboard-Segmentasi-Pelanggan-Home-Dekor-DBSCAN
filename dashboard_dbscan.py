import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.neighbors import NearestNeighbors

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Segmentasi Pelanggan Home Decor (DBSCAN)",
    layout="wide",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# --- FITUR LOGIN KEAMANAN ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="background-color: #FFFFFF; padding: 2rem; border-radius: 10px; border: 1px solid #E2E8F0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: #1A5F7A; margin-bottom: 0;">🔒 Akses Dasbor Terkunci</h2>
            <p style="color: #4A5568; margin-bottom: 1.5rem;">Silakan masukkan password untuk masuk ke dashboard analisa.</p>
        </div>
        """, unsafe_allow_html=True)
        
        pwd = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Masukkan Password...")
        
        if st.button("Buka Dasbor", use_container_width=True):
            if pwd == "naka123":  # Ganti "naka123" dengan password rahasia klien
                st.session_state["logged_in"] = True
                st.rerun()
            else:
                st.error("❌ Password salah! Akses ditolak.")
    
    st.stop()

# --- 2. STYLING CSS ---
st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar Theme: Dark Slate to Navy */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2D3748 0%, #151A22 100%);
    }
    [data-testid="stSidebar"] * {
        color: #F5F0E8 !important;
        font-size: 1rem;
    }
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #C9A96E !important;
        font-size: 1.4rem;
    }
    [data-testid="stSidebar"] .stCaption {
        font-size: 0.95rem;
        color: #A0AEC0 !important;
    }
    [data-testid="stSidebar"] label {
        font-weight: 600 !important;
    }
    
    /* Mengunci lebar background menu radio di sidebar agar seragam 100% dan rapi */
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label {
        background: rgba(255,255,255,0.15);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin-bottom: 0.25rem;
        transition: all 0.2s;
        font-weight: 600;
        width: 100% !important;
        box-sizing: border-box;
    }
    [data-testid="stSidebar"] .stRadio > div[role="radiogroup"] > label:hover {
        background: rgba(255,255,255,0.3);
    }
    
    /* Slider */
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background: #E0D6CC !important;
    }
    .stSlider [data-baseweb="slider"] div[data-testid="stThumbValue"] ~ div {
        background: #C9A96E !important;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"] {
        background-color: #C9A96E !important; 
        border: 3px solid #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3) !important;
        border-radius: 50% !important;
        transition: all 0.2s ease;
    }
    .stSlider [data-baseweb="slider"] div[role="slider"]:hover {
        background-color: #C9A96E !important;
        border-color: #F5F0E8 !important;
        transform: scale(1.1);
    }
    
    /* Multiselect - Ukuran Tag Diperkecil agar Rapi */
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border: 1px solid #C9A96E !important;
        border-radius: 8px !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="select"] input {
        color: #F5F0E8 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
        background-color: #C9A96E !important;
        border-radius: 4px !important;
        border: none !important;
        padding: 0px 4px !important;
        margin: 2px !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
        color: #151A22 !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="icon"] {
        fill: #F5F0E8 !important;
    }
    [data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] [data-baseweb="icon"] {
        fill: #151A22 !important;
    }

    /* Styling untuk st.expander di Sidebar */
    [data-testid="stSidebar"] [data-testid="stExpander"] {
        background: rgba(255,255,255,0.05);
        border: 1px solid #C9A96E;
        border-radius: 8px;
    }
    [data-testid="stSidebar"] [data-testid="stExpander"] summary {
        color: #C9A96E !important;
        font-weight: 700 !important;
    }
    
    /* Metrik di Sidebar */
    [data-testid="stSidebar"] [data-testid="stMetric"] {
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid #C9A96E !important;
        border-radius: 12px;
        padding: 0.8rem;
        margin: 0.5rem 0;
        box-shadow: none !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] label {
        color: #A0AEC0 !important;
        font-weight: 600;
        font-size: 0.9rem;
    }
    [data-testid="stSidebar"] [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 700;
        font-size: 1.4rem;
    }

    /* --- PERBAIKAN: Tema Metrik Utama (Main Page) Warna Angka jadi HITAM PEKAT --- */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    [data-testid="stMetric"] label {
        color: #4A5568 !important;
        font-weight: 700;
        font-size: 1rem;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #000000 !important; /* Diubah jadi Hitam */
        font-weight: 700;
        font-size: 2rem;
    }
    
    /* --- PERBAIKAN: Judul Utama dan Sub-Judul diubah jadi HITAM PEKAT --- */
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #000000; /* Diubah jadi Hitam */
        border-bottom: 4px solid #1A5F7A; /* Garis bawah tetap biru sebagai aksen */
        padding-bottom: 0.6rem;
        margin-bottom: 1.2rem;
    }
    .sub-header {
        font-size: 1.6rem;
        font-weight: 600;
        color: #000000; /* Diubah jadi Hitam */
        margin-top: 1.5rem;
    }
    
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. PERSIAPAN DATA ---
REFERENCE_DATE = datetime(2026, 6, 17)

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('data_interior_lengkap.csv')
        df['Frequency'] = df['Frequency'].clip(lower=1)
    except FileNotFoundError:
        st.error("Data transaksi tidak ditemukan! Harap pastikan file data_interior_lengkap.csv tersedia.")
        st.stop()

    if 'Tanggal Terakhir Belanja' in df.columns:
        df['Tanggal Terakhir Belanja'] = pd.to_datetime(df['Tanggal Terakhir Belanja'])
        df['Recency'] = (REFERENCE_DATE - df['Tanggal Terakhir Belanja']).dt.days
        df['Tanggal Tampilan'] = df['Tanggal Terakhir Belanja'].dt.strftime('%Y-%m-%d')
    else:
        if 'Recency' not in df.columns:
            st.error("Kolom 'Recency' atau 'Tanggal Terakhir Belanja' tidak ditemukan di CSV.")
            st.stop()
        df['Tanggal Terakhir Belanja'] = REFERENCE_DATE - pd.to_timedelta(df['Recency'], unit='d')
        df['Tanggal Tampilan'] = df['Tanggal Terakhir Belanja'].dt.strftime('%Y-%m-%d')

    kolom_demografi = {
        'Pekerjaan': ['Karyawan Swasta', 'Wiraswasta', 'PNS', 'Ibu Rumah Tangga', 'Profesional'],
        'Jenis Kelamin': ['Laki-laki', 'Perempuan'],
        'Pendidikan': ['SMA', 'Diploma', 'S1', 'S2'],
        'Kategori Produk': ['Klasik', 'Modern'],
        'Tingkat Harga': ['Ekonomis', 'Menengah', 'Premium'],
        'Ruangan': ['Ruang Tamu', 'Dapur', 'Kamar Tidur', 'Ruang Kantor'],
        'Item Spesifik': ['Kursi / Sofa', 'Lemari / Kabinet', 'Meja', 'Dekorasi / Aksesoris', 'Full Set']
    }
    
    np.random.seed(42)
    for kol, pilihan in kolom_demografi.items():
        if kol not in df.columns:
            df[kol] = np.random.choice(pilihan, size=len(df))

    df.loc[df['Pekerjaan'] == 'Ibu Rumah Tangga', 'Jenis Kelamin'] = 'Perempuan'
    return df

df = load_data()
kolom_rfm = ['Usia', 'Recency', 'Frequency', 'Monetary']
data_rfm = df[kolom_rfm].copy()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(data_rfm)

# Tema Warna Baru yang Kontras
tema_warna = ['#1A5F7A', '#C9A96E', '#D05B43', '#5C7C66', '#6D4C69', '#2D3748']

# --- 4. SIDEBAR ---
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1.2rem;">
        <span style="font-size: 2.0rem; font-weight: 700; color: #C9A96E; letter-spacing: 2px;">DASHBOARD</span><br>
        <span style="font-size: 1.1rem; color: #E2E8F0;">Home Decor</span>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Segmentasi Pelanggan Cerdas (DBSCAN)")
    st.divider()

    menu = st.radio("Menu Analisis:", [
        "Ringkasan Bisnis", "Pengelompokan Pelanggan", "Kebiasaan Belanja",
        "Demografi Pelanggan", "Minat Produk & Ruangan", "Data Pelanggan Lengkap"
    ])
    st.divider()

    st.markdown("**Pengaturan Pengelompokan**")
    st.caption("Geser untuk menyesuaikan seberapa mirip pelanggan dalam satu kelompok.")
    
    st.markdown("""
    <div style="background-color: rgba(255, 255, 255, 0.08); border-left: 4px solid #C9A96E; padding: 0.8rem 1rem; border-radius: 6px; margin-bottom: 0.8rem;">
        <p style="color: #C9A96E; font-weight: 700; margin: 0 0 0.2rem 0; font-size: 0.9rem;">PARAMETER REKOMENDASI</p>
        <p style="color: #E2E8F0; margin: 0; font-size: 0.85rem;">
            Berdasarkan hasil optimasi, gunakan <b>eps = 0.7</b> dan <b>min_samples = 8</b> untuk hasil pengelompokan terbaik.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    eps = st.slider("eps (Toleransi Kemiripan)", 0.1, 2.0, 0.7, 0.05,
                    help="eps (epsilon): Jarak maksimum antar pelanggan agar dianggap mirip. Semakin kecil -> kelompok semakin ketat.")
    min_samples = st.slider("min_samples (Minimal Tetangga)", 2, 20, 8, 1,
                             help="min_samples: Jumlah minimal pelanggan yang harus mirip untuk membentuk satu kelompok.")
    st.divider()

    with st.expander("🛠️ Filter Tampilan Data", expanded=False):
        st.caption("Saring data tanpa mengubah bentuk kelompok utama.")
        pilihan_pekerjaan = st.multiselect("Pekerjaan", options=df['Pekerjaan'].unique(), default=df['Pekerjaan'].unique())
        pilihan_kelamin = st.multiselect("Jenis Kelamin", options=df['Jenis Kelamin'].unique(), default=df['Jenis Kelamin'].unique())
        pilihan_pendidikan = st.multiselect("Pendidikan", options=df['Pendidikan'].unique(), default=df['Pendidikan'].unique())
        pilihan_kategori = st.multiselect("Kategori Produk", options=df['Kategori Produk'].unique(), default=df['Kategori Produk'].unique())
        pilihan_harga = st.multiselect("Tingkat Harga", options=df['Tingkat Harga'].unique(), default=df['Tingkat Harga'].unique())
        pilihan_ruangan = st.multiselect("Ruangan", options=df['Ruangan'].unique(), default=df['Ruangan'].unique())
        pilihan_item = st.multiselect("Item Spesifik", options=df['Item Spesifik'].unique(), default=df['Item Spesifik'].unique())
    st.divider()

    # Model DBSCAN
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(df_scaled)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    def dynamic_cluster_name(cluster_label, labels, df_rfm):
        if cluster_label == -1:
            return '<i class="fa-solid fa-star"></i> Pembeli Spesial (VIP)'
        mask = labels == cluster_label
        cluster_data = df_rfm[mask]
        overall_mean = df_rfm.mean()
        recency_mean = cluster_data['Recency'].mean()
        frequency_mean = cluster_data['Frequency'].mean()
        monetary_mean = cluster_data['Monetary'].mean()
        is_low_recency = recency_mean < overall_mean['Recency']
        is_high_freq = frequency_mean > overall_mean['Frequency']
        is_high_mon = monetary_mean > overall_mean['Monetary']

        if is_high_mon and is_high_freq: return '<i class="fa-solid fa-crown"></i> Langganan Loyal'
        elif is_high_mon and not is_high_freq: return '<i class="fa-solid fa-gem"></i> Pembeli Borongan'
        elif not is_high_mon and is_high_freq: return '<i class="fa-solid fa-cart-shopping"></i> Reguler'
        elif is_low_recency and not is_high_mon: return '<i class="fa-solid fa-leaf"></i> Pembeli Baru'
        elif not is_low_recency and not is_high_mon: return '<i class="fa-solid fa-clock"></i> Hampir Hilang'
        else: return f'<i class="fa-solid fa-tag"></i> Kelompok {cluster_label}'

    df['Cluster'] = labels
    df['Nama_Cluster_HTML'] = [dynamic_cluster_name(lbl, labels, data_rfm) for lbl in labels]
    df['Nama_Cluster'] = df['Nama_Cluster_HTML'].str.replace(r'<[^>]*>', '', regex=True)

    if n_clusters_ > 1:
        mask_valid = labels != -1
        if mask_valid.sum() > 1 and len(set(labels[mask_valid])) > 1:
            sil_score = silhouette_score(df_scaled[mask_valid], labels[mask_valid])
            dbi_score = davies_bouldin_score(df_scaled[mask_valid], labels[mask_valid])
            st.metric("Silhouette Score (Skor Kualitas)", f"{sil_score:.2f}", help="Skala 0-1. Semakin mendekati 1 semakin baik.")
            st.metric("Davies-Bouldin Index (Indeks Davies-Bouldin)", f"{dbi_score:.2f}", help="Semakin rendah semakin baik (0 ke atas).")
        else:
            st.metric("Silhouette Score (Skor Kualitas)", "N/A")
            st.metric("Davies-Bouldin Index (Indeks Davies-Bouldin)", "N/A")
    else:
        st.metric("Silhouette Score (Skor Kualitas)", "N/A")
        st.metric("Davies-Bouldin Index (Indeks Davies-Bouldin)", "N/A")
        st.warning("Belum terbentuk kelompok.")

    st.caption(f"eps: {eps} | min_samples: {min_samples} | Kelompok: {n_clusters_}")

    df_display = df[
        (df['Pekerjaan'].isin(pilihan_pekerjaan)) &
        (df['Jenis Kelamin'].isin(pilihan_kelamin)) &
        (df['Pendidikan'].isin(pilihan_pendidikan)) &
        (df['Kategori Produk'].isin(pilihan_kategori)) &
        (df['Tingkat Harga'].isin(pilihan_harga)) &
        (df['Ruangan'].isin(pilihan_ruangan)) &
        (df['Item Spesifik'].isin(pilihan_item))
    ].copy()

# ========== HALAMAN 1: RINGKASAN BISNIS ==========
if menu == "Ringkasan Bisnis":
    st.markdown('<div class="main-header">Ringkasan Bisnis</div>', unsafe_allow_html=True)
    st.write("Sekilas informasi penting mengenai pelanggan dan pendapatan Anda (sesuai filter).")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Pelanggan", f"{len(df_display)}")
    col2.metric("Total Omzet", f"Rp {df_display['Monetary'].sum():,.0f}")
    
    nama_kelompok = [nama.strip() for nama in df_display['Nama_Cluster'].unique() if "VIP" not in nama]
    gabungan_nama = ", ".join(nama_kelompok) if nama_kelompok else "-"
    
    # --- PERBAIKAN: Angka di Card Custom diubah ke Hitam (#000000) ---
    custom_card = f"""
    <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem; box-shadow: 0 4px 12px rgba(0,0,0,0.05); height: 100%;">
        <div style="color: #4A5568; font-weight: 700; font-size: 1rem; margin-bottom: 0.2rem;">Kelompok Terbentuk</div>
        <div style="color: #000000; font-weight: 700; font-size: 2rem; margin-bottom: 0.8rem;">{n_clusters_}</div>
        <div style="color: #4A5568; font-size: 0.9rem; font-weight: 500; line-height: 1.4;">
            <b>Yaitu:</b> {gabungan_nama}
        </div>
    </div>
    """
    col3.markdown(custom_card, unsafe_allow_html=True)
    
    st.divider()
    col_kiri, col_kanan = st.columns([1.5, 1], gap="medium")
    with col_kiri:
        st.markdown('<div class="sub-header">Kontribusi Pendapatan</div>', unsafe_allow_html=True)
        st.caption("Semakin besar kotak, semakin besar kontribusi pendapatannya.")
        fig_tree = px.treemap(df_display, path=['Nama_Cluster'], values='Monetary',
                              color='Nama_Cluster', color_discrete_sequence=tema_warna)
        fig_tree.update_traces(textinfo="label+value", textfont=dict(size=14, color="#FFFFFF"))
        fig_tree.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_tree, use_container_width=True)
    with col_kanan:
        st.markdown('<div class="sub-header">Jumlah Pelanggan</div>', unsafe_allow_html=True)
        fig_pie = px.pie(df_display, names='Nama_Cluster', hole=0.5,
                         color_discrete_sequence=tema_warna)
        fig_pie.update_traces(textinfo="percent+label", textfont=dict(size=13, color="#FFFFFF"))
        fig_pie.update_layout(height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

# ========== HALAMAN 2: PENGELOMPOKAN PELANGGAN ==========
elif menu == "Pengelompokan Pelanggan":
    st.markdown('<div class="main-header">Pengelompokan Pelanggan</div>', unsafe_allow_html=True)
    st.write("Lihat bagaimana pelanggan Anda terbagi ke dalam kelompok berdasarkan kemiripan perilaku.")
    if n_clusters_ == 0:
        st.warning("Belum ada kelompok. Silakan ubah pengaturan di panel kiri.")
    else:
        col_kiri, col_kanan = st.columns(2, gap="medium")
        with col_kiri:
            st.markdown('<div class="sub-header">Grafik Penentuan Kemiripan</div>', unsafe_allow_html=True)
            st.caption("Garis merah adalah batas kemiripan. Titik di bawah garis berarti cukup mirip untuk digabungkan dalam satu kelompok.")
            
            n_neighbors_safe = min(min_samples, len(df_scaled) - 1)
            if n_neighbors_safe > 0:
                neighbors = NearestNeighbors(n_neighbors=n_neighbors_safe).fit(df_scaled)
                distances, _ = neighbors.kneighbors(df_scaled)
                distances = np.sort(distances[:, -1], axis=0)
                fig_eval = px.line(y=distances, labels={'index': 'Urutan Pelanggan', 'y': 'Tingkat Perbedaan'})
                fig_eval.add_hline(y=eps, line_dash="dash", line_color="#D05B43") 
                fig_eval.update_traces(line_color='#1A5F7A', line_width=3) 
                fig_eval.update_layout(height=380)
                st.plotly_chart(fig_eval, use_container_width=True)
            else:
                st.warning("Data terlalu sedikit untuk menampilkan grafik kemiripan. Silakan longgarkan filter Anda.")
                
        with col_kanan:
            st.markdown('<div class="sub-header">Peta Pelanggan (Tanggal Terakhir vs Total Belanja)</div>', unsafe_allow_html=True)
            st.caption("Setiap titik adalah satu pelanggan, warna menunjukkan kelompoknya.")
            fig_scatter = px.scatter(df_display, x='Tanggal Tampilan', y='Monetary', color='Nama_Cluster',
                                     labels={'Tanggal Tampilan': 'Tanggal Terakhir Belanja', 'Monetary': 'Monetary (Total Nominal Belanja)'},
                                     color_discrete_sequence=tema_warna)
            fig_scatter.update_traces(marker=dict(size=11, opacity=1.0, line=dict(width=1.5, color='#FFFFFF')))
            fig_scatter.update_layout(height=380)
            st.plotly_chart(fig_scatter, use_container_width=True)
        st.markdown('<div class="sub-header">Ringkasan Tiap Kelompok</div>', unsafe_allow_html=True)
        df_sum = df_display.groupby('Nama_Cluster').agg(Jumlah_Pelanggan=('Cluster', 'count'), Rata_Belanja=('Monetary', 'mean')).reset_index()
        df_sum['Rata_Belanja'] = "Rp " + df_sum['Rata_Belanja'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(df_sum, use_container_width=True, hide_index=True)
        st.divider()
        st.markdown('<div class="sub-header">Lihat Detail per Kelompok</div>', unsafe_allow_html=True)
        daftar_klaster = ["Semua"] + sorted(df_display['Nama_Cluster'].unique())
        pilihan_klaster = st.radio("Pilih Kelompok:", options=daftar_klaster, horizontal=True)
        df_tampil = df_display if pilihan_klaster == "Semua" else df_display[df_display['Nama_Cluster'] == pilihan_klaster]
        kolom_tampil = ['Usia', 'Tanggal Tampilan', 'Frequency', 'Monetary', 'Nama_Cluster',
                        'Pekerjaan', 'Jenis Kelamin', 'Pendidikan', 'Kategori Produk', 'Tingkat Harga', 'Ruangan', 'Item Spesifik']
        
        df_tampil_final = df_tampil[kolom_tampil].rename(columns={
            'Usia': 'Usia (Umur Pelanggan)', 
            'Tanggal Tampilan': 'Tanggal Terakhir Belanja',
            'Frequency': 'Frequency (Total Jumlah Transaksi)', 
            'Monetary': 'Monetary (Total Nominal Belanja)', 
            'Nama_Cluster': 'Kelompok'
        })
        st.dataframe(df_tampil_final, use_container_width=True)

# ========== HALAMAN 3: KEBIASAAN BELANJA ==========
elif menu == "Kebiasaan Belanja":
    st.markdown('<div class="main-header">Kebiasaan Belanja</div>', unsafe_allow_html=True)
    st.write("Bandingkan pola belanja nyata antar kelompok pelanggan.")
    st.markdown('<div class="sub-header">Rata-rata Indikator Belanja</div>', unsafe_allow_html=True)
    df_profil = df_display.groupby('Nama_Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
    df_melt = pd.melt(df_profil, id_vars='Nama_Cluster', value_vars=['Recency', 'Frequency', 'Monetary'],
                      var_name='Indikator', value_name='Rata-rata')
    
    df_melt['Indikator'] = df_melt['Indikator'].replace({
        'Recency': 'Recency (Lama Waktu Tidak Belanja)', 
        'Frequency': 'Frequency (Total Jumlah Transaksi)', 
        'Monetary': 'Monetary (Total Nominal Belanja)'
    })
    
    fig_bar = px.bar(df_melt, x='Nama_Cluster', y='Rata-rata', color='Indikator', barmode='group', text_auto='.1f',
                     color_discrete_sequence=['#1A5F7A', '#C9A96E', '#D05B43'])
    
    fig_bar.update_traces(textfont=dict(size=13, color='#2D3748'), textposition="outside", width=0.22)
    fig_bar.update_layout(height=420, legend=dict(orientation="h", y=1.15), xaxis=dict(type='category'), bargap=0.2)
    st.plotly_chart(fig_bar, use_container_width=True)
    st.caption("*Semakin kecil angka 'Recency', berarti pelanggan semakin baru saja berbelanja.*")
    
    col_kiri, col_kanan = st.columns(2, gap="medium")
    with col_kiri:
        st.markdown('<div class="sub-header">Variasi Total Belanja</div>', unsafe_allow_html=True)
        fig_box = px.box(df_display, x="Nama_Cluster", y="Monetary", color="Nama_Cluster",
                         labels={'Monetary': 'Monetary (Total Nominal Belanja)'}, color_discrete_sequence=tema_warna)
        fig_box.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)
    with col_kanan:
        st.markdown('<div class="sub-header">Profil Kekuatan (Diagram Jaring)</div>', unsafe_allow_html=True)
        if n_clusters_ > 0 or -1 in labels:
            max_vals = data_rfm[['Recency', 'Frequency', 'Monetary']].max()
            df_radar = df_profil.copy()
            
            df_radar['Recency'] = 1 - (df_radar['Recency'] / max_vals['Recency'])
            for col in ['Frequency', 'Monetary']:
                df_radar[col] = df_radar[col] / max_vals[col]
            
            df_melt_radar = pd.melt(df_radar, id_vars='Nama_Cluster', value_vars=['Recency', 'Frequency', 'Monetary'],
                                    var_name='Kriteria', value_name='Skor')
            
            df_melt_radar['Kriteria'] = df_melt_radar['Kriteria'].replace({
                'Recency': 'Kebaruan Belanja', 
                'Frequency': 'Keseringan Transaksi', 
                'Monetary': 'Total Nominal Belanja'
            })
            fig_radar = px.line_polar(df_melt_radar, r='Skor', theta='Kriteria', color='Nama_Cluster', line_close=True,
                                      color_discrete_sequence=tema_warna)
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), height=380)
            st.plotly_chart(fig_radar, use_container_width=True)
            st.caption("*Semakin penuh jaring, semakin kuat profil kelompok tersebut.")
        else:
            st.info("Diagram jaring tidak dapat ditampilkan karena tidak ada kelompok.")

# ========== HALAMAN 4: DEMOGRAFI PELANGGAN ==========
elif menu == "Demografi Pelanggan":
    st.markdown('<div class="main-header">Demografi Pelanggan</div>', unsafe_allow_html=True)
    st.write("Profil demografi dan preferensi produk berdasarkan data yang ditampilkan.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="sub-header">Pekerjaan</div>', unsafe_allow_html=True)
        fig_job = px.pie(df_display, names='Pekerjaan', color_discrete_sequence=tema_warna)
        fig_job.update_traces(textinfo="percent+label", textfont=dict(color="#FFFFFF"))
        fig_job.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_job, use_container_width=True)
    with col2:
        st.markdown('<div class="sub-header">Jenis Kelamin</div>', unsafe_allow_html=True)
        fig_gen = px.pie(df_display, names='Jenis Kelamin', color_discrete_sequence=tema_warna)
        fig_gen.update_traces(textinfo="percent+label", textfont=dict(color="#FFFFFF"))
        fig_gen.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_gen, use_container_width=True)
    with col3:
        st.markdown('<div class="sub-header">Pendidikan</div>', unsafe_allow_html=True)
        fig_edu = px.pie(df_display, names='Pendidikan', color_discrete_sequence=tema_warna)
        fig_edu.update_traces(textinfo="percent+label", textfont=dict(color="#FFFFFF"))
        fig_edu.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_edu, use_container_width=True)
    col4, col5 = st.columns(2)
    with col4:
        st.markdown('<div class="sub-header">Preferensi Kategori Produk</div>', unsafe_allow_html=True)
        fig_cat = px.histogram(df_display, x='Kategori Produk', color='Nama_Cluster', barmode='group', text_auto=True,
                               color_discrete_sequence=tema_warna,
                               labels={'count': 'count (Jumlah)'})
        fig_cat.update_layout(height=350, yaxis_title='count (Jumlah)')
        st.plotly_chart(fig_cat, use_container_width=True)
    with col5:
        st.markdown('<div class="sub-header">Tingkat Harga yang Diminati</div>', unsafe_allow_html=True)
        fig_price = px.histogram(df_display, x='Tingkat Harga', color='Nama_Cluster', barmode='group', text_auto=True,
                                 color_discrete_sequence=tema_warna,
                                 labels={'count': 'count (Jumlah)'})
        fig_price.update_layout(height=350, yaxis_title='count (Jumlah)')
        st.plotly_chart(fig_price, use_container_width=True)
    st.divider()
    st.markdown('<div class="sub-header">Distribusi Kelompok per Demografi & Preferensi</div>', unsafe_allow_html=True)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Pekerjaan", "Pendidikan", "Jenis Kelamin", "Kategori Produk", "Tingkat Harga"])
    with tab1:
        cross_job = pd.crosstab(df_display['Pekerjaan'], df_display['Nama_Cluster'])
        st.dataframe(cross_job, use_container_width=True)
    with tab2:
        cross_edu = pd.crosstab(df_display['Pendidikan'], df_display['Nama_Cluster'])
        st.dataframe(cross_edu, use_container_width=True)
    with tab3:
        cross_gen = pd.crosstab(df_display['Jenis Kelamin'], df_display['Nama_Cluster'])
        st.dataframe(cross_gen, use_container_width=True)
    with tab4:
        cross_cat = pd.crosstab(df_display['Kategori Produk'], df_display['Nama_Cluster'])
        st.dataframe(cross_cat, use_container_width=True)
    with tab5:
        cross_price = pd.crosstab(df_display['Tingkat Harga'], df_display['Nama_Cluster'])
        st.dataframe(cross_price, use_container_width=True)

# ========== HALAMAN 5: MINAT PRODUK & RUANGAN ==========
elif menu == "Minat Produk & Ruangan":
    st.markdown('<div class="main-header">Minat Produk & Ruangan</div>', unsafe_allow_html=True)
    st.write("Analisis bagian rumah mana yang paling sering didekorasi dan item apa yang paling laku per kelompok pelanggan.")
    col_kiri, col_kanan = st.columns(2, gap="medium")
    with col_kiri:
        st.markdown('<div class="sub-header">Ruangan Terpopuler</div>', unsafe_allow_html=True)
        fig_ruangan = px.histogram(df_display, y='Ruangan', color='Nama_Cluster', orientation='h', barmode='stack',
                                   text_auto=True, color_discrete_sequence=tema_warna,
                                   labels={'count': 'count (Jumlah)'})
        fig_ruangan.update_layout(height=400, yaxis={'categoryorder':'total ascending'}, xaxis_title='count (Jumlah)')
        st.plotly_chart(fig_ruangan, use_container_width=True)
    with col_kanan:
        st.markdown('<div class="sub-header">Item Paling Dicari</div>', unsafe_allow_html=True)
        fig_item = px.histogram(df_display, x='Item Spesifik', color='Nama_Cluster', barmode='group', text_auto=True,
                                color_discrete_sequence=tema_warna,
                                labels={'count': 'count (Jumlah)'})
        fig_item.update_layout(height=400, yaxis_title='count (Jumlah)')
        st.plotly_chart(fig_item, use_container_width=True)
    st.divider()
    st.markdown('<div class="sub-header">Peta Warna Minat Dekorasi</div>', unsafe_allow_html=True)
    st.caption("Warna yang lebih gelap menunjukkan minat yang sangat tinggi pada kombinasi Ruangan dan Item tersebut.")
    df_heatmap = pd.crosstab(df_display['Ruangan'], df_display['Item Spesifik'])
    fig_heat = px.imshow(df_heatmap, text_auto=True, aspect="auto", color_continuous_scale='YlOrBr', labels=dict(color="Jumlah Beli"))
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)

# ========== HALAMAN 6: DATA PELANGGAN LENGKAP ==========
elif menu == "Data Pelanggan Lengkap":
    st.markdown('<div class="main-header">Data Pelanggan Lengkap</div>', unsafe_allow_html=True)
    st.write("Tabel seluruh pelanggan beserta kelompok dan demografinya (sesuai filter).")
    kolom_tampil = ['Usia', 'Tanggal Tampilan', 'Frequency', 'Monetary', 'Nama_Cluster',
                    'Pekerjaan', 'Jenis Kelamin', 'Pendidikan', 'Kategori Produk', 'Tingkat Harga', 'Ruangan', 'Item Spesifik']
    
    df_display_renamed = df_display[kolom_tampil].rename(columns={
        'Usia': 'Usia (Umur Pelanggan)', 
        'Tanggal Tampilan': 'Tanggal Terakhir Belanja',
        'Frequency': 'Frequency (Total Jumlah Transaksi)', 
        'Monetary': 'Monetary (Total Nominal Belanja)', 
        'Nama_Cluster': 'Kelompok'
    })
    st.dataframe(df_display_renamed, use_container_width=True)