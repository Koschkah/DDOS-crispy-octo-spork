
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

def load_data(data_folder, set_name):
    """Esemény és komponens adatok betöltése és összefésülése"""
    e_path = os.path.join(data_folder, f"SCLDDOS2024_{set_name}_events.csv")
    c_path = os.path.join(data_folder, f"SCLDDoS2024_{set_name}_components.csv")
    
    # Csak a szükséges oszlopok (Attack code szigorúan kikerül)
    e_df = pd.read_csv(e_path, usecols=['Attack ID', 'Type', 'Start time', 'End time'])
    c_df = pd.read_csv(c_path, usecols=['Attack ID', 'Packet speed', 'Data speed', 'Avg packet len', 'Source IP count'])
    
    df = e_df.merge(c_df, on='Attack ID', how='left')
    df['Set'] = set_name
    
    # Idő konverzió és Duration számítás
    df['Start time'] = pd.to_datetime(df['Start time'], errors='coerce')
    df['End time'] = pd.to_datetime(df['End time'], errors='coerce')
    df['Duration_sec'] = (df['End time'] - df['Start time']).dt.total_seconds()
    
    return df

def feature_engineering_l1(df):
    """Komponens szintű jellemzők (Delta, DC-scaling, Packets_per_IP)"""
    df = df.copy()
    
    # 1. Packets per IP (DDoS vs DoS elkülönítéséhez)
    # 1e-6 hozzáadása a zéróosztás elkerülésére
    df['Packets_per_IP'] = df['Packet speed'] / (df['Source IP count'] + 1e-6)
    
    # 2. Delta features (változás sebessége az eseményen belül)
    cols_to_diff = ['Packet speed', 'Data speed', 'Avg packet len', 'Source IP count', 'Packets_per_IP']
    for col in cols_to_diff:
        df[f'{col}_delta'] = df.groupby('Attack ID')[col].diff().fillna(0)
    
    # 3. DC-specifikus skálázás (Normal traffic-hoz képest)
    cols_to_scale = ['Packet speed', 'Data speed', 'Avg packet len', 'Source IP count', 'Packets_per_IP']
    for col in cols_to_scale:
        normal_mean = df[df['Type'] == 'Normal traffic'][col].mean()
        df[f'{col}_vs_normal'] = df[col] / (normal_mean + 1e-6)

    # 4. Idő alapú jellemzők
    df['Hour'] = df['Start time'].dt.hour
    df['Day_of_week'] = df['Start time'].dt.dayofweek
    
    return df

def add_unsupervised_features(df, features_list, random_state=42):
    """PCA és KMeans anomália detektálás"""
    X = df[features_list].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2, random_state=random_state)
    pca_res = pca.fit_transform(X_scaled)
    df['PC1'] = pca_res[:, 0]
    df['PC2'] = pca_res[:, 1]
    
    # KMeans távolság (Anomália pontszám)
    kmeans = KMeans(n_clusters=5, random_state=random_state)
    kmeans.fit(X_scaled)
    df['cluster_dist'] = np.min(kmeans.transform(X_scaled), axis=1)
    
    return df

def aggregate_to_meta(df, label_encoder):
    """Komponensek aggregálása esemény szintre a Meta-modellhez"""
    # Alap aggregációk
    agg_dict = {
        'Duration_sec': 'first',
        'cluster_dist': ['mean', 'max'],
        'PC1': ['mean'],
        'Packet speed': ['max', 'mean'],
        'Source IP count': ['max', 'mean'],
        'Packets_per_IP': ['max', 'mean'], # <--- Új aggregáció
        'Packet speed_delta': 'max',
        'Label': 'first'
    }
    
    # Valószínűségek aggregációja (ha már lefutott az L1 predikció)
    for cls in label_encoder.classes_:
        if f'proba_{cls}' in df.columns:
            agg_dict[f'proba_{cls}'] = ['mean', 'max', lambda x: x.quantile(0.75)]
            
    meta = df.groupby('Attack ID').agg(agg_dict)
    
    # Oszlopnevek tisztítása
    meta.columns = ['_'.join(col).strip().replace('<lambda>', 'q75').replace('<', '').replace('>', '') for col in meta.columns.values]
    return meta
