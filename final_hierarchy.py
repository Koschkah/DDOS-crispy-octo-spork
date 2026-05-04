
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import tools # Saját modul betöltése

# 1. KONFIGURÁCIÓ ÉS OPTIMÁLIS PARAMÉTEREK
DATA_FOLDER = "data"
VIS_FOLDER = "vis"
SETS = ['SetA', 'SetB', 'SetC']
RANDOM_STATE = 42

# Az Optuna által talált legújabb, dúsított jellemzőkre optimalizált paraméterek
BEST_PARAMS = {
    'weight_DDoS': 3.954574589526711,
    'weight_Suspicious': 3.2272369854634686,
    'n_estimators': 398,
    'max_depth': 7,
    'learning_rate': 0.013241877560997274,
    'subsample': 0.8786215741375665,
    'colsample_bytree': 0.8088574050259245,
    'gamma': 1.6708917282931974,
}

def main():
    # 2. ADATOK BETÖLTÉSE ÉS ELŐKÉSZÍTÉSE
    print("--- 1. Lépés: Adatok betöltése és tisztítása ---")
    all_data = pd.concat([tools.load_data(DATA_FOLDER, s) for s in SETS], ignore_index=True)
    all_data = all_data.dropna(subset=['Packet speed', 'Type', 'Attack ID'])
    
    print("--- 2. Lépés: Komponens szintű Feature Engineering ---")
    all_data = tools.feature_engineering_l1(all_data)
    
    # Kezdeti jellemzők az unsupervised tanuláshoz
    base_features = ['Packet speed', 'Data speed', 'Avg packet len', 'Source IP count', 
                     'Packets_per_IP', 'Duration_sec', 'Packet speed_delta', 
                     'Packets_per_IP_delta', 'Packet speed_vs_normal', 'Packets_per_IP_vs_normal']
    
    all_data = tools.add_unsupervised_features(all_data, base_features, random_state=RANDOM_STATE)
    
    # LEVEL 1 Jellemzők (Attack code kikerült!)
    features_l1 = base_features + ['PC1', 'PC2', 'cluster_dist']
    
    le = LabelEncoder()
    all_data['Label'] = le.fit_transform(all_data['Type'].astype(str))
    
    train_df = all_data[all_data['Set'].isin(['SetA', 'SetB'])]
    test_df = all_data[all_data['Set'] == 'SetC']

    # 3. LEVEL 1: KOMPONENS SZINTŰ XGBOOST
    print("\n--- 3. Lépés: Level 1 (Komponens) modell tanítása ---")
    xgb_l1 = XGBClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, tree_method='hist', n_jobs=-1)
    xgb_l1.fit(train_df[features_l1], train_df['Label'])
    
    # Valószínűségek kinyerése
    for i, cls in enumerate(le.classes_):
        train_df[f'proba_{cls}'] = xgb_l1.predict_proba(train_df[features_l1])[:, i]
        test_df[f'proba_{cls}'] = xgb_l1.predict_proba(test_df[features_l1])[:, i]

    # 4. LEVEL 2: META-JELLEMZŐK GENERÁLÁSA
    print("--- 4. Lépés: Aggregáció esemény (Attack ID) szintre ---")
    train_meta = tools.aggregate_to_meta(train_df, le)
    test_meta = tools.aggregate_to_meta(test_df, le)
    
    X_train_meta = train_meta.drop(columns=['Label_first'])
    y_train_meta = train_meta['Label_first']
    X_test_meta = test_meta.drop(columns=['Label_first'])
    y_test_meta = test_meta['Label_first']

    # Súlyozás beállítása a legjobb paraméterekkel
    weights = {
        le.transform(['DDoS attack'])[0]: BEST_PARAMS['weight_DDoS'],
        le.transform(['Suspicious traffic'])[0]: BEST_PARAMS['weight_Suspicious'],
        le.transform(['Normal traffic'])[0]: 1.0
    }
    sample_weights = np.array([weights[y] for y in y_train_meta])

    # 5. LEVEL 2: VÉGSŐ META-XGBOOST
    print("--- 5. Lépés: Level 2 (Meta) modell tanítása optimalizált paraméterekkel ---")
    l2_params = {k: v for k, v in BEST_PARAMS.items() if not k.startswith('weight_')}
    xgb_meta = XGBClassifier(**l2_params, random_state=RANDOM_STATE, n_jobs=-1)
    xgb_meta.fit(X_train_meta, y_train_meta, sample_weight=sample_weights)

    # 6. KIÉRTÉKELÉS ÉS VIZUALIZÁCIÓ
    print("\n--- 6. Lépés: Kiértékelés és Vizualizáció ---")
    y_pred = xgb_meta.predict(X_test_meta)
    y_proba = xgb_meta.predict_proba(X_test_meta)
    
    # Report mentése
    print(classification_report(y_test_meta, y_pred, target_names=le.classes_))
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_meta, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Final Model Confusion Matrix (Event Level)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(os.path.join(VIS_FOLDER, 'final_confusion_matrix.png'))
    
    # 2. Feature Importance
    plt.figure(figsize=(12, 8))
    fi = pd.DataFrame({'feature': X_train_meta.columns, 'importance': xgb_meta.feature_importances_}).sort_values('importance', ascending=False)
    sns.barplot(x='importance', y='feature', data=fi.head(15), palette='magma')
    plt.title('Top 15 Features (Meta Model)')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_FOLDER, 'final_feature_importance.png'))

    # 3. Class Separation Plot (DDoS valószínűségek eloszlása)
    plt.figure(figsize=(10, 6))
    ddos_idx = le.transform(['DDoS attack'])[0]
    for i, class_name in enumerate(le.classes_):
        subset = y_proba[y_test_meta == i, ddos_idx]
        sns.kdeplot(subset, label=f'True {class_name}', fill=True, alpha=0.3)
    plt.title('DDoS Probability Distribution by True Class')
    plt.xlabel('Predicted DDoS Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(VIS_FOLDER, 'class_separation_dist.png'))

    # 4. Precision-Recall Curve (Multi-class)
    plt.figure(figsize=(10, 8))
    y_test_bin = label_binarize(y_test_meta, classes=[0, 1, 2])
    for i, class_name in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_proba[:, i])
        avg_prec = average_precision_score(y_test_bin[:, i], y_proba[:, i])
        plt.plot(recall, precision, label=f'{class_name} (AP={avg_prec:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve by Class')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(VIS_FOLDER, 'precision_recall_curves.png'))

    # 5. Meta-Feature Correlation Heatmap
    plt.figure(figsize=(14, 10))
    corr = X_train_meta.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='RdBu_r', center=0, annot=False, linewidths=.5)
    plt.title('Meta-Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_FOLDER, 'meta_feature_correlation.png'))
    
    print(f"\nSiker! Minden vizualizáció mentve a '{VIS_FOLDER}' mappába.")

if __name__ == "__main__":
    main()
