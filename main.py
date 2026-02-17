import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 0)
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Modeller
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_style("whitegrid")

# Algoritmalar
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")

# 1. VERİ YÜKLEME VE İLK HAZIRLIK (Sueda & Saliha & Seda)
# =================================================================
df = pd.read_csv(r"D:\Users\sedan\Downloads\archive\mobiles.csv.csv")
print(df.head(4))


def clean_price(price):
    if pd.isna(price) or price == "": return np.nan
    price_str = str(price).replace('₹', '').replace(',', '').strip()
    return float(price_str) if price_str else np.nan


df['price_numeric'] = df['price'].apply(clean_price)
print(df.price_numeric.head(4))

# Sueda'nın Marka/RAM bazlı akıllı dolgusu için geçici kolonlar
df['brand_temp'] = df['mobile_name'].str.split().str[0]
df['price_numeric'] = df['price_numeric'].fillna(df.groupby('brand_temp')['price_numeric'].transform('median'))
df['price_numeric'] = df['price_numeric'].fillna(df['price_numeric'].median())

# Hedef Değişken Oluşturma (Low, Mid, High)
df['price_category'] = pd.qcut(df['price_numeric'], q=3, labels=['Low', 'Mid', 'High'])

# =================================================================
# 2. MODELLEME ÖNCESİ DETAYLI ANALİZ (EDA - TÜM GRAFİKLER)
# =================================================================
print("\n--- 2. DETAYLI VERİ ANALİZİ VE GÖRSELLEŞTİRME ---")

# 1. Sınıf Dağılımı (Pasta ve Sütun Grafiği Yan Yana)
fig, ax = plt.subplots(1, 2, figsize=(16, 6))
class_counts = df['price_category'].value_counts()
ax[0].pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140,
          colors=sns.color_palette('pastel'))
ax[0].set_title('Fiyat Kategorilerinin Pasta Dağılımı')

sns.countplot(x='price_category', data=df, palette='viridis', ax=ax[1])
for p in ax[1].patches:
    ax[1].annotate(f'{p.get_height()}', (p.get_x() + 0.3, p.get_height() + 5))
ax[1].set_title('Fiyat Kategorilerinin Sayısal Dağılımı')
plt.show()

# 2. Korelasyon Matrisi (Sayısal Özellikler Arası İlişki)
plt.figure(figsize=(12, 10))
numeric_df = df.select_dtypes(include=[np.number])
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Özellikler Arası Korelasyon Matrisi')
plt.show()

# 3. Ayırt Edicilik Analizi (Kutu Grafikleri - Boxplots)
# Hangi özellik fiyatı daha çok etkiliyor?
features_to_box = ['specs_score', 'rating', 'price_numeric']
plt.figure(figsize=(18, 5))
for i, col in enumerate(features_to_box, 1):
    plt.subplot(1, 3, i)
    sns.boxplot(x='price_category', y=col, data=df, palette="Set2")
    plt.title(f'Fiyat Kategorisine Göre {col}')
plt.tight_layout()
plt.show()


# 4. İşlemci Markası ve Fiyat Segmenti İlişkisi (Sueda'nın analizi)
# Önce işlemci markasını çıkaralım (Sadece görselleştirme için geçici)
def temp_proc(text):
    text = str(text).lower()
    if 'snapdragon' in text:
        return 'Snapdragon'
    elif 'dimensity' in text:
        return 'Dimensity'
    elif 'exynos' in text:
        return 'Exynos'
    elif 'apple' in text:
        return 'Apple'
    else:
        return 'Other'


df['temp_processor'] = df['processor'].apply(temp_proc)

plt.figure(figsize=(12, 6))
sns.countplot(x='temp_processor', hue='price_category', data=df, palette='magma')
plt.title('İşlemci Markalarına Göre Fiyat Kategorisi Dağılımı')
plt.show()

# 5. Ekran Boyutu vs Batarya Kapasitesi (Scatter Plot)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='specs_score', y='price_numeric', hue='price_category', data=df, palette='deep',
                style='price_category', s=100)
plt.title('Specs Score vs Fiyat (Kategorilere Göre)')
plt.show()


# =================================================================
# 3. ÖZELLİK MÜHENDİSLİĞİ VE PIPELINE (Seda & Sueda & Saliha)
# =================================================================
class RobustMobileExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['brand'] = X['mobile_name'].str.split().str[0]

        # RAM (MB/GB çevrimi dahil - Saliha mantığı)
        def extract_ram(text):
            text = str(text)
            ram_match = re.search(r'(\d+)\s*GB\s*RAM', text)
            if ram_match: return float(ram_match.group(1))
            mb_match = re.search(r'(\d+)\s*MB\s*RAM', text)
            if mb_match: return float(mb_match.group(1)) / 1024
            return np.nan

        X['ram_gb'] = X['storage'].apply(extract_ram)

        X['rom_gb'] = X['storage'].apply(lambda x: 1024 if 'TB' in str(x) else (
            float(re.search(r'(\d+)', str(x)).group(1)) if re.search(r'(\d+)', str(x)) else np.nan))
        X['processor_speed'] = X['processor'].str.extract(r'(\d+(\.\d+)?)\s*GHz')[0].astype(float)
        X['processor_brand'] = X['processor'].apply(temp_proc)
        X['battery_mah'] = X['battery'].str.extract(r'(\d+)\s*mAh').astype(float)
        X['charging_w'] = X['battery'].str.extract(r'(\d+)\s*W').astype(float)
        X['screen_size'] = X['display'].str.extract(r'(\d+(\.\d+)?)\s*inches')[0].astype(float)
        X['has_5g'] = X['connectivity'].apply(lambda x: 1 if '5G' in str(x) else 0)
        X['main_camera_mp'] = X['camera'].str.extract(r'(\d+)\s*MP')[0].astype(float)

        drop_cols = ['mobile_name', 'processor', 'storage', 'battery', 'display', 'camera', 'connectivity',
                     'extra_storage', 'os', 'brand_temp', 'temp_processor', 'price_numeric']
        return X.drop(columns=[c for c in drop_cols if c in X.columns])


# Veri Bölme
X = df.drop(columns=['price', 'price_category'])
y = df['price_category'].cat.codes  # 0, 1, 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Pipeline Ön İşleme (Seda)
numeric_features = ['ram_gb', 'rom_gb', 'processor_speed', 'battery_mah', 'charging_w', 'screen_size', 'main_camera_mp',
                    'specs_score', 'rating']
categorical_features = ['brand', 'processor_brand']

preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline(
        [('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]),
     categorical_features)
])

# =================================================================
# 4. ÇOKLU MODELLEME VE SONUÇ ANALİZİ
# =================================================================
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel='rbf', probability=True),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=150, random_state=42)
}

results = {}
pipelines = {}

print("\n--- 4. MODEL EĞİTİMİ BAŞLADI ---")
for name, model in models.items():
    pipe = Pipeline([('extractor', RobustMobileExtractor()), ('preprocessor', preprocessor), ('classifier', model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    results[name] = accuracy_score(y_test, y_pred)
    pipelines[name] = pipe
    print(f"{name} Başarı Skoru: %{results[name] * 100:.2f}")

# =================================================================
# 5. MODELLEME SONRASI GÖRSELLEŞTİRMELER (FİNAL)
# =================================================================
print("\n--- 5. MODELLEME SONRASI ANALİZLER ---")

# 1. Tüm Modellerin Başarı Kıyaslaması (Bar Chart)
plt.figure(figsize=(12, 6))
res_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy']).sort_values(by='Accuracy', ascending=False)
ax = sns.barplot(x='Model', y='Accuracy', data=res_df, palette='coolwarm')
plt.ylim(0, 1.0)
for p in ax.patches:
    ax.annotate(f'%{p.get_height() * 100:.1f}', (p.get_x() + 0.3, p.get_height() + 0.02))
plt.title('Modellerin Başarı Oranı Kıyaslaması')
plt.show()

# 2. En İyi Model İçin Confusion Matrix (Hata Matrisi)
best_model_name = max(results, key=results.get)
y_pred_best = pipelines[best_model_name].predict(X_test)

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred_best)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Mid', 'High']).plot(cmap='Blues', ax=plt.gca())
plt.title(f'En İyi Model ({best_model_name}) Karmaşıklık Matrisi')
plt.grid(False)
plt.show()

# 3. Özellik Önemi (Random Forest - Sueda'nın Analizi)
rf_model = pipelines['Random Forest'].named_steps['classifier']
ohe_feature_names = pipelines['Random Forest'].named_steps['preprocessor'].transformers_[1][1].named_steps[
    'encoder'].get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(ohe_feature_names)
importances = rf_model.feature_importances_

feature_imp_df = pd.DataFrame({'Feature': all_feature_names, 'Importance': importances}).sort_values(by='Importance',
                                                                                                     ascending=False)

plt.figure(figsize=(10, 10))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df.head(20), palette='rocket')
plt.title('Fiyat Segmentini Belirleyen En Önemli 20 Özellik (RF)')
plt.show()

# 4. Hata Analizi (Post-Processing - Hangi Telefonlarda Yanıldık?)
X_test_with_results = X_test.copy()
X_test_with_results['Gerçek'] = y_test
X_test_with_results['Tahmin'] = y_pred_best
errors = X_test_with_results[X_test_with_results['Gerçek'] != X_test_with_results['Tahmin']]

if len(errors) > 0:
    plt.figure(figsize=(10, 5))
    errors['mobile_name'].str.split().str[0].value_counts().head(10).plot(kind='bar', color='salmon')
    plt.title('En Çok Hata Yapılan Markalar')
    plt.ylabel('Hata Sayısı')
    plt.show()

print(f"\nİşlem Tamamlandı. En iyi modeliniz: {best_model_name}")

from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# =================================================================
# 6. POST-PROCESSING GÖRSELLEŞTİRME (PROFESYONEL ANALİZ)
# =================================================================
print("\n--- 6. POST-PROCESSING GÖRSELLEŞTİRME VE GÜVEN ANALİZİ ---")

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(2, 2)

# --- A. GÜVEN (CALIBRATION) ANALİZİ ---
# Modelin tahmin olasılıkları gerçekle ne kadar örtüşüyor?
ax1 = fig.add_subplot(gs[0, 0])
for name, pipe in pipelines.items():
    # Sadece 'High' (2) sınıfı için olasılıkları alalım
    probs = pipe.predict_proba(X_test)[:, 2]
    prob_true, prob_pred = calibration_curve(y_test == 2, probs, n_bins=10)
    ax1.plot(prob_pred, prob_true, marker='o', label=name)

ax1.plot([0, 1], [0, 1], linestyle='--', color='black', label='Mükemmel Kalibrasyon')
ax1.set_title('Model Güven Analizi (Pahalı Segment İçin)')
ax1.set_xlabel('Tahmin Edilen Olasılık')
ax1.set_ylabel('Gerçekleşme Oranı')
ax1.legend()

# --- B. HATA YOĞUNLUK ANALİZİ (Specs Score vs Rating) ---
# Hatalar verinin neresinde toplanıyor?
ax2 = fig.add_subplot(gs[0, 1])
X_test_err = X_test.copy()
X_test_err['Hata'] = (y_test != y_pred_best).astype(int)

sns.scatterplot(data=X_test_err, x='specs_score', y='rating', hue='Hata',
                palette={0: 'lightgrey', 1: 'red'}, alpha=0.6, ax=ax2)
ax2.set_title('Hataların Teknik Puan/Rating Düzlemindeki Dağılımı')
# Kırmızı noktalar modelin yanıldığı telefonları gösterir.

# --- C. SINIF BAZLI BAŞARI (F1-SCORE KIYASLAMASI) ---
# Hangi segmenti (Low/Mid/High) daha iyi tahmin ediyoruz?
ax3 = fig.add_subplot(gs[1, 0])
from sklearn.metrics import f1_score

f1_results = []
for name, pipe in pipelines.items():
    pred = pipe.predict(X_test)
    f1 = f1_score(y_test, pred, average=None)
    for i, score in enumerate(f1):
        f1_results.append({'Model': name, 'Segment': ['Low', 'Mid', 'High'][i], 'F1-Score': score})

f1_df = pd.DataFrame(f1_results)
sns.barplot(data=f1_df, x='Segment', y='F1-Score', hue='Model', ax=ax3)
ax3.set_title('Segment Bazlı Model Başarıları (F1-Score)')
ax3.set_ylim(0, 1.1)

# --- D. YANLIŞ TAHMİN EDİLENLERİN ÖZELLİKLERİ ---
# Model pahalı telefonları neden ucuz sanıyor?
ax4 = fig.add_subplot(gs[1, 1])
err_subset = X_test_err[X_test_err['Hata'] == 1]
if not err_subset.empty:
    # En çok hata yapılan ilk 8 markayı göster
    err_subset['mobile_name'].str.split().str[0].value_counts().head(8).plot(kind='barh', color='salmon', ax=ax4)
    ax4.set_title('En Çok Hata Yapılan Markalar (Top 8)')
    ax4.set_xlabel('Hata Sayısı')

plt.tight_layout()
plt.show()

# ==========================================
# 7. MODELİN KARAR MEKANİZMASI (INTERPRETABILITY) - DÜZELTİLMİŞ
# ==========================================
try:
    import shap

    print("\n--- SHAP ANALİZİ (Açıklanabilirlik) ---")

    # En iyi model Random Forest veya Gradient Boosting ise (Tree-based modeller)
    target_model_name = "Random Forest"  # Veya en iyi model hangisiyse
    if target_model_name in pipelines:
        final_pipe = pipelines[target_model_name]

        # 1. Adım: Veriyi transform edelim
        # Önce Extractor (Regex kısımları)
        X_test_extracted = final_pipe.named_steps['extractor'].transform(X_test)

        # Sonra Preprocessor (Scaling ve One-Hot)
        # NOT: .toarray() ekleyerek sparse matrix hatasını çözüyoruz
        X_test_proc_final = final_pipe.named_steps['preprocessor'].transform(X_test_extracted)

        if hasattr(X_test_proc_final, "toarray"):
            X_test_proc_final = X_test_proc_final.toarray()

        # 2. Adım: Explainer oluşturma
        model = final_pipe.named_steps['classifier']
        explainer = shap.TreeExplainer(model)

        # 3. Adım: SHAP değerlerini hesaplama
        # check_additivity=False bazı küçük yuvarlama hatalarını görmezden gelir
        shap_values = explainer.shap_values(X_test_proc_final, check_additivity=False)

        # 4. Adım: Görselleştirme
        # SHAP versiyonuna göre çıktı formatı değişebilir, kontrol ederek ilerleyelim
        plt.figure(figsize=(10, 6))

        # Eğer shap_values bir listeyse (Multi-class eski sürüm)
        if isinstance(shap_values, list):
            # Class 2: High Segmenti
            shap.summary_plot(shap_values[2], X_test_proc_final,
                              feature_names=all_feature_names, max_display=10, show=False)
        else:
            # Eğer shap_values 3 boyutlu bir array ise (Yeni sürüm: samples, features, classes)
            # High segmenti (index 2) için tüm örnekleri ve özellikleri al
            shap.summary_plot(shap_values[:, :, 2], X_test_proc_final,
                              feature_names=all_feature_names, max_display=10, show=False)

        plt.title(f"Pahalı Segment (High) Tahminini Etkileyen Özellikler ({target_model_name})")
        plt.tight_layout()
        plt.show()
    else:
        print(f"\n[Uyarı]: SHAP analizi şu an sadece {target_model_name} için optimize edilmiştir.")


except ImportError:
    print("\n[Not]: SHAP kütüphanesi yüklü değil. 'pip install shap' komutuyla yükleyebilirsiniz.")
except Exception as e:
    print(f"\n[Hata]: SHAP analizi sırasında bir sorun oluştu: {e}")
# --- BU KISIM KODUN EN SONUNA, TAHMİNLER BİTTİKTEN SONRA EKLENECEK ---
# --- AMAÇ: Modelin sınıflandırma sonuçlarını kullanarak Fiyat/Performans analizi yapmak ---

# NOTE: This cell depends on 'df', 'X_test', 'y_test', 'y_pred' being defined from previous steps.
# Please ensure the 'FileNotFoundError' in the data loading cell is resolved first.
# Test verileri ile Tahminleri birleştiriyoruz
analiz_df = df.loc[X_test.index].copy()
analiz_df['Gercek_Segment'] = y_test
analiz_df['Tahmin_Segment'] = y_pred

# Ekstra özellik mühendisliğini analiz_df üzerinde uyguluyoruz
extractor = RobustMobileExtractor()
# Extractor price_numeric'i çıkaracağı için, onu geçici olarak drop ederek transform ediyoruz.
# Sonra geri ekleyeceğiz.
analiz_df_extracted = extractor.transform(analiz_df.drop(columns=['price_numeric']))

# Orijinal kolonları ve price_numeric'i koruyarak yeni çıkarılan kolonları ekleyelim
analiz_df_extracted['mobile_name'] = analiz_df['mobile_name']
analiz_df_extracted['price_numeric_display'] = analiz_df['price_numeric'] # Temizlenmiş fiyatı ekliyoruz
analiz_df_extracted['Gercek_Segment'] = analiz_df['Gercek_Segment']
analiz_df_extracted['Tahmin_Segment'] = analiz_df['Tahmin_Segment']

# Fiyat/Performans Fırsatı Olan Telefonlar
# Mantık: Telefonun özellikleri (donanımı) modelimize göre "Yüksek Segment"i hak ediyor,
# ama gerçek fiyatı (etiketi) "Düşük/Orta" segmentte kalmış.
fp_firsatlari = analiz_df_extracted[analiz_df_extracted['Gercek_Segment'] < analiz_df_extracted['Tahmin_Segment']]

print(f"\n--- EKSTRA ANALİZ: FİYAT/PERFORMANS FIRSATLARI ---")
print(f"Toplam {len(fp_firsatlari)} adet Fiyat/Performans ürünü tespit edildi.")
print("Örnek 5 Cihaz:")
display(fp_firsatlari[['mobile_name', 'price_numeric_display', 'ram_gb', 'processor_speed', 'Gercek_Segment', 'Tahmin_Segment']].head(5))