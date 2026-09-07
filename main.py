#############################################
# ADIM 1: EDA
#############################################
import labels
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import pyplot as plt
import subprocess
subprocess.run(["pip", "install", "missingno"])
import missingno as msno
import datetime as dt
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
#pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)


r1 = pd.read_csv("reviews_0-250.csv")
r2 = pd.read_csv("reviews_250-500.csv")
r3 = pd.read_csv("reviews_500-750.csv")
r4 = pd.read_csv("reviews_750-1250.csv")
r5 = pd.read_csv("reviews_1250-end.csv")

reviews = pd.concat([r1,r2,r3,r4,r5])
products = pd.read_csv("product_info.csv")

main_data = pd.merge(reviews,products,on="product_id",how="left")

main_data.to_csv('sephora_all_data.csv', index=False)

df__ = pd.read_csv("sephora_all_data.csv", low_memory=False)
df = df__.copy()

#############################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
#############################################



silinecekler = ["product_name_y","price_usd_y","rating_y","brand_name_y"] #x ve y nin bifarki yok
df.drop(silinecekler, axis=1, inplace=True)

df.rename(columns = {
    "product_name_x": "product_name",
    "price_usd_x": "price",
    "rating_x": "rating",
    "brand_name_x": "brand_name"
}, inplace = True)


def grab_col_names(dataframe, cat_th=20,car_th=200):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == 'object']
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                    dataframe[col].dtypes != 'object']
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == 'object']
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != 'object']
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"num_but_cat: {len(num_but_cat)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols,cat_but_car = grab_col_names(df)


#############################################
#  ADIM 2: Data Cleaning (Veri Temizleme)
#############################################
#############################################
# 1.Missing number (Kayıp Değerler)
#############################################

#msno.matrix(df)

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


#%50 den fazla boslari var bunlarin
col_names_to_drop = ["variation_desc","sale_price_usd","value_price_usd",
                     "child_min_price","child_max_price","helpfulness",
                     "review_text","review_title", "highlights", #modelimiz bunlari okuyamaz
                     "Unnamed: 0", "brand_id"] #matematiksel karsiligi yok bu id lerin

df.drop(columns = col_names_to_drop, inplace=True, errors="ignore")

#%10 ve %20 arasi boslari var ve bizim icin onemliler
important_cols = ["hair_color","eye_color","skin_tone","is_recommended",
                  "tertiary_category","skin_type"]

for col in important_cols:
    print(df[col].dtype)

for col in important_cols:
    df[col].fillna("Unknown", inplace=True)

df[important_cols].isnull().sum()

#%5 ve daha az olanlar yani geri kalan tum eksikler diger hepsine islem
#yaptim yukarida
df.dropna(inplace=True)

df.isnull().sum().sum()

cat_cols, num_cols,cat_but_car = grab_col_names(df)



#############################################
# 2.Outliers (Aykırı Değerler)
#############################################

# for col in num_cols:
#     fig, ax = plt.subplots(figsize=(8, 4))
#     sns.boxplot(x=df[col], ax=ax)
#     ax.set_title(col)
#     plt.tight_layout()
#     plt.show()


def outlier_thresholds(dataframe,col_name,q1=0.01,q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit,up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)][col_name].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    print(col,check_outlier(df,col))



def outlier_count(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    outliers = dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)]
    print(f"{col_name}: {len(outliers)} outlier -- Rate: %{round(len(outliers) / len(dataframe) * 100, 2)}")

for col in num_cols:
    outlier_count(df, col)

#############################################
# Aykırı Değer Problemini Çözme
#############################################


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))





check_df(df)


#############################################
# 3.Korelasyon Analizi
#############################################

corr_matrix = df[num_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Sayısal Değişkenler Arası Korelasyon")
plt.show()



#############################################
#  ADIM 3: ÖZELLİK MÜHENDİSİĞİ (FEATURE ENGINEERING)
#############################################

# --- 1. Ürün Etkileşim ve Popülerlik Metrikleri ---
# Sadece yorum sayısı değil, "loves" üzerinden gerçek arzuyu ölçüyoruz.
df["interaction_score"] = df["loves_count"] / (df["reviews"] + 1 )

# Viral Ürün: 5 yıldızlı ve ortalamanın üzerinde ilgi gören "yıldız" ürünler.
average_loves = df["loves_count"].mean()
df["is_viral_product"] = ((df["rating"] == 5.0) & (df["loves_count"] > average_loves)).astype(int)

# Popülerlik Kategorisi: Ürünleri popülaritelerine göre 4 gruba ayırıyoruz.
#df['popularity_label'] = pd.qcut(df['loves_count'], q=4, labels=['Low', 'Medium', 'High', 'Viral'])

#############################################
# --- 2. Fiyat ve Değer Analizi ---
# Para-Değer Skoru (Value Score): 1 birim fiyat başına düşen beğeni.
df['value_score'] = df['loves_count'] / (df['price'] + 1 )

# Fiyat Kategorisi: Bütçeden lükse segmente ediyoruz.
df['price_category'] = pd.cut(
    df['price'],
    bins=[0, 25, 50, 100, df['price'].max()],
    labels=['bütçe', 'orta', 'premium', 'lüks']
)

# Lüks Segment Alıcı Flag: En pahalı %25'lik dilimde mi?
premium_threshold = df["price"].quantile(0.75)
df["is_premium_shopper"] = (df["price"] > premium_threshold).astype(int)

#############################################
# --- 3. Geri Bildirim ve Marka Kalitesi ---
# Olumlu geri bildirimlerin toplam geri bildirime oranı (Helpfulness).
#df['pos_feedback_ratio'] = df['total_pos_feedback_count'] / (df['total_feedback_count'] + 1)

# Marka Ortalama Rating: Her markanın genel performansını veriye ekliyoruz.
df['brand_avg_rating'] = df.groupby('brand_name')['rating'].transform('mean')

#############################################
# --- 4. Kullanıcı ve Ürün Karakteristiği ---
# Özel Ürün: Sınırlı üretim veya Sephora'ya özel ürün mü?
#df['is_special'] = ((df['limited_edition'] == 1) | (df['sephora_exclusive'] == 1)).astype(int)

# Profil Zenginliği: Kullanıcı ne kadar detaylı profil bilgisi vermiş? (0-4 arası)
profile_cols = ["skin_type", "skin_tone", "hair_color", "eye_color"]
df["profile_engagement"] = (df[profile_cols] != "Unknown").sum(axis=1)

# Cilt Tipi Uyumluluğu: One-Hot Encoding mantığıyla cilt tiplerini ayırıyoruz.
# for skin in ['dry', 'oily', 'combination', 'normal']:
#     df[f'is_{skin}'] = df['skin_type'].apply(lambda x: 1 if skin in str(x).lower() else 0)
#





################################################################################
# SONUÇ KONTROLÜ
################################################################################
df.head()
cat_cols, num_cols,cat_but_car = grab_col_names(df)

num_cols.append("rating")
cat_cols.remove("rating")

for col in num_cols:
    print(df[col].dtypes)

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

df.describe().T

print(f"Özellik mühendisliği tamamlandı. Yeni sütun sayısı: {len(df.columns)}")
display_cols = ['product_name', 'interaction_score', 'is_viral_product',
                'price_category', 'is_special', 'profile_engagement']


# Doğrulama Testi
verification_cols = ['interaction_score','is_viral_product', 'value_score','brand_avg_rating', 'profile_engagement']

# İstatistiksel özet
print("--- Yeni Özelliklerin İstatistiksel Özeti ---")
print(df[verification_cols].describe().T)

# Mantık Kontrolü: Viral ürünlerin rating ortalaması 5 mi?
print("\n--- Viral Ürün Kontrolü (Rating Ortalaması) ---")
print(df.groupby('is_viral_product')['rating'].mean())

# Mantık Kontrolü: Profil doluluğu 0 olanlar gerçekten 'Unknown' mu?
print("\n--- Profil Bilgisi 'Unknown' Olanların Engagement Skoru ---")
print(df[df['profile_engagement'] == 0][['skin_type', 'skin_tone']].head(3))




#encoding
ohe_cols = ["skin_type","skin_tone","hair_color","eye_color","secondary_category"]
#sadece bunlara yaptim cunku digerlerinin sayisi cok fazla ve modelimiz kaldirmaz en onemlileri aldim
df = pd.get_dummies(df,columns=ohe_cols,drop_first=True)

df.columns
df.head()

###############################################################
# ADIM 4: Makine Öğrenmesi ile Müşteri Kümeleme (K-Means Clustering)
###############################################################
df['price_original'] = df['price']
df['rating_original'] = df['rating']
################################################################################
# 1. Model Giriş Özelliklerinin Ölçeklendirilmesi (StandardScaler)
################################################################################

###for customers
################################################################################
# Müşteri Davranış Metriklerinin Hesaplanması
################################################################################

musteri_profil = df.groupby('author_id').agg(
    ort_harcama=('price', 'mean'),           # ortalama ürün fiyatı
    toplam_harcama=('price', 'sum'),         # monetary
    satin_alma_sayisi=('product_id', 'count'), # frequency
    ort_value_score=('value_score', 'mean'), # fiyat/performans eğilimi
    farkli_marka=('brand_name', 'nunique'),  # kaç farklı marka
    ort_rating=('rating', 'mean')            # verdiği ortalama puan
).reset_index()

print(musteri_profil.shape)
musteri_profil.head()


scaler = StandardScaler()
features = ['ort_harcama', 'toplam_harcama', 'satin_alma_sayisi', 'ort_value_score', 'farkli_marka', 'ort_rating']
musteri_scaled = scaler.fit_transform(musteri_profil[features])


###for products
cols_scale = [col for col in num_cols]
ss = StandardScaler()
df[cols_scale] = ss.fit_transform(df[cols_scale])


################################################################################
# 2. Optimum Küme Sayısının Belirlenmesi (Elbow Method / Dirsek Yöntemi)
################################################################################


###for customers

inertia = []
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(musteri_scaled)
    inertia.append(kmeans.inertia_)


plt.plot(range(2, 11), inertia, marker='o')
plt.xlabel('Küme Sayısı')
plt.ylabel('Inertia')
plt.title('Elbow Yöntemi')
plt.show()

###for products

df_model = df.select_dtypes( exclude = ['object','category'])
df_model.shape

wcss = []
for k in range(2,11):
    kmeans = KMeans(n_clusters = k , random_state = 42)
    kmeans.fit(df_model)
    wcss.append(kmeans.inertia_)

plt.figure(figsize = (8,5))
plt.plot(range(2,11),wcss, marker ='o')
plt.title('Elbow plot')
plt.xlabel( 'Kume Sayisi (k)')
plt.ylabel( 'WCSS (Hata Kareler Toplami)')
plt.xticks(range(2,11))
plt.grid(True)
plt.show()


################################################################################
# 3. Modelin Fit Edilmesi ve Segmentlerin Oluşturulması
################################################################################
###for customers

kmeans = KMeans(n_clusters=4, random_state=42)
musteri_profil['segment'] = kmeans.fit_predict(musteri_scaled)


df["price_original"].describe().T

# Segmentlere bak
musteri_profil['segment'].value_counts().sort_index()


musteri_profil.groupby('segment').agg({
    'ort_harcama': 'mean',
    'toplam_harcama': 'mean',
    'satin_alma_sayisi': 'mean',
    'ort_value_score': 'mean',
    'farkli_marka': 'mean',
    'ort_rating': 'mean'
}).round(2)


segment_isimleri = {
    0: 'Bilinçli Alıcılar',  # 32$, az marka, yüksek rating
    1: 'Kaşif & Sadık',  # 58$,20 marka, dengeli
    2: 'Lüks Alıcılar',  # 82$, az marka, seçici
    3: 'Potansiyel Sadıklar',  # 448$ toplam, 5 marka
}

musteri_profil['segment_adi'] = musteri_profil['segment'].map(segment_isimleri)
musteri_profil['segment_adi'].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(
    [35, 25, 15, 25],
    labels=[segment_isimleri[i] for i in sorted(segment_isimleri.keys())],
    autopct='%1.1f%%',  # Yüzdelik dilimleri gösterme formatı
    startangle=140,     # Başlangıç açısı
    colors=['#aec7e8', '#ffbb78', '#98df8a', '#ff9896'],
    explode=(0, 0, 0, 0.05),
    shadow=True         # Hafif gölge efekti
)
plt.axis('equal')
plt.title('Müşteri Segmentleri Dağılımı', fontsize=14, fontweight='bold', pad=20)
plt.show()

### for products

kmeans_final = KMeans(n_clusters = 4, random_state = 42)
kmeans_final.fit(df_model)

df['Segment'] = kmeans_final.labels_

#fiyat bazli icin
segment_profil = df.groupby("Segment"). agg({
    "price" : "mean",
    "rating" : "mean",
    "interaction_score" : "mean",
    "value_score" : "mean",
    "profile_engagement" : "mean",
    "is_viral_product" : "mean",
    "is_premium_shopper" : "mean",
})

print(segment_profil.T)



#karakteristik icin
ohe_demografi = [col for col in df.columns if any(i in col for i in ["skin_type", "skin_tone", "hair_color", "eye_color"])]
baskin_demografi = df.groupby("Segment")[ohe_demografi].mean().T
print(baskin_demografi)

#musteri segmentasyonu grafiksel gosterimi

segment_counts = df["Segment"].value_counts().sort_index()

labels = {
    0: 'Sosyal Trend & F/P Avcıları', #En uygun fiyatlı, f/p odaklı ve viral/popüler ürünleri kovalayan kitle.
    1: 'Ultra VIP Lüks', #Fiyata hiç bakmayan, sadece en ekstrem pahalı ve premium ürünlere odaklanan tepe kitle.
    2: 'Premium Sadıklar', #Kesinlikle lüks/premium tercih eden ama fiyat dengesini koruyan en mutlu kitle.
    3: 'Dengeli Ortalamalar' #Her metriği orta/dengeli seyreden, doğru tekliflerle premium tarafa kayabilecek kitle.
}



plt.figure(figsize=(8, 8))
plt.pie(
    [40, 10, 25, 25],
    labels=[labels[i] for i in sorted(labels.keys())],
    autopct='%1.1f%%',  # Yüzdelik dilimleri gösterme formatı
    startangle=140,     # Başlangıç açısı
    colors=['#f472b6', '#701a75', '#a855f7', '#fbcfe8'],
    explode=(0, 0.1, 0.05, 0),
    shadow=True,        # Derinlik hissi için gölge efekti
)
plt.axis('equal')
plt.title('Müşteri Segmentleri Dağılımı', fontsize=14, fontweight='bold', pad=20)
plt.show()


print(labels)
df.groupby('Segment').size()

###############################################################
# 4. ÜRÜN DNA'SI (KOSİNÜS BENZERLİĞİ MATRİSİNİN ÜRETİLMESİ)
###############################################################

print("Kosinüs Benzerlik Matrisi Hesaplanıyor...")

# Ürünlerin içerik tabanlı benzerliğini ölçmek için sayısal ve One-Hot özellikleri seçiyoruz
kosinus_ozellikleri = ['price', 'rating', 'interaction_score',
                       'skin_type_combination', 'skin_type_dry', 'skin_type_normal', 'skin_type_oily']

df.columns
# Her ürünü tekilleştirip özelliklerini alıyoruz
df_model = df.drop_duplicates(subset=['product_name']).set_index('product_name')[kosinus_ozellikleri]
df_model.fillna(0, inplace=True)

# Özellikleri standartlaştırıp matrisi çıkarıyoruz
model_scaled = scaler.fit_transform(df_model)
cosine_sim = cosine_similarity(model_scaled)

# Matrisi Pandas DataFrame'ine çeviriyoruz (Shape Mismatch hatasını çözdüğümüz yer!)
cosine_sim_df = pd.DataFrame(cosine_sim, index=df_model.index, columns=df_model.index)

cosine_sim_df.shape

###############################################################
# ADIM 5: Tavsiye Sisteminin Geliştirilmesi (Recommendation Systems)
###############################################################

################################################################################
# 5.1. Müşteri Segmentlerine Göre Popüler ve Yüksek Puanlı Ürünlerin Tespiti
################################################################################

# Ana df ile musteri_profil'i birleştir
df_ = df.merge(musteri_profil[['author_id', 'segment_adi']],
                     on='author_id', how='left')

df_.head()

df_.columns
# Her segmentte en yüksek puanlı ürünler
segment_urunler = df_.groupby(['segment_adi', 'product_id']).agg(
    urun_adi=('product_name', 'first'),
    marka=('brand_name', 'first'),
    ort_rating=('rating_original', 'mean'),
    fiyat=('price_original', 'first'),
    loves=('loves_count', 'first'),
    cilt_tipi_kuru=('skin_type_dry', 'first'),
    cilt_tipi_yagly=('skin_type_oily', 'first'),
    cilt_tipi_karma=('skin_type_combination', 'first'),
    cilt_tipi_normal=('skin_type_normal', 'first')
).reset_index()

# Web sitesinin veriyi hızlıca okuyabilmesi için csv olarak kaydediyoruz
segment_urunler.to_csv('segment_urunler.csv', index=False)

segment_urunler['segment_adi'].unique()
segment_urunler.head()
################################################################################
# 5.2. Karar Mekanizması: Kullanıcı Profili ve Tercihlerine Dayalı Filtreleme Algoritması
################################################################################
def content_based_recommender(product_name=None, musteri_segmenti=None, cosine_sim_df=cosine_sim_df, df=df,
                              ort_harcama=None, farkli_marka=None, cilt_tipi=None, max_fiyat=None, min_rating=4.0,segment_urunler=segment_urunler):
    # --------------------------------------------------------------------------
    # DURUM 1: SOĞUK BAŞLANGIÇ (Ürün Yok, İlk Defa Siteye Giren Ziyaretçi)
    # --------------------------------------------------------------------------
    if product_name is None:
        print("🔍 [Sistem]: Anket Modu Aktif (Yeni Kullanıcı Filtrelemesi)")

        # Segmenti belirle
        if ort_harcama < 35 and farkli_marka <= 2:
            tahmini_segment = 'Bilinçli Alıcılar'
        elif ort_harcama >= 75 and farkli_marka <= 2:
            tahmini_segment = 'Lüks Alıcılar'
        elif farkli_marka >= 10:
            tahmini_segment = 'Kaşif & Sadık'
        else:
            tahmini_segment = 'Potansiyel Sadıklar'

        print(f"Segmentiniz: {tahmini_segment}")

        # O segmentin ürünlerini filtrele
        oneri = segment_urunler[segment_urunler['segment_adi'] == tahmini_segment].copy()

        # Kullanıcının verdiği cilt tipi cevabına göre filtreleme
        if cilt_tipi == 'kuru':
            oneri = oneri[oneri['cilt_tipi_kuru'] == 1]
        elif cilt_tipi == 'yağlı':
            oneri = oneri[oneri['cilt_tipi_yagly'] == 1]
        elif cilt_tipi == 'karma':
            oneri = oneri[oneri['cilt_tipi_karma'] == 1]
        elif cilt_tipi == 'normal':
            oneri = oneri[oneri['cilt_tipi_normal'] == 1]

        # Fiyat ve rating filtrele

        oneri = oneri[oneri['fiyat'] <= max_fiyat]
        oneri = oneri[oneri['ort_rating'] >= min_rating]

        # En iyi 5 ürünü getir
        return oneri.sort_values('ort_rating', ascending=False)[['urun_adi', 'marka', 'fiyat', 'ort_rating']].head(
            5)

    # --------------------------------------------------------------------------
    # DURUM 2: YAPAY ZEKA MODU (Ürün Var, Kosinüs + K-Means Aktif)
    # --------------------------------------------------------------------------
    else:
        print(f" [Sistem]: Yapay Zeka Aktif ('{product_name}' için kişiselleştiriliyor)")
        # 1. Ürün kontrolü
        if product_name not in cosine_sim_df.index:
            return "Urun Bulunamadi. Urun ismini kontrol ediniz!"

        # 2. Ürünün içerik/fiziksel ikizlerini bul (İlk 20 benzer ürün)
        similarity_scores = cosine_sim_df[product_name].sort_values(ascending=False)
        similar_product_indices = similarity_scores[1:21].index.tolist()

        oneri_havuzu = df[df["product_name"].isin(similar_product_indices)].copy()
        oneri_havuzu = oneri_havuzu.drop_duplicates(subset=["product_name"])

        # 3. Müşteri segmentine göre Akıllı Sıralama (Yeni K-Means Proffiline Göre)
        if musteri_segmenti == 0:
            oneri_havuzu = oneri_havuzu.sort_values(by=["is_viral_product", "value_score"], ascending=[False, False])

            # Segment 1: Ultra VIP Lüks
            # Karakteri: Fiyata bakmazlar, en pahalı (is_premium_shopper ve price) ürünleri isterler.
        elif musteri_segmenti == 1:
            oneri_havuzu = oneri_havuzu.sort_values(by=["is_premium_shopper", "price_original"],
                                                    ascending=[False, False])

            # Segment 2: Premium Sadıklar
            # Karakteri: Kaliteli, premium ürünleri severler ama fiyattan ziyade memnuniyete (rating) önem verirler.
        elif musteri_segmenti == 2:
            oneri_havuzu = oneri_havuzu.sort_values(by=["rating", "is_premium_shopper"], ascending=[False, False])

            # Segment 3: Dengeli Ortalamalar
            # Karakteri: Genel etkileşimi (interaction_score) yüksek, stabil ana kitle.
        elif musteri_segmenti == 3:
            oneri_havuzu = oneri_havuzu.sort_values(by=["interaction_score", "rating"], ascending=[False, False])

            # Cold Start / Tanımlanamayan Segment Durumu (Fallback)
        else:
            oneri_havuzu = oneri_havuzu.sort_values(by=["interaction_score"], ascending=False)
        return oneri_havuzu[["product_name", "price_original", "rating_original"]].head(5)




################################################################################
# 5.3. Tavsiye Motorunun Test Edilmesi ve Örnek Kullanıcı Çıktısı
################################################################################

print("\n" + "="*50)
print("🧪 TEST 1: SOĞUK BAŞLANGIÇ")
print("="*50)
# Kullanıcı siteye yeni girdi, 120 dolar harcamayı planlıyor (Lüks Odaklı olmalı)
test_1_sonuc = content_based_recommender(
    ort_harcama=50,
    farkli_marka=2,
    cilt_tipi='kuru',
    max_fiyat=150,
    min_rating=3
)
print(test_1_sonuc)


print("\n" + "="*50)
print("🧪 TEST 2: YAPAY ZEKA MODU ÇALIŞIYOR MU?")
print("="*50)
# Kullanıcı sitede bir ürüne tıkladı ve segmenti "Aşırı Bütçe Odaklı"
# Sistem Kosinüs matrisine bakıp, en ucuz/viral olanları üste dizmeli!
rastgele_bir_urun = df_model.index[0] # Verisetindeki ilk ürünü test için alıyoruz

test_2_sonuc = content_based_recommender(
    product_name=rastgele_bir_urun,
    musteri_segmenti='Aşırı Bütçe Odaklı'
)
print(f"Seçilen Ürün: {rastgele_bir_urun}")
print(test_2_sonuc)








