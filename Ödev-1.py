import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)


#### DEGISKENLER ####

# reviewerID: Kullanıcı ID'si
# asin: Ürün ID'si
# reviewerName: Kullanıcı adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Kullanıcının yazdığı inceleme metni (deüerlendirme)
# overall: Ürün rating'i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı (unix time)
# reviewTime: Değerlendirme zamanı (RAW)
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı

df = pd.read_csv('datasets/amazon_review.csv')
df.head(50)

#### Görev 1 ####


# Puan dağılımları:

df['overall'].value_counts()

# Average: (= 4.5875)
df['overall'].mean()

# Time-Based Weighted Average: (=4.6987)
# Ağırlıkların toplamı 100 olmalı ve etkisini kaybettiği için çok fazla bölemeyiz.

df.loc[df['day_diff'] <= 30, 'overall'].mean() * 28 / 100 + \
df.loc[(df['day_diff'] > 30) & (df['day_diff'] <= 90), 'overall'].mean() * 26 / 100 + \
df.loc[(df['day_diff'] > 90) & (df['day_diff'] <= 180), 'overall'].mean() * 24 / 100 + \
df.loc[(df['day_diff'] > 180), 'overall'].mean() * 22 / 100

def time_based_weighted_average (dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe['day_diff'] <= 30, 'overall'].mean() * 28 / 100 + \
           dataframe.loc[(dataframe['day_diff'] > 30) & (dataframe['day_diff'] <= 90), 'overall'].mean() * 26 / 100 + \
           dataframe.loc[(dataframe['day_diff'] > 90) & (dataframe['day_diff'] <= 180), 'overall'].mean() * 24 / 100 + \
           dataframe.loc[(dataframe['day_diff'] > 180), 'overall'].mean() * 22 / 100

time_based_weighted_average(df)

# İlgili ürün zamana göre hesaplandığında rating average, time based rating average'a göre daha düşük çıkmıştır. Ürünün son zamanlarda ilgisiz kalıp
# kalmadığı faktörü incelenmiş ve puanlara yantısılmıştır. Görüldüğü gibi ürünün son zamanlarda ilgisiz kalma durumu yoktur.

#### Görev 2 ####

# Wilson Lower Bound Score:

def wilson_lower_bound(up , down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2*n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df.sort_values('helpful', ascending=False)

df['helpful_no'] = df['total_vote'] - df['helpful_yes']

df['wilson_lower_bound_score']  = df.apply(lambda x: wilson_lower_bound(x['helpful_yes'], x['helpful_no']) , axis=1)

df.head()
df.sort_values('wilson_lower_bound_score', ascending=False).head(20)
# Elimizdeki örnekleme ilişkin uprate oranının istatistiksel olarak % 95 güvenilirlik ile en az kaç olabileceğini bulduk.