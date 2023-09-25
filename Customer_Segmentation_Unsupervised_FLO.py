###############################################################
# Customer Segmentation with Unsupervised Learning
###############################################################

##################################
# Business Problem
##################################
## Unsupervised Learning methods (Kmeans, Hierarchical Clustering)
# are used to divide customers into clusters and observe their behavior.

##################################
# Data Set Story
##################################

# The dataset consists of the past shopping behavior of customers who made
# their last purchases in 2020 - 2021 as OmniChannel (both online and offline)
# consists of the information obtained.

# 20,000 observations, 13 variables

# master_id: Unique customer number
# order_channel : Which channel of the shopping platform is used (Android, iOS, Desktop, Mobile, Offline)
# last_order_channel : The channel where the last purchase was made
# first_order_date : Date of the customer's first purchase
# last_order_date : Date of the customer's last purchase
# last_order_date_online : The date of the last purchase made by the customer on the online platform
# last_order_date_offline : Date of the last purchase made by the customer on the offline platform
# order_num_total_ever_online : Total number of purchases made by the customer on the online platform
# order_num_total_ever_offline : Total number of purchases made by the customer offline
# customer_value_total_ever_offline : Total price paid by the customer for offline purchases
# customer_value_total_ever_online : Total price paid by the customer for online purchases
# interested_in_categories_12 : List of categories the customer has shopped in the last 12 months
# store_type : Refers to 3 different companies. If the person who shopped from company A also shopped
# from company B, it is written as A,B.

#####################
#EXPLORATORY DATA ANALYSIS
#####################

####################
#Needed libraries und Functions
####################

import pandas as pd
from scipy import stats
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import linkage
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None
warnings.simplefilter(action='ignore', category=Warning)

df_= pd.read_csv("DataScience/datasets/flo_data_20k.csv")
df = df_.copy()

####################
#Observations
####################

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


# Convert
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)

df["last_order_date"].apply(pd.to_datetime)
df["last_order_date"].max()
analysis_date = dt.datetime(2021,6,1)

df["recency"] = (analysis_date - df["last_order_date"]).dt.days
df["tenure"] = (df["last_order_date"]-df["first_order_date"]).dt.days

model_df = df[["order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
model_df.head()

#######################
#Customer Segmentation with K_Means
#######################

##################
#Standardization
##################

def check_skew(df_skew, column):
    skew = stats.skew(df_skew[column])
    skewtest = stats.skewtest(df_skew[column])
    plt.title('Distribution of ' + column)
    sns.distplot(df_skew[column],color = "g")
    print("{}'s: Skew: {}, : {}".format(column, skew, skewtest))
    return

plt.figure(figsize=(9, 9))
plt.subplot(6, 1, 1)
check_skew(model_df,'order_num_total_ever_online')
plt.subplot(6, 1, 2)
check_skew(model_df,'order_num_total_ever_offline')
plt.subplot(6, 1, 3)
check_skew(model_df,'customer_value_total_ever_offline')
plt.subplot(6, 1, 4)
check_skew(model_df,'customer_value_total_ever_online')
plt.subplot(6, 1, 5)
check_skew(model_df,'recency')
plt.subplot(6, 1, 6)
check_skew(model_df,'tenure')
plt.tight_layout()
plt.savefig('before_transform.png', format='png', dpi=1000)
plt.show()

# Log transformation to ensure normal distribution
model_df['order_num_total_ever_online']=np.log1p(model_df['order_num_total_ever_online'])
model_df['order_num_total_ever_offline']=np.log1p(model_df['order_num_total_ever_offline'])
model_df['customer_value_total_ever_offline']=np.log1p(model_df['customer_value_total_ever_offline'])
model_df['customer_value_total_ever_online']=np.log1p(model_df['customer_value_total_ever_online'])
model_df['recency']=np.log1p(model_df['recency'])
model_df['tenure']=np.log1p(model_df['tenure'])
model_df.head()

# Scaling
sc = MinMaxScaler((0, 1))
model_scaling = sc.fit_transform(model_df)
model_df=pd.DataFrame(model_scaling,columns=model_df.columns)
model_df.head()

#Determining the optimal number of clusters.
kmeans = KMeans(n_clusters=4, random_state=17).fit(model_df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_


kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(model_df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("SSE/SSR/SSD for Different K Values")
plt.title("Elbow Method for Optimum Number of Clusters")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 20))
elbow.fit(model_df)
elbow.show()

elbow.elbow_value_

model_df.head()

##################
#Base Model
##################
k_means = KMeans(n_clusters = 7, random_state= 42).fit(model_df)
segments=k_means.labels_
segments

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df["segment"] + 1
final_df.head()

########################
#Analyzing each segment statistically
########################

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})

###############################################################
# Customer Segmentation with Hierarchical Clustering
###############################################################

# Selecting the method used for clustering
hc_ward = linkage(model_df, 'ward')

plt.figure(figsize=(7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_ward,
           truncate_mode="lastp",
           p=10,
           show_contracted=True,
           leaf_font_size=10)
plt.axhline(y=12, color='r', linestyle='--')
plt.show()

##########################
#Modelling
#########################

hc = AgglomerativeClustering(n_clusters=6,linkage="ward" )
segments = hc.fit_predict(model_df)

final_df = df[["master_id","order_num_total_ever_online","order_num_total_ever_offline","customer_value_total_ever_offline","customer_value_total_ever_online","recency","tenure"]]
final_df["segment"] = segments
final_df.head()

########################
#Analyzing each segment statistically
########################

final_df.groupby("segment").agg({"order_num_total_ever_online":["mean","min","max"],
                                  "order_num_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_offline":["mean","min","max"],
                                  "customer_value_total_ever_online":["mean","min","max"],
                                  "recency":["mean","min","max"],
                                  "tenure":["mean","min","max","count"]})
