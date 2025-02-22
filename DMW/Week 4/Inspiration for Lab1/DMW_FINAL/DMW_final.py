import numpy as np
import pandas as pd
from itertools import chain, product
from pandas.api.types import is_numeric_dtype
from pandas.api.types import is_object_dtype

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

from sklearn.decomposition import TruncatedSVD
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from scipy.spatial.distance import euclidean, cityblock

import fim

import warnings
warnings.filterwarnings("ignore")


from IPython.display import HTML
from sklearn.cluster import OPTICS, cluster_optics_dbscan
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import fcluster
from fastcluster import linkage
from tqdm.notebook import tqdm

import psycopg2
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
import re

conn = psycopg2.connect(dbname="postgis", 
                 user="gsa2022", 
                 password="g5!V%T1Vmd", 
                 host="192.168.212.99", 
                 port=32771)


class FigureLabeler:
    """
    This class add a figure or table number and caption
    """
    def __init__(self):
        self.fig_num = 1
        self.table_num = 1
    
    def fig_caption(self, title, caption):
        global fig_num
        """Print figure caption on jupyter notebook"""
        display(HTML(
            f"""<p style="font-size:12px;font-style:default;"><b>
            Figure {self.fig_num}. {title}.</b><br>{caption}</p>"""))
        self.fig_num += 1

    def table_caption(self, title, caption):
        global table_num
        """Print table caption on jupyter notebook"""
        display(HTML(
            f"""<p style="font-size:12px;font-style:default;"><b>
            Table {self.table_num}. {title}.</b><br>{caption}</p>""")
               )
        self.table_num += 1
    
    def reset_to(self, fig_num=None, table_num=None):
        """Manually reset figure number or table number."""
        if fig_num is not None:
            self.fig_num = fig_num
        if table_num is not None:
            self.table_num = table_num
 
            
def get_cat_num_df(df):
    """This function will return the numerical and
    categorical dataframes with the original dataframe as an input"""
    num_df = []
    cat_df = []
    for col in df.columns:
        if is_numeric_dtype(df[col]):
            num_df.append(col)
                
        if is_object_dtype(df[col]):
            cat_df.append(col)
    return num_df, cat_df


def get_null_df(df):
    """This function will return the null dataframe with
    the original dataframe as an input"""
    null_df = pd.DataFrame(columns = ['Column', 'Type', 'Total NaN', '%'])
    col_null = df.columns[df.isna().any()].to_list()
    L = len(df)
    for i, col in enumerate(col_null):
        T = 0
        if is_numeric_dtype(df[col]):
            T = "Numerical"  
        else:
            T = "Categorical"
        nulls = len(df[df[col].isna() == True][col])
        null_df.loc[i] = {'Column': col, 
                          'Type': T,
                          'Total NaN': nulls,
                          '%': (nulls / L)*100}
    return null_df


def calc_interquartile(df, column):
    """This function will calculate the interquartile range
    for outlier handling"""
    first_quartile, third_quartile = (
        np.percentile(df[column], 25), np.percentile(df[column], 75)
    )
    iqr = third_quartile - first_quartile
    cutoff = iqr*1.5
    lower, upper = first_quartile - cutoff , third_quartile + cutoff
    upper_outliers = df[df[column] > upper]
    lower_outliers = df[df[column] < lower]
    return lower, upper, lower_outliers.shape[0]+upper_outliers.shape[0]

def get_outliers(df, num_feat):
    """
    This function gets the outliers for the given dataframe and
    numerical features list. It returns a dataframe containing
    feature names, total number of outliers, and upper and lower limits.
    """
    outlier_df = pd.DataFrame(columns=['Feature', 'Total Outliers',
                                       'Upper limit', 'Lower limit'])
    for col in num_feat:
        lower, upper, total = calc_interquartile(df, col)
        
        if total and upper and lower:
            row = {
                'Feature': col,
                'Total Outliers': total,
                'Upper limit': upper,
                'Lower limit': lower
            }
            outlier_df = pd.concat([outlier_df,
                                    pd.DataFrame(row, index=[0])],
                                   ignore_index=True)
    return outlier_df


def remove_outliers(df, outlier_df, num_feat):
    """This function will drag the outliers back to the interquartile range
    range for the result of outlier handling"""
    for col in outlier_df['Feature'].to_list():
        upper = (outlier_df[outlier_df['Feature']== col ]
                 ['Upper limit'].values[0])
        lower = (outlier_df[outlier_df['Feature']== col ]
                 ['Lower limit'].values[0])
        df[col] = np.where(df[col]>upper, upper, df[col])
        df[col] = np.where(df[col]<lower, lower, df[col])
    return df


def make_upper(df):
    """
    This function will take in a dataframe and make the columns indicated
    upper and stripped
    """
    region_list = ['REGION', 'PROVINCE', 'DONOR', 'ORGANIZATION',
                   'CITY/ MUNICIPALITY', 'BARANGAY',
                   'CLUSTER (select from the list)',
                   'ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)']
    for i in region_list:
        df[i] = df[i].str.upper()
        df[i] = df[i].str.strip()
        
        
def pooled_within_ssd(X, y, centroids, dist):
    """Compute pooled within-cluster sum of squares around the cluster mean
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
        
    Returns
    -------
    float
        Pooled within-cluster sum of squares around the cluster mean
    """
    score_list = []
    for i in range(len(centroids)):
        df = pd.DataFrame(X)
        df['labeled'] = y
        df = (
            df[df['labeled']==i]
            .drop(columns='labeled')
            .reset_index(drop=True)
        )
        ls = []
        for j in range(len(df)):
            ls.append(df.loc[j].tolist())
        score = 0
        for k in range(len(ls)):
            score += 1/(2*len(ls)) * dist(ls[k], centroids[i])**2
        score_list.append(score)
    return sum(score_list)


def gap_statistic(X, y, centroids, dist, b, clusterer, random_state=None):
    """Compute the gap statistic
    
    Parameters
    ----------
    X : array
        Design matrix with each row corresponding to a point
    y : array
        Class label of each point
    centroids : array
        Cluster centroids
    dist : callable
        Distance between two points. It should accept two arrays, each 
        corresponding to the coordinates of each point
    b : int
        Number of realizations for the reference distribution
    clusterer : KMeans
        Clusterer object that will be used for clustering the reference 
        realizations
    random_state : int, default=None
        Determines random number generation for realizations
        
    Returns
    -------
    gs : float
        Gap statistic
    gs_std : float
        Standard deviation of gap statistic
    """
    rng = np.random.default_rng(random_state)
    
    gap_stats = []
    w_k = pooled_within_ssd(X, y, centroids, dist)
    
    for _ in range(b):
        X_new = rng.uniform(low=np.min(X, axis=0),
                         high=np.max(X, axis=0),
                         size=X.shape)
        
        k_sorter = clusterer
        y_new = k_sorter.fit_predict(X_new)
        centroids_new = k_sorter.cluster_centers_
        
        w_ki = pooled_within_ssd(X_new, y_new, centroids_new, dist)
        
        gap_stats.append(np.log(w_ki) - np.log(w_k))
        
    gap_statistic = np.sum(gap_stats) / b
    gap_statistic_std = np.std(gap_stats)
    
    return gap_statistic, gap_statistic_std


def cluster_range(X, clusterer, k_start, k_stop, actual=None):
    """
    This function will get the cluster range from start to stop
    and X as the input in numpy
    """
    ys = []
    centers = []
    inertias = []
    chs = []
    scs = []
    dbs = []
    gss = []
    gssds = []
    ps = []
    amis = []
    ars = []
    for k in range(k_start, k_stop+1):
        clusterer_k = clone(clusterer)
        clusterer_k.set_params(n_clusters=k, random_state=1337)

        y = clusterer_k.fit_predict(X)
        ys.append(y)

        centers.append(clusterer_k.cluster_centers_)

        inertias.append(clusterer_k.inertia_)
        chs.append(calinski_harabasz_score(X, y))
        scs.append(silhouette_score(X, y))
        dbs.append(davies_bouldin_score(X, y))
        gs = gap_statistic(X, y, clusterer_k.cluster_centers_, 
                                 euclidean, 5, 
                                 clone(clusterer).set_params(n_clusters=k), 
                                 random_state=1337)
        gss.append(gs[0])
        gssds.append(gs[1])

        if actual is not None:
            ps.append(purity(actual, y))
            amis.append(adjusted_mutual_info_score(actual, y))
            ars.append(adjusted_rand_score(actual, y))

    keys = ['ys', 'centers', 'inertias', 'chs',
            'scs', 'dbs', 'gss', 'gssds']
    
    if actual is not None:
        keys.extend(['ps', 'amis', 'ars'])
        dict_act_Not_None = (
            dict(zip(keys,
                     [ys, centers, inertias, chs, scs,
                      dbs, gss, gssds, ps, amis, ars]))
        )
        return dict_act_Not_None
    else:
        dict_act_None = (
            dict(zip(keys,
                     [ys, centers, inertias, chs, 
                      scs, dbs, gss, gssds]))
        )
        return dict_act_None
    
def plot_clusters(X, ys, centers, transformer):
    """Plot clusters given the design matrix and cluster labels"""
    k_max = len(ys) + 1
    k_mid = k_max//2 + 2
    fig, ax = plt.subplots(2, k_max//2, dpi=150, sharex=True, sharey=True, 
                           figsize=(7,4), subplot_kw=dict(aspect='equal'),
                           gridspec_kw=dict(wspace=0.01))
    for k,y,cs in zip(range(2, k_max+1), ys, centers):
        centroids_new = transformer.transform(cs)
        if k < k_mid:
            ax[0][k%k_mid-2].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[0][k%k_mid-2].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y)) + 1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[0][k%k_mid-2].set_title('$k=%d$'%k)
        else:
            ax[1][k%k_mid].scatter(*zip(*X), c=y, s=1, alpha=0.8)
            ax[1][k%k_mid].scatter(
                centroids_new[:,0],
                centroids_new[:,1],
                s=10,
                c=range(int(max(y))+1),
                marker='s',
                ec='k',
                lw=1
            );
            ax[1][k%k_mid].set_title('$k=%d$'%k)
    return ax

def plot_internal(inertias, chs, scs, dbs, gss, gssds):
    """Plot internal validation values"""
    fig, ax = plt.subplots()
    ks = np.arange(2, len(inertias)+2)
    ax.plot(ks, inertias, '-o', label='SSE')
    ax.plot(ks, chs, '-ro', label='CH')
    ax.set_xlabel('$k$')
    ax.set_ylabel('SSE/CH')
    lines, labels = ax.get_legend_handles_labels()
    ax2 = ax.twinx()
    ax2.errorbar(ks, gss, gssds, fmt='-go', label='Gap statistic')
    ax2.plot(ks, scs, '-ko', label='Silhouette coefficient')
    ax2.plot(ks, dbs, '-gs', label='DB')
    ax2.set_ylabel('Gap statistic/Silhouette/DB')
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines+lines2, labels+labels2)
    return ax


def read_odette_excel():
    "Read the Excel file for Typhoon Odette"
    odette_df = (
        pd.read_excel('220702_3w-typhoon-rai_odette-consolidated-hdx.xlsx',
                      header=1)
    )
    return odette_df


def process_data(odette_df):
    """
    This function will complete the necessary processing for the 
    frequent itemsent mining analysis for this implementation
    """
    # Total Categorical Features and Total Numerical Features
    num_features, cat_features = get_cat_num_df(odette_df)
    
    # Null Values DataFrame
    null_df = get_null_df(odette_df)
    out_null = null_df.sort_values(by='%', ascending=False).head(10)
    
    odette_processed = odette_df[
        ['ORGANIZATION', 'DONOR', 'CLUSTER (select from the list)',
         'REGION', 'PROVINCE', 'CITY/ MUNICIPALITY', 'BARANGAY',
         'EVACUATION SITE', 'CLUSTER ACTIVITY',
         'ACTIVITY DESCRIPTION (short specific activity description)',
         'NUMBER OF PEOPLE REACHED (Individuals)', ' ACTIVITY START DATE',
         'ACTIVITY END DATE',
         'ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)',
         'CASH MODALITY (if cash assistance)']
    ]
    
    odette_processed['ORGANIZATION'] = (  
        odette_processed['ORGANIZATION'].str.upper()
    )
    odette_processed['ORGANIZATION'] = (
        odette_processed['ORGANIZATION'].str.strip()
    )
    odette_processed['REGION'] = odette_processed['REGION'].str.upper()
    odette_processed['REGION'] = odette_processed['REGION'].str.strip()
    odette_processed['PROVINCE'] = odette_processed['PROVINCE'].str.upper()
    odette_processed['PROVINCE'] = odette_processed['PROVINCE'].str.strip()
    odette_processed['CLUSTER (select from the list)'] = (
        odette_processed['CLUSTER (select from the list)'].str.upper()
    )
    odette_processed['CLUSTER (select from the list)'] = (
        odette_processed['CLUSTER (select from the list)'].str.strip()
    )
    (
        odette_processed['ORGANIZATION']
        .replace(['ABSCBN FOUNDATION INC - BANTAY BATA 163'],
                 ['ABS-CBN FOUNDATION, INC. - BANTAY BATA 163'],
                 inplace=True)
    )

    odette_processed['CITY/ MUNICIPALITY'] = (
        odette_processed['CITY/ MUNICIPALITY'].str.upper()
    )
    odette_processed['BARANGAY'] = odette_processed['BARANGAY'].str.upper()

    odette_processed['CITY/ MUNICIPALITY'] = (
        odette_processed['PROVINCE']
        + ', '
        + odette_processed['CITY/ MUNICIPALITY']
    )
    odette_processed['BARANGAY'] = odette_processed['BARANGAY'].astype(str)
    odette_processed['BARANGAY'] = (
        odette_processed['CITY/ MUNICIPALITY']
        + ', '
        + odette_processed['BARANGAY']
    )
    
    odette_processed['CITY/ MUNICIPALITY'] = (
        odette_processed['CITY/ MUNICIPALITY'].str.upper()
    )
    odette_processed['CITY/ MUNICIPALITY'] = (
        odette_processed['CITY/ MUNICIPALITY'].str.strip()
    )

    odette_processed['ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)'] = (
        odette_processed['ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)']
        .str.upper()
    )
    odette_processed['ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)'] = (
        odette_processed['ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)']
        .str.strip()
    )
    return odette_processed


def split_to_cityprov(df, col_name, prefix):
    """
    Split a column whose values is a single string of the names
    of a province and city. Lowercase them and remove parenthesis and the
    string inside.
    """
    df = df.copy()
    split_prov_city = (pd.DataFrame(df[col_name]
                                .apply(lambda x: x.lower().split(', '))
                                .to_list()))
    split_prov_city.columns = [prefix+'_prov', prefix+'_city']
    split_cols = list(split_prov_city.columns)

    df_pc = pd.concat([df, split_prov_city], axis=1)
    pc_cols = list(df_pc.columns)
    
    # align city names with ph_shp dataframe
    df_pc[prefix+'_city'] = (
        df_pc[prefix+'_city'].str.replace(
            r'(\s\(\w.*?\))', '', regex=True)
    )
    # city of maasin => maasin city
    df_pc[prefix+'_city'] = (
        df_pc[prefix+'_city'].str.replace(
            'city of maasin', 'maasin city')
    )
    df_pc[prefix+'_city'] = (
        df_pc[prefix+'_city'].str.replace(
            'city of maasin (capital)', 'maasin city')
    )
    # display(df_pc)
    return df_pc


def load_final_df():
    """
    Return the Typhoon Odette data in DataFrame format with
    renamed columns.
    """
    odette_df = read_odette_excel()
    final_df = (
        process_data(odette_df)
        .rename(columns={
            'ACTIVITY STATUS \n(ONGOING, COMPLETED, PLANNED)':
            'ACTIVITY_STATUS',
            'CITY/ MUNICIPALITY': 'CITY_MUNICIPALITY',
            'CLUSTER (select from the list)': 'CLUSTER',
            'ACTIVITY DESCRIPTION (short specific activity description)':
            'ACTIVITY_DESCRIPTION',
            ' ACTIVITY START DATE': 'ACTIVITY_START_DATE',
            'ACTIVITY END DATE': 'ACTIVITY_END_DATE'}
               )
    )
    return final_df


def load_ph_shp():
    """
    Return the psycopg2 SQL database containing shape files for
    the Philippines. Columns name_1 and name_2 are lowercase'd.
    """
    qry = """
    SELECT *
    FROM
        gadm.ph
    """
    ph_shp = gpd.read_postgis(qry, con=conn, geom_col='geom')
    # A copy of ph_shp with lower() name_1 and name_2 for uniformity
    ph_shp_l = ph_shp.copy()
    ph_shp_l['name_1'] = ph_shp_l['name_1'].str.lower()
    ph_shp_l['name_2'] = ph_shp_l['name_2'].str.lower()
    return ph_shp_l


def plot_caraga_shp(ax, caraga_shp=False, legend=True):
    """
    Function that plots the CARAGA region on an plt.Axis object
    """
    caraga_shp.plot(ax=ax, column='population',
                    cmap='Greys',
                    edgecolor='black',
                    linewidth=.3,
                    alpha=1,
                    legend=legend,
                    legend_kwds={'label': "Population as of July 2021",
                                 'orientation': "vertical", 'shrink': 0.75})
    return ax


def fim_gsa(df_tmp_grp, caraga_shp=False, cols=4):
    """
    Iterates over the 2-week time periods of the transaction-item database
    and implements steps 1 to 4. Plots the recommended underserved
    provinces for all time periods.
    """
    periods_weeks = df_tmp_grp['ACTIVITY_START_DATE'].unique()

    # Subplots are organized in a Rows x Cols Grid
    # Tot and Cols are known
    Tot = len(periods_weeks)
    Cols = cols

    # Compute Rows required
    Rows = Tot // Cols 

    # EDIT for correct number of rows:
    # If one additional row is necessary -> add one:
    if Tot % Cols != 0:
        Rows += 1

    # Create a Position index
    position = range(1,Tot + 1)

    # Create main figure
    fig = plt.figure(1, figsize=(25,25))

    for k in tqdm(range(Tot)):
        wk = (k+1)*2

        df_wk = df_tmp_grp.copy().set_index('ACTIVITY_START_DATE')
        df_wk = df_wk.loc[[periods_weeks[k]], :]
        # df_wk = df_wk.loc[periods_weeks[k]:, :]

        if df_wk.shape[0] < 2:
            continue

        fim_data = df_wk['CLUSTER'].to_numpy()
        freq_itm = fim.apriori(fim_data, supp=-2, target='s',
                               report='a', zmin=1)
        freq_itm_srtd = sorted(freq_itm, key=lambda x: -x[1])

        # Get the top most frequent itemsets
        top3_freq_itm = freq_itm_srtd[:5]

        df_wk['SERVICE'] = (
            df_wk['CLUSTER']
            .apply(lambda x: sum([len(set(itm[0]).intersection(x))
                                  for itm in top3_freq_itm]))
        )
        split_ = split_to_cityprov(df_wk.reset_index(),
                                              'CITY_MUNICIPALITY',
                                              f'{wk}W')
        df_overview = caraga_shp.merge(split_, how='left',
                                       left_on=list(caraga_shp.columns[:2]),
                                       right_on=[f'{wk}W_prov',
                                                 f'{wk}W_city'])
        df_overview.drop(columns=[f'{wk}W_prov',f'{wk}W_city',
                                  'PROVINCE','CITY_MUNICIPALITY'],
                         inplace=True)

        # Extract aided cities/municipalities
        df_aided = df_overview.query('SERVICE.notna()')
        # display(df_aided)

        # Extract unaided cities/municipalities
        df_n_aided = df_overview.query('SERVICE.isna()')

        # measure the distance of every aided province
        # with unaided cities/municipalities
        unsrvd_res = []
        for i, row in df_aided.iterrows():
            row_geom = row['geom']

            # distance of one aided province from all
            # unaided cities/municipalities
            dists_unserved = (gpd.GeoSeries(df_n_aided['geom'])
                              .distance(row_geom.centroid)
                              .sort_values(ascending=True)
                              .to_frame().rename(columns={0:'distance'}))

            # Get nearest 5 unaided cities/municipalities
            nearest_unsrvd = dists_unserved.iloc[:5] 
            nrst_idx = nearest_unsrvd.index

            # Get the population of the nearest unaided cities/municipalities
            pop_nrst_unsrvd = (df_n_aided.loc[nrst_idx]
                               .sort_values(by='population', ascending=False)
                               .iloc[[0]])
            pop_nrst_unsrvd = (pop_nrst_unsrvd.merge(dists_unserved,
                                                     left_index=True,
                                                     right_index=True))
            unsrvd_res.append(pop_nrst_unsrvd)

        # Sort nearest unaided cities/municipalities
        # by largest to smallest population
        unsrvd_res = pd.concat(unsrvd_res)
        # Get the top 3 by nearest distance and largest population
        unsrvd_res_reco = (unsrvd_res
                           .dissolve(by=['province',
                                         'city_municipality',
                                         'population'],aggfunc='mean',
                                     as_index=False)
                           .sort_values(by=['population','distance'],
                                        ascending=[False,True]).iloc[:3,:])

        # add every single subplot to the figure with a for loop
        ax = fig.add_subplot(Rows, Cols, position[k])

        plot_caraga_shp(ax, caraga_shp, legend=False)

        df_aided.plot(ax=ax,
                        facecolor='none',
                        edgecolor='green',
                        linewidth=1.5,
                        alpha=1)

        unsrvd_res_reco.plot(ax=ax, column='population',
                        facecolor='none',
                        edgecolor='blue',
                        linewidth=1,
                        hatch='//', alpha=0.7)

        ax.set_title(f'Week {wk}', fontsize=16)

    # plt.tight_layout()
    plt.suptitle('Recommended areas for crucial aid\nfrom Week 2 to 28',
                 fontsize=20, weight='bold', y=.925, x=.35)
    plt.subplots_adjust(wspace=.75, right=.6, hspace=.05, top=.9, bottom=.1)
    plt.show()
    
    
def week_fim_gsa(df_tmp_grp, caraga_shp=False, wk_idx=0):
    """
    Implements steps 1 to 4 for a specified 2-week time period. Plots
    the recommended underserved provinces for the time period. Returns
    three dataframes that will help in coordinating relief operations
    given a scenario.
    """
    wk = (wk_idx+1)*2
    periods_weeks = df_tmp_grp['ACTIVITY_START_DATE'].unique()
    
    df_wk = df_tmp_grp.copy().set_index('ACTIVITY_START_DATE')
    df_wk = df_wk.loc[[periods_weeks[wk]], :]
    
    # get the most frequent itemsets in Week 2
    fim_data = df_wk['CLUSTER'].to_numpy()
    freq_itm = fim.apriori(fim_data, supp=-2, target='s', report='a', zmin=1)
    freq_itm_srtd = sorted(freq_itm, key=lambda x: -x[1])
    # print(freq_itm_srtd)

    # Get the top most frequent itemsets
    top_freq_itm = freq_itm_srtd[:5]
    top_freq_itm_df = pd.DataFrame(top_freq_itm,
                                   columns=['most_freq_relief',
                                            'transaction_cnt'])
    # display(top_freq_itm_df)
    
    df_wk['AID_CNT'] = (df_wk['CLUSTER']
                     .apply(lambda x: sum([len(set(itm[0]).intersection(x))
                                           for itm in top_freq_itm])))

    split_ = split_to_cityprov(df_wk.reset_index(),
                                          'CITY_MUNICIPALITY',
                                          f'{wk}W')
    df_overview = caraga_shp.merge(split_, how='left',
                                   left_on=list(caraga_shp.columns[:2]),
                                   right_on=[f'{wk}W_prov', f'{wk}W_city'])
    df_overview.drop(columns=[f'{wk}W_prov',f'{wk}W_city',
                              'PROVINCE','CITY_MUNICIPALITY'], inplace=True)

    # Extract aided cities/municipalities
    df_aided = df_overview.query('AID_CNT.notna()')
    # display(df_aided)

    # print(f'Locations that received crucial aid:')
    aided_provs = df_aided.copy()[['province','city_municipality','AID_CNT']]
    aided_provs['province'] = aided_provs['province'].str.upper()
    aided_provs['city_municipality'] = (aided_provs['city_municipality']
                                        .str
                                        .upper())
    aided_provs = aided_provs.sort_values(by='AID_CNT', ascending=False)
    # display(aided_provs)

    # Extract unaided cities/municipalities
    df_n_aided = df_overview.query('AID_CNT.isna()')
    n_aided_provs = (df_n_aided.copy()[['province',
                                        'city_municipality',
                                        'population']])
    n_aided_provs['province'] = n_aided_provs['province'].str.upper()
    n_aided_provs['city_municipality'] = (n_aided_provs['city_municipality']
                                          .str
                                          .upper())
    n_aided_provs = (
        n_aided_provs.sort_values(by='population', ascending=False)
    )
    # display(n_aided_provs)
    
    # measure the distance of every aided province
    # with unaided cities/municipalities
    unsrvd_res = []
    for i, row in df_aided.iterrows():
        row_geom = row['geom']

        # distance of one aided province from all
        # unaided cities/municipalities
        dists_unserved = (gpd.GeoSeries(df_n_aided['geom'])
                          .distance(row_geom.centroid)
                          .sort_values(ascending=True)
                          .to_frame().rename(columns={0:'distance'}))

        # Get nearest 5 unaided cities/municipalities
        nearest_unsrvd = dists_unserved.iloc[:5] 
        nrst_idx = nearest_unsrvd.index

        # Get the population of the nearest unaided cities/municipalities
        pop_nrst_unsrvd = (df_n_aided.loc[nrst_idx]
                           .sort_values(by='population', ascending=False)
                           .iloc[[0]])
        pop_nrst_unsrvd = (pop_nrst_unsrvd.merge(dists_unserved,
                                                 left_index=True,
                                                 right_index=True))
        unsrvd_res.append(pop_nrst_unsrvd)

    # Sort nearest unaided cities/municipalities
    # by largest to smallest population
    unsrvd_res = pd.concat(unsrvd_res)
    # Get the top 3 by nearest distance and largest population
    unsrvd_res_reco = (unsrvd_res
                       .dissolve(by=['province',
                                     'city_municipality',
                                     'population'],aggfunc='mean',
                                 as_index=False)
                       .sort_values(by=['population','distance'],
                                    ascending=[False,True]).iloc[:3,:])
    # display(unsrvd_res_reco)

    fig, ax = plt.subplots(figsize=(10,10))

    plot_caraga_shp(ax, caraga_shp)

    df_aided.plot(ax=ax,
                    facecolor='none',
                    edgecolor='green',
                    linewidth=1.5,
                    alpha=1)

    unsrvd_res_reco.plot(ax=ax, column='population',
                    facecolor='none',
                    edgecolor='blue',
                    linewidth=1,
                    hatch='//', alpha=0.7)

    plt.title(
        f'3 Recommended provinces by\nWeek {wk} for crucial aid',
        fontsize=20
    )
    plt.show()
    return top_freq_itm_df, aided_provs, unsrvd_res_reco