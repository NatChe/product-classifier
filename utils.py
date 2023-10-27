import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import metrics
from sklearn import manifold, decomposition
from sklearn import cluster, metrics

from yellowbrick.cluster.elbow import kelbow_visualizer
from yellowbrick.cluster import silhouette_visualizer

def extract_category(category_tree, level):
    """Return a category depending on the level.
    
    Input:
        category_tree: a string containing the category tree
        level: a int with a level corresponding to the category starting from 0
    Output:
        category: string
    """
    
    # remove surrounding characters
    category_tree = category_tree.strip('["]')
    
    # split into array
    categories = category_tree.split(' >> ')
    
    # return last category if level is bigger than number of categories
    if (level > len(categories) - 1):
        return categories[-1]
    
    
    return categories[level]


def display_images(X, y=None, columns=12, x_size=1, y_size=1, colorbar=False, y_pred=None, cmap='gray', norm=None, spines_alpha=1, interpolation='lanczos'):
    
    """
    Show some images in a grid, with legends
    args:
        X             : images array
        y             : real classes or labels or None (None)
        columns       : number of columns (12)
        x_size,y_size : figure size (1), (1) 
        colorbar      : show colorbar (False)
        y_pred        : predicted classes (None)
        cmap          : Matplotlib color map (binary)
        norm          : Matplotlib imshow normalization (None)
        spines_alpha  : Spines alpha (1.)
    returns: 
        void
    """
    
    indices = range(len(X))
        
    if norm and len(norm) == 2: norm = matplotlib.colors.Normalize(vmin=norm[0], vmax=norm[1])
        
    draw_labels = (y is not None)
    draw_pred   = (y_pred is not None)
    rows        = math.ceil(len(indices)/columns)
    padding     = 1
    fontsize    = 10
    
    fig=plt.figure(figsize=(columns * (x_size + padding), rows * (y_size + padding)))
    n=1
    for i in indices:
        axs=fig.add_subplot(rows, columns, n)
        n+=1

        img=axs.imshow(X[i], cmap = cmap)

        axs.spines['right'].set_visible(True)
        axs.spines['left'].set_visible(True)
        axs.spines['top'].set_visible(True)
        axs.spines['bottom'].set_visible(True)
        axs.spines['right'].set_alpha(spines_alpha)
        axs.spines['left'].set_alpha(spines_alpha)
        axs.spines['top'].set_alpha(spines_alpha)
        axs.spines['bottom'].set_alpha(spines_alpha)
        axs.set_yticks([])
        axs.set_xticks([])
        if draw_labels and not draw_pred:
            axs.set_xlabel(y[i],fontsize=fontsize)
        if draw_labels and draw_pred:
            if y[i]!=y_pred[i]:
                axs.set_xlabel(f'{y_pred[i]} ({y[i]})',fontsize=fontsize)
                axs.xaxis.label.set_color('red')
            else:
                axs.set_xlabel(y[i],fontsize=fontsize)
        if colorbar:
            fig.colorbar(img,orientation="vertical", shrink=0.65)
    
    plt.show()


def get_pca_reduced_data(data, n_components=0.99):
    """
    Applies PCA dimensiality reduction to the provided data

    Input:
        data: numpy array

    Output:
        numpy array with reduced dimensions
    """
    
    print("Dimensions before PCA : ", data.shape)

    pca = decomposition.PCA(n_components=n_components)
    data_pca = pca.fit_transform(data)

    print("Dimensions after PCA : ", data_pca.shape)

    return data_pca


def get_tsne_reduced_dataframe(data, n_dimensions=2):
    """
    Reduces given data using T-SNE method

    Input: 
        data: array to reduce dimensions for
        n_dimensions: number of dimensions to produce
    Output:
        dataframe with reduced dimensions
    """

    tsne = manifold.TSNE(n_components=n_dimensions, perplexity=30,
                         n_iter=2000, init='random', random_state=0)
    X_tsne = tsne.fit_transform(data)

    return pd.DataFrame(X_tsne[:,0:2], columns=['tsne1', 'tsne2'])


def display_tsne_categories(df, labels, method_name=''):
    """
    Displays a scatterplot with the highlighted categories
    
    Input:
        df: dataframe with reduced dimensions, should have tsne1 and tsne2 features
        labels: Series of labels
        method_name: string, optional, method name used to be displayed on the plot
    """
    
    # Add category feature 
    df["category"] = labels
    
    # Create a scatterplot
    plt.figure(figsize=(8,5))

    sns.scatterplot(
        x="tsne1", y="tsne2", hue="category", data=df, legend="brief",
        palette=sns.color_palette('Set2', n_colors=7), s=50, alpha=0.6)

    plt.title(f'TSNE with the categories [{method_name}]', fontsize = 18, pad = 24, fontweight = 'bold')
    plt.xlabel('tsne1', fontsize = 16, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 16, fontweight = 'bold')
    plt.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0), loc='upper left') 

    plt.show()


def perform_clustering(df_tsne, true_labels, n_clusters=7, class_names=[], method_name=''):
    """
    Performs and displays K-Means clustering of df
    
    Input:
        df: dataframe with reduced dimensions, should have tsne1 and tsne2 features
        true_labels: array of labels
        n_clusters: int, default 7, number of clusters
        method_name: string, optional, method name used to be displayed on the plot
    """
    
    # Perform the KMeans clustering
    cls = cluster.KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    cls.fit(df_tsne[['tsne1', 'tsne2']].values)

    pred_labels = cls.labels_
    
    # Get the ARI score
    print("ARI : ", round(metrics.adjusted_rand_score(true_labels, pred_labels), 4))
    
    # Display the categories
    df_tsne['cluster'] = pred_labels
    
    plt.figure(figsize=(10,6))

    sns.scatterplot(
        x="tsne1", y="tsne2",
        hue="cluster",
        palette=sns.color_palette('Set2', n_colors=7), s=50, alpha=0.6,
        data=df_tsne,
        legend="brief")

    plt.title(f'TSNE by clusters [{method_name}]', fontsize = 18, pad = 24, fontweight = 'bold')
    plt.xlabel('tsne1', fontsize = 16, fontweight = 'bold')
    plt.ylabel('tsne2', fontsize = 16, fontweight = 'bold')
    plt.legend(prop={'size': 12}, bbox_to_anchor=(1.05, 1.0), loc='upper left') 

    plt.show()


def get_cluster_scores(df, max_cluster=11):
    """
    Computes KMeans inertia and silhouette score for multiple cluster parameters
    
    Input:
        df: dataframe to perform the clustering, should have tsne1 and tsne2 features
        max_cluster: int, default 11
    """
    
    n_cluster = range(2, max_cluster, 1)
    inertia = []
    silhouette = []
    X = df[['tsne1', 'tsne2']].values

    for i in n_cluster:
        model = cluster.KMeans(n_clusters=i, n_init=10, random_state=0)
        y_labels = model.fit_predict(X)
        inertia.append(model.inertia_)

        score = metrics.silhouette_score(X, y_labels)
        silhouette.append(score)
        print(f'cluster {i}: {score} score')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.plot(n_cluster, inertia, marker="o")
    ax1.set_xlabel("Number of Segments")
    ax1.set_ylabel("Inertia Value")

    ax2.plot(n_cluster, silhouette, marker="o", c="green")
    ax2.set_xlabel("Number of Segments")
    ax2.set_ylabel("Silhouette score")

    fig.tight_layout()
    plt.show()
    
    kelbow_visualizer(cluster.KMeans(random_state=0, n_init=10), X, k=(2, max_cluster), timings=False)
