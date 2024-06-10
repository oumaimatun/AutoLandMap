import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from osgeo import gdal
import tempfile
import io
import matplotlib.image as mpimg
import rasterio as raster

# Function to preprocess the image and convert to test.csv
def preprocess_image(image_path):
    datasetTest = gdal.Open(image_path)
    dataTest2d = datasetTest.ReadAsArray()
    dataTest2d = np.swapaxes(dataTest2d, 0, 2)
    dataTest1d = dataTest2d.reshape(dataTest2d.shape[0] * dataTest2d.shape[1], -1)
    np.save('test_all.npy', dataTest1d)
    dfTest = pd.DataFrame(dataTest1d)
    dfTest.columns = ['Blue', 'Green', 'Red', 'NIR']
    dfTest.to_csv('test.csv', index=False)
    return dfTest

# Streamlit interface
st.title("Cartographie automatisée de la couverture terrestre")
st.write("Téléchargez une image .tif pour la classification.")

uploaded_file = st.file_uploader("Choisissez un fichier .tif", type="tif")

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Preprocess the image and convert to test.csv
    st.write("Traitement de l'image en cours...")
    test_df = preprocess_image(tmp_path)
    st.write("Image traitée et sauvegardée sous test.csv")

    # Load and preprocess data
    st.write("Chargement et prétraitement des données...")
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    # Drop the 'Unnamed: 0' column if it exists
    if 'Unnamed: 0' in train.columns:
        train.drop('Unnamed: 0', axis=1, inplace=True)
    if 'Unnamed: 0' in test.columns:
        test.drop('Unnamed: 0', axis=1, inplace=True)

    train = train[train.Code != 0]

    # Scale Data
    scaled_train = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    scaled_test = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)

    data = np.array(train.drop("Code", axis=1)[::10])
    data1d = np.array(train["Code"][::10])

    X_train, X_test, y_train, y_test = train_test_split(data, data1d, test_size=0.3)

# Add NDVI calculation and plot
    label_path = "S2A_MSIL1C_20220516_Train_GT.tif"
    data_path = "S2A_MSIL1C_20220516_TrainingData.tif"

# Open raster files using GDAL
    dataset_train = gdal.Open(data_path)
    dataset_label = gdal.Open(label_path)

# Read raster data
    featuresTrain = dataset_train.ReadAsArray()
    featuresTest = dataset_label.ReadAsArray()

# Calculate NDVI
    ndvi = (featuresTrain[3] - featuresTrain[2]) / (featuresTrain[3] + featuresTrain[2])

# Plot NDVI
    fig, ax = plt.subplots(figsize=(7, 10))
    im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title("Normalized Difference Vegetation Index (NDVI)")
    st.pyplot(fig)


    # Train k-NN
    st.write("Entraînement du modèle k-NN...")
    k = 5
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    accuracy_knn = accuracy_score(y_test, knn_pred)

    st.write(f'Précision de k-NN : {accuracy_knn*100:.2f}%')
    # st.write("Rapport de classification pour k-NN :")
    # st.text(classification_report(y_test, knn_pred))

    y_pred_KNN = knn.predict(test)

    # Visualize k-NN result
    st.write("Visualisation du résultat de k-NN...")
    fig, ax = plt.subplots()
    ax.imshow(y_pred_KNN.reshape(2309, 2001), cmap=ListedColormap(['green', 'darkolivegreen', 'limegreen', 'goldenrod', 'red', 'khaki', 'snow', 'dodgerblue']))
    ax.set_title("Résultat de la classification k-NN")
    plt.savefig("knn_result.png")
    st.image("knn_result.png")
    st.write("Cliquez sur le bouton ci-dessous pour télécharger le résultat de la prédiction k-NN en format PNG")

    # Add download button for k-NN prediction result
    with open("knn_result.png", "rb") as img_file:
        img_bytes = img_file.read()

    st.download_button(
        label="Télécharger le résultat de la prédiction k-NN en PNG",
        data=img_bytes,
        file_name="knn_prediction_result.png",
        mime="image/png"
    )

    # Add descriptive table
    labels = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice', 'Permanent water bodies']
    colors = ['green', 'darkolivegreen', 'limegreen', 'goldenrod', 'red', 'khaki', 'snow', 'dodgerblue']
    descriptions = [
        'Zones couvertes d\'arbres',
        'Régions avec des arbustes et de petits buissons',
        'Zones dominées par de l\'herbe',
        'Terres utilisées pour l\'agriculture',
        'Zones urbaines avec des bâtiments et des infrastructures',
        'Régions avec une végétation très clairsemée ou inexistante',
        'Zones couvertes de neige et de glace',
        'Plans d\'eau présents toute l\'année'
    ]

    color_boxes = [f'<div style="background-color:{color}; width: 50px; height: 20px;"></div>' for color in colors]

    df = pd.DataFrame({
        'Label': labels,
        'Color': color_boxes,
        'Description': descriptions
    })

    st.write("Labels de classification et couleurs:")
    st.write(df.to_html(escape=False), unsafe_allow_html=True)

    # Train Random Forest
    st.write("Entraînement du modèle Random Forest...")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    rf_pred = clf.predict(X_test)
    accuracy_rf = accuracy_score(y_test, rf_pred)

    st.write(f'Précision de Random Forest : {accuracy_rf*100:.2f}%')
    # st.write("Rapport de classification pour Random Forest :")
    # st.text(classification_report(y_test, rf_pred))

# Predict with Random Forest
    rf_grid_pred = clf.predict(test)

# Visualize Random Forest result
    st.write("Visualisation du résultat de Random Forest...")
    fig, ax = plt.subplots()
    ax.imshow(rf_grid_pred.reshape(2309, 2001), cmap="RdBu")
    ax.set_title("Résultat de la classification Random Forest")
    plt.savefig("random_forest_result.png")  # Save the plot
    st.image("random_forest_result.png")
    st.write("Cliquez sur le bouton ci-dessous pour télécharger le résultat de la prédiction Random Forest en format PNG")

# Add download button for Random Forest prediction result
    with open("random_forest_result.png", "rb") as img_file:
      img_bytes_rf = img_file.read()

    st.download_button(
      label="Télécharger le résultat de la prédiction Random Forest en PNG",
      data=img_bytes_rf,
      file_name="random_forest_prediction_result.png",
      mime="image/png"
)


# Liste des labels
    labelss = ['Tree cover', 'Shrubland', 'Grassland', 'Cropland', 'Built-up', 'Bare/sparse vegetation', 'Snow and ice', 'Permanent water bodies']

# Palette de couleurs RdBu
    colorss = ['#430C05', '#e98457', '#EBACA2', '#D46F4D', '#f2f3f3', '#F0BE86', '#f2f3f3', '#5784BA']

# Descriptions pour chaque label
    descriptionss = [
    'Zones couvertes d\'arbres',
    'Régions avec des arbustes et de petits buissons',
    'Zones dominées par de l\'herbe',
    'Terres utilisées pour l\'agriculture',
    'Zones urbaines avec des bâtiments et des infrastructures',
    'Régions avec une végétation très clairsemée ou inexistante',
    'Zones couvertes de neige et de glace',
    'Plans d\'eau présents toute l\'année'
]

    color_boxess = [f'<div style="background-color:{color}; width: 50px; height: 20px;"></div>' for color in colorss]
# Créer un DataFrame avec les informations
    dfs = pd.DataFrame({
    'Label': labelss,
    'Color': color_boxess,
    'Description': descriptionss
})

# Afficher le DataFrame dans Streamlit
    st.write("Labels de classification et couleurs:")
    st.write(dfs.to_html(escape=False), unsafe_allow_html=True)
