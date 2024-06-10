# AutoLandMap

The project presented concerns “Automated land cover mapping” using Machine Learning and image processing techniques. Its aim is to classify land areas into categories such as tree cover, agricultural land or urban areas from satellite images.

The process begins by downloading an image in .tif format via a Streamlit interface. This image is pre-processed with the GDAL library to extract the necessary data, then converted into a .csv file for analysis.

Training and test data are prepared for compatibility with Machine Learning models. Two algorithms, k-NN and Random Forest, are then used to classify image areas. Their performance is evaluated in terms of accuracy.

Visualizations are generated to show the classification results, with colored maps representing the different land cover categories. Finally, a descriptive table details these categories and their colors in the visualizations.
