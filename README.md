# Solar Filament Segmentation

Upload a full-disk H-alpha solar image and select **Detect filaments** to view the original image, the predicted mask, and a red overlay. The app uses the trained U-Net checkpoint at epoch 39, CLAHE preprocessing, and a threshold of 0.40.

## Streamlit Community Cloud

Put the files from this folder at the root of a GitHub repository. In Streamlit Community Cloud, choose the repository and its branch, set the main file to `app.py`, and select Python 3.12 in Advanced settings. Test the deployed app with a PNG or JPEG.
