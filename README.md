# Used Bike Price Analysis and Prediction

A Statistical Inference and Machine Learning project that explores used-bike listings and predicts an estimated resale price.

## Project contents

- `app.py` — Streamlit web application for used-bike price prediction.
- `si_project_b1b2_28.ipynb` — Original Jupyter notebook containing EDA, probability distributions, sampling, CLT, confidence intervals, hypothesis tests, correlation, ANOVA, regression and classification.
- `requirements.txt` — Python dependencies for running the project.

## Dataset

The app downloads the dataset when it starts from:

https://raw.githubusercontent.com/KrutinBhat/Used-Bike-Prices-Analysis/main/bikes.csv

An internet connection is required for the first dataset load.

## Run locally

1. Install Python 3.10 or later.
2. Open a terminal in this project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

5. Open the local URL printed in the terminal, usually `http://localhost:8501`.

## Deploy on Streamlit Community Cloud

1. Push this project folder to a GitHub repository.
2. Sign in at https://share.streamlit.io/ using your GitHub account.
3. Select **Create app**.
4. Choose your repository, branch (usually `main`) and main file path `app.py`.
5. Select **Deploy**. Streamlit installs packages from `requirements.txt`.
6. After deployment, copy the public app URL.

## Push to GitHub

Create an empty repository on GitHub first. Do not add a README or license there if you are pushing this folder as-is.

Run these commands from this project folder, replacing `YOUR-USERNAME` and `YOUR-REPOSITORY`:

```bash
git init
git add .
git commit -m "Add used bike price prediction project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
git push -u origin main
```

If Git asks you to sign in, complete the GitHub authentication flow in your browser.

## Model evaluation

The app displays Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R-squared (R²) on a held-out test split. These are evaluation metrics, not a guarantee of the exact market resale price.

## Notes

- The notebook is preserved as uploaded.
- The Streamlit app is extracted from the notebook's `%%writefile app.py` cell.
- The app trains a Random Forest regression pipeline when it starts and caches the trained model.
