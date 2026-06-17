# FIFA Prediction Model - Local Setup & Testing Guide

## ✅ Prerequisites

Ensure you have Python 3.8+ installed on your device:
```bash
python --version
```

## 📦 Step 1: Install Dependencies

Run this command in the project directory:

```bash
pip install -r requirements.txt
```

**Required packages:**
- streamlit==1.41.0
- pandas==2.2.2
- numpy==1.26.4
- scikit-learn==1.5.0
- joblib==1.4.2
- catboost==1.2.5
- pillow==11.0.0

## 🤖 Step 2: Train the Model (Optional - Pre-trained models included)

The model is already trained and included in `model/artifacts/`. 

If you want to retrain with updated data:
```bash
python -m model.train
```

**Expected training time:** 5-10 minutes

## 🚀 Step 3: Run the Streamlit Application

```bash
streamlit run app.py
```

The app will automatically open in your browser at:
- **Local:** http://localhost:8503
- **Network:** http://192.168.0.178:8503

## 📊 What You Can Do with the App

1. **Match Predictions** - Predict international football match outcomes
2. **Player Analysis** - View player statistics and performance
3. **Tournament Data** - Explore World Cup 2026 information
4. **Squad Data** - Check team rosters and rankings

## 📁 Project Structure

```
fifa-prediction-main/
├── app.py                          # Main Streamlit application
├── model/
│   ├── train.py                   # Model training script
│   ├── predict.py                 # Prediction logic
│   ├── goals.py                   # Goals prediction model
│   └── artifacts/                 # Pre-trained models
├── data/
│   ├── processed/                 # Processed historical data
│   └── external/                  # External data sources
├── scripts/                        # Utility scripts
├── tests/                         # Test files
└── requirements.txt               # Dependencies
```

## 🔧 Troubleshooting

### Port Already in Use
If port 8503 is already in use, run:
```bash
streamlit run app.py --server.port=8504
```

### Model Not Found
Ensure `model/artifacts/` contains these files:
- `model.pkl`
- `feature_columns.json`
- `meta.json`

### Memory Issues
For large dataset operations, increase available memory or run on a device with at least 8GB RAM.

## 📝 Data Files

The application loads data from:
- `matches.csv` - Historical match results
- `data/processed/matches_all.csv` - Full processed match history
- `data/external/world_cup_2026/` - World Cup 2026 specific data

## ✨ Features

- Real-time match predictions
- Player statistics integration
- World Cup 2026 analysis
- Interactive visualizations
- Historical data analysis

## 📞 Support

For issues or questions, check the README.md file for additional documentation.

---

**Setup Complete!** You're now ready to test the FIFA Prediction model locally.
