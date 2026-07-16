# House Price Prediction: Results

## Evaluation on the held-out test set

| model                                     |    MAE |   RMSE |     R2 |
|:------------------------------------------|-------:|-------:|-------:|
| Simple Linear Regression (MedInc)         | 0.6299 | 0.8421 | 0.4589 |
| Multiple Linear Regression (all features) | 0.5332 | 0.7456 | 0.5758 |

The model with the higher R² in this run is **Multiple Linear Regression (all features)**. Lower MAE/RMSE indicate smaller prediction errors; higher R² indicates more variance in house values explained by the model.

## Coefficient interpretation (multiple linear regression)

Features are median-imputed and standardized before fitting. Coefficients therefore describe a one-standard-deviation change, and values are in $100,000 units.
- **MedInc**: a one-standard-deviation increase increases the predicted house value by **0.854** $100,000 units, holding the other features constant.
- **HouseAge**: a one-standard-deviation increase increases the predicted house value by **0.123** $100,000 units, holding the other features constant.
- **AveRooms**: a one-standard-deviation increase decreases the predicted house value by **0.294** $100,000 units, holding the other features constant.
- **AveBedrms**: a one-standard-deviation increase increases the predicted house value by **0.339** $100,000 units, holding the other features constant.
- **Population**: a one-standard-deviation increase decreases the predicted house value by **0.002** $100,000 units, holding the other features constant.
- **AveOccup**: a one-standard-deviation increase decreases the predicted house value by **0.041** $100,000 units, holding the other features constant.
- **Latitude**: a one-standard-deviation increase decreases the predicted house value by **0.897** $100,000 units, holding the other features constant.
- **Longitude**: a one-standard-deviation increase decreases the predicted house value by **0.870** $100,000 units, holding the other features constant.
- **Intercept (2.072)**: predicted house value when every scaled feature is at its mean (zero after standardization).

## Intercept interpretation

Because the predictor variables are standardized, the intercept is the model's prediction for a district whose features are all at their average values. It is a baseline, not a literal house with zero bedrooms or zero population.
