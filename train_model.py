{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "e6b7c397",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model trained successfully!\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "import joblib\n",
    "\n",
    "# Load dataset\n",
    "df = pd.read_csv(\"./CO2 Emissions_Canada.csv\")\n",
    "\n",
    "# 🔹 Convert ALL text columns to numeric\n",
    "df = pd.get_dummies(df)\n",
    "\n",
    "# 🔹 Target column (CHANGE if name slightly different)\n",
    "target = \"CO2 Emissions(g/km)\"\n",
    "\n",
    "X = df.drop(target, axis=1)\n",
    "y = df[target]\n",
    "\n",
    "# Split data\n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "# Train model\n",
    "model = LinearRegression()\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save model\n",
    "joblib.dump(model, \"model.pkl\")\n",
    "\n",
    "print(\"✅ Model trained successfully!\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
