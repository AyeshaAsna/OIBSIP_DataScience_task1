# Iris Flower Classification with Machine Learning
# This script demonstrates how to classify iris flowers into three species:
# setosa, versicolor, and virginica using various ML algorithms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_iris_data():
    """Load iris dataset from scikit-learn"""
    print("Loading Iris Dataset...")
    iris = load_iris()
    
    # Create DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    df['species'] = iris.target_names[iris.target]
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"Target classes: {iris.target_names}")
    print("\nFirst few rows:")
    print(df.head())
    
    return df, iris

def explore_data(df):
    """Explore and visualize the dataset"""
    print("\n" + "="*50)
    print("DATA EXPLORATION")
    print("="*50)
    
    # Basic statistics
    print("\nDataset Statistics:")
    print(df.describe())
    
    # Check for missing values
    print(f"\nMissing values: {df.isnull().sum().sum()}")
    
    # Class distribution
    print("\nClass Distribution:")
    print(df['species'].value_counts())
    
    # Create visualizations
    plt.figure(figsize=(15, 10))
    
    # Feature distributions by species
    plt.subplot(2, 2, 1)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.hist(subset['sepal length (cm)'], alpha=0.7, label=species, bins=20)
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Frequency')
    plt.title('Sepal Length Distribution by Species')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.hist(subset['sepal width (cm)'], alpha=0.7, label=species, bins=20)
    plt.xlabel('Sepal Width (cm)')
    plt.ylabel('Frequency')
    plt.title('Sepal Width Distribution by Species')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.hist(subset['petal length (cm)'], alpha=0.7, label=species, bins=20)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.title('Petal Length Distribution by Species')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for species in df['species'].unique():
        subset = df[df['species'] == species]
        plt.hist(subset['petal width (cm)'], alpha=0.7, label=species, bins=20)
    plt.xlabel('Petal Width (cm)')
    plt.ylabel('Frequency')
    plt.title('Petal Width Distribution by Species')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.drop(['target', 'species'], axis=1).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.show()
    
    # Pairplot
    plt.figure(figsize=(12, 10))
    sns.pairplot(df, hue='species', diag_kind='hist')
    plt.show()

def prepare_data(df, iris):
    """Prepare data for machine learning"""
    print("\n" + "="*50)
    print("DATA PREPARATION")
    print("="*50)
    
    # Separate features and target
    X = iris.data
    y = iris.target
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Testing set size: {X_test.shape[0]}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple machine learning models"""
    print("\n" + "="*50)
    print("MODEL TRAINING")
    print("="*50)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=3),
        'Support Vector Machine': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
    }
    
    # Train and evaluate models
    results = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'y_pred': y_pred
        }
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Cross-validation: {cv_mean:.4f} (+/- {cv_std * 2:.4f})")
    
    return results

def evaluate_models(results, y_test, iris):
    """Evaluate and compare model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION")
    print("="*50)
    
    # Compare accuracies
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    names = list(accuracies.keys())
    values = list(accuracies.values())
    bars = plt.bar(names, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy Comparison')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    # Cross-validation comparison
    plt.subplot(1, 2, 2)
    cv_means = [results[name]['cv_mean'] for name in names]
    cv_stds = [results[name]['cv_std'] for name in names]
    
    bars = plt.bar(names, cv_means, yerr=cv_stds, capsize=5, 
                   color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
    plt.xlabel('Models')
    plt.ylabel('Cross-validation Score')
    plt.title('Cross-validation Performance')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, cv_means):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Find best model
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = results[best_model_name]['model']
    best_accuracy = accuracies[best_model_name]
    
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    
    return best_model_name, best_model

def detailed_evaluation(best_model, best_model_name, X_test, y_test, iris):
    """Detailed evaluation of the best model"""
    print("\n" + "="*50)
    print(f"DETAILED EVALUATION: {best_model_name}")
    print("="*50)
    
    # Make predictions
    y_pred = best_model.predict(X_test)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.show()
    
    # Feature importance (for tree-based models)
    if hasattr(best_model, 'feature_importances_'):
        feature_importance = best_model.feature_importances_
        feature_names = iris.feature_names
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(feature_names, feature_importance, color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title(f'Feature Importance - {best_model_name}')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, importance in zip(bars, feature_importance):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{importance:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning for the best model"""
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING")
    print("="*50)
    
    # Tune Random Forest (usually performs well on this dataset)
    rf = RandomForestClassifier(random_state=42)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    print("Tuning Random Forest hyperparameters...")
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def make_predictions(best_model, scaler, iris):
    """Make predictions on new data"""
    print("\n" + "="*50)
    print("MAKING PREDICTIONS")
    print("="*50)
    
    # Example new measurements (you can modify these values)
    new_measurements = [
        [5.1, 3.5, 1.4, 0.2],  # Example setosa
        [6.3, 3.3, 4.7, 1.6],  # Example versicolor
        [6.5, 3.0, 5.2, 2.0]   # Example virginica
    ]
    
    # Scale the new measurements
    new_measurements_scaled = scaler.transform(new_measurements)
    
    # Make predictions
    predictions = best_model.predict(new_measurements_scaled)
    prediction_proba = best_model.predict_proba(new_measurements_scaled)
    
    print("Predictions on new data:")
    for i, (measurement, pred, proba) in enumerate(zip(new_measurements, predictions, prediction_proba)):
        species = iris.target_names[pred]
        print(f"\nSample {i+1}:")
        print(f"Measurements: Sepal Length={measurement[0]}, Sepal Width={measurement[1]}, "
              f"Petal Length={measurement[2]}, Petal Width={measurement[3]}")
        print(f"Predicted Species: {species}")
        print(f"Confidence: {max(proba):.3f}")
        
        # Show all probabilities
        for j, (spec, prob) in enumerate(zip(iris.target_names, proba)):
            print(f"  {spec}: {prob:.3f}")

def main():
    """Main function to run the complete iris classification pipeline"""
    print("IRIS FLOWER CLASSIFICATION WITH MACHINE LEARNING")
    print("="*60)
    
    try:
        # Load data
        df, iris = load_iris_data()
        
        # Explore data
        explore_data(df)
        
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = prepare_data(df, iris)
        
        # Train models
        results = train_models(X_train, X_test, y_train, y_test)
        
        # Evaluate models
        best_model_name, best_model = evaluate_models(results, y_test, iris)
        
        # Detailed evaluation of best model
        detailed_evaluation(best_model, best_model_name, X_test, y_test, iris)
        
        # Hyperparameter tuning
        tuned_model = hyperparameter_tuning(X_train, y_train)
        
        # Make predictions
        make_predictions(tuned_model, scaler, iris)
        
        print("\n" + "="*60)
        print("IRIS CLASSIFICATION PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
