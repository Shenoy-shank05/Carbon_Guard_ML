from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
import joblib
import shap
from catboost import CatBoostRegressor

app = Flask(__name__)
CORS(app)

# Load CatBoost model
model_path = 'catboost_model_new.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("CatBoost model loaded successfully")
else:
    print(f"Model file not found at {model_path}")
    model = None

# Initialize SHAP explainer
explainer = None
if model is not None:
    try:
        # For CatBoost, we can use TreeExplainer
        explainer = shap.TreeExplainer(model)
        print("SHAP explainer initialized successfully")
    except Exception as e:
        print(f"Error initializing SHAP explainer: {str(e)}")

def preprocess_input_for_catboost(data):
    """
    Minimal preprocessing for CatBoost with proper data type handling
    """
    print(f"Preprocessing data for CatBoost: {data}")
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Define expected columns and their types based on your training data
    numerical_columns = [
        'Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 'Waste Bag Weekly Count',
        'How Long TV PC Daily Hour', 'How Many New Clothes Monthly', 'How Long Internet Daily Hour'
    ]
    
    categorical_columns = [
        'Body Type', 'Sex', 'Diet', 'How Often Shower', 'Heating Energy Source',
        'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air',
        'Waste Bag Size', 'Energy efficiency', 'Recycling', 'Cooking_With'
    ]
    
    all_expected_columns = numerical_columns + categorical_columns
    
    # Fill missing columns with default values
    for col in all_expected_columns:
        if col not in input_df.columns:
            if col in numerical_columns:
                input_df[col] = 0.0  # Use float for numerical columns
            else:
                input_df[col] = 'None'  # Use string for categorical columns
    
    # Handle list-type columns (Recycling, Cooking_With)
    for col in ['Recycling', 'Cooking_With']:
        if col in input_df.columns:
            if isinstance(input_df[col].iloc[0], list):
                input_df[col] = input_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
            else:
                input_df[col] = str(input_df[col].iloc[0])
    
    # Ensure proper data types
    for col in numerical_columns:
        if col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce').fillna(0.0).astype(float)
            except:
                input_df[col] = 0.0
    
    for col in categorical_columns:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    # Ensure column order matches training data
    input_df = input_df.reindex(columns=all_expected_columns)
    
    # Final data type verification
    for col in numerical_columns:
        input_df[col] = input_df[col].astype(float)
    
    for col in categorical_columns:
        input_df[col] = input_df[col].astype(str)
    
    print(f"Preprocessed DataFrame for CatBoost:")
    print(f"Shape: {input_df.shape}")
    print(f"Data types:\n{input_df.dtypes}")
    print(f"Sample data:\n{input_df.head()}")
    
    return input_df

def get_shap_feature_importance(input_df, top_n=7):
    """
    Calculate SHAP values and return major contributing features with percentages
    """
    try:
        if explainer is None:
            print("SHAP explainer not available")
            return []
        
        print(f"Input DataFrame for SHAP:")
        print(f"Shape: {input_df.shape}")
        print(f"Data types: {input_df.dtypes.to_dict()}")
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(input_df)
        
        # Handle different SHAP output formats
        if isinstance(shap_values, list):
            # For multi-class or multi-output models
            shap_vals = shap_values[0][0] if len(shap_values[0].shape) > 1 else shap_values[0]
        else:
            # For single output models
            shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
        
        # Get feature names
        feature_names = input_df.columns.tolist()
        
        # Ensure we have the right number of SHAP values
        if len(shap_vals) != len(feature_names):
            print(f"Mismatch: {len(shap_vals)} SHAP values vs {len(feature_names)} features")
            return []
        
        # Create feature importance dictionary
        feature_importance = {}
        for i, feature in enumerate(feature_names):
            feature_importance[feature] = abs(float(shap_vals[i]))
        
        # Calculate total importance for percentage calculation
        total_importance = sum(feature_importance.values())
        
        if total_importance == 0:
            print("Total importance is zero, returning empty list")
            return []
        
        # Sort features by importance
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        major_features = []
        others_importance = 0
        
        for i, (feature, importance) in enumerate(sorted_features):
            percentage = (importance / total_importance) * 100
            
            if i < top_n and percentage > 1:  # Only include features with >1% contribution
                major_features.append({
                    'feature': feature,
                    'contribution': float(shap_vals[feature_names.index(feature)]),
                    'importance': float(importance),
                    'percentage': round(percentage, 2)
                })
            else:
                others_importance += importance
        
        # Add "Others" category if there are remaining features
        if others_importance > 0:
            others_percentage = (others_importance / total_importance) * 100
            if others_percentage > 1:
                major_features.append({
                    'feature': 'Others',
                    'contribution': 0,
                    'importance': float(others_importance),
                    'percentage': round(others_percentage, 2)
                })
        
        print(f"SHAP-based major contributing features: {major_features}")
        return major_features
        
    except Exception as e:
        print(f"Error calculating SHAP values: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

def generate_targeted_recommendations(data, major_features):
    """
    Generate recommendations based on SHAP feature importance
    """
    print(f"Generating targeted recommendations based on SHAP features: {major_features}")
    
    recommendations = []
    
    # Feature-specific recommendation mapping
    feature_recommendations = {
        'Vehicle Monthly Distance Km': {
            'category': 'Transport',
            'title': 'Reduce vehicle distance',
            'description': 'Your vehicle usage is a major contributor. Consider carpooling, public transport, or working from home to reduce monthly driving distance.',
            'impact': 'high'
        },
        'Monthly Grocery Bill': {
            'category': 'Food',
            'title': 'Optimize food consumption',
            'description': 'Your grocery spending significantly impacts your footprint. Focus on local, seasonal produce and reduce food waste.',
            'impact': 'high'
        },
        'How Many New Clothes Monthly': {
            'category': 'Consumption',
            'title': 'Reduce clothing purchases',
            'description': 'Clothing consumption is a major factor. Try second-hand shopping, clothing swaps, or extending garment lifespan.',
            'impact': 'medium'
        },
        'Waste Bag Weekly Count': {
            'category': 'Waste',
            'title': 'Minimize waste generation',
            'description': 'Your waste production significantly contributes to your footprint. Focus on reducing, reusing, and composting.',
            'impact': 'high'
        },
        'How Long TV PC Daily Hour': {
            'category': 'Energy',
            'title': 'Reduce screen time energy use',
            'description': 'Your device usage contributes notably. Use energy-efficient devices and enable power-saving modes.',
            'impact': 'medium'
        },
        'How Long Internet Daily Hour': {
            'category': 'Digital',
            'title': 'Optimize digital consumption',
            'description': 'Your internet usage impacts your footprint. Reduce streaming quality and use energy-efficient devices.',
            'impact': 'low'
        }
    }
    
    # Transport-related recommendations
    transport_recommendations = {
        'private': {
            'category': 'Transport',
            'title': 'Switch to sustainable transport',
            'description': 'Private vehicle use is impactful. Consider electric vehicles, public transport, or active transportation.',
            'impact': 'high'
        },
        'public': {
            'category': 'Transport',
            'title': 'Optimize public transport use',
            'description': 'Great choice using public transport! Consider combining with cycling or walking for shorter trips.',
            'impact': 'medium'
        }
    }
    
    # Diet-related recommendations
    diet_recommendations = {
        'omnivore': {
            'category': 'Food',
            'title': 'Reduce meat consumption',
            'description': 'Your diet significantly impacts your footprint. Try plant-based meals 2-3 times per week.',
            'impact': 'high'
        },
        'vegetarian': {
            'category': 'Food',
            'title': 'Optimize vegetarian choices',
            'description': 'Good dietary choice! Focus on local, organic produce and minimize dairy consumption.',
            'impact': 'medium'
        },
        'vegan': {
            'category': 'Food',
            'title': 'Maintain sustainable diet',
            'description': 'Excellent dietary choice! Focus on local, seasonal produce to further reduce impact.',
            'impact': 'low'
        }
    }
    
    # Energy-related recommendations
    energy_recommendations = {
        'coal': {
            'category': 'Energy',
            'title': 'Switch from coal heating',
            'description': 'Coal heating has high emissions. Consider switching to natural gas, electricity, or renewable sources.',
            'impact': 'high'
        },
        'natural gas': {
            'category': 'Energy',
            'title': 'Improve heating efficiency',
            'description': 'Natural gas is better than coal. Improve insulation and consider heat pumps for efficiency.',
            'impact': 'medium'
        },
        'electricity': {
            'category': 'Energy',
            'title': 'Use renewable electricity',
            'description': 'Electric heating can be clean. Switch to renewable energy providers if available.',
            'impact': 'medium'
        }
    }
    
    # Generate recommendations based on major contributing features
    for feature_info in major_features[:5]:  # Focus on top 5 features
        feature_name = feature_info['feature']
        percentage = feature_info['percentage']
        
        # Skip "Others" category
        if feature_name == 'Others':
            continue
            
        # Add feature-specific recommendations
        if feature_name in feature_recommendations:
            rec = feature_recommendations[feature_name].copy()
            rec['description'] = f"This factor contributes {percentage}% to your footprint. " + rec['description']
            recommendations.append(rec)
    
    # Add category-specific recommendations based on data values
    transport_type = data.get('Transport', '').lower()
    if transport_type in transport_recommendations:
        recommendations.append(transport_recommendations[transport_type])
    
    diet_type = data.get('Diet', '').lower()
    if diet_type in diet_recommendations:
        recommendations.append(diet_recommendations[diet_type])
    
    heating_type = data.get('Heating Energy Source', '').lower()
    if heating_type in energy_recommendations:
        recommendations.append(energy_recommendations[heating_type])
    
    # Add energy efficiency recommendations
    if data.get('Energy efficiency') == 'No':
        recommendations.append({
            'category': 'Energy',
            'title': 'Improve energy efficiency',
            'description': 'Upgrade to energy-efficient appliances and LED lighting to reduce your energy footprint.',
            'impact': 'medium'
        })
    
    # Add recycling recommendations
    recycling = data.get('Recycling', [])
    if not recycling or 'None' in str(recycling):
        recommendations.append({
            'category': 'Waste',
            'title': 'Start comprehensive recycling',
            'description': 'Implement recycling for paper, plastic, glass, and metal to significantly reduce waste impact.',
            'impact': 'medium'
        })
    
    # Remove duplicates and limit to top 5
    unique_recommendations = []
    seen_titles = set()
    for rec in recommendations:
        if rec['title'] not in seen_titles:
            unique_recommendations.append(rec)
            seen_titles.add(rec['title'])
    
    print(f"Generated {len(unique_recommendations)} unique recommendations")
    return unique_recommendations[:5]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json
        print(f"Received data for prediction: {data}")
        
        # Preprocess input for CatBoost
        input_df = preprocess_input_for_catboost(data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = abs(float(prediction))  # Ensure positive value
        
        print(f"CatBoost prediction result: {prediction}")
        
        # Get SHAP-based feature importance
        major_features = get_shap_feature_importance(input_df)
        
        # Generate targeted recommendations
        recommendations = generate_targeted_recommendations(data, major_features)
        
        # Return comprehensive response
        return jsonify({
            'prediction': prediction,
            'major_contributing_features': major_features,
            'recommendations': recommendations
        })
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def get_insights():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
            
        data = request.json.get('carbonData', {})
        print(f"Received data for insights: {data}")
        
        # Preprocess input for CatBoost
        input_df = preprocess_input_for_catboost(data)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        prediction = abs(float(prediction))  # Ensure positive value
        
        print(f"Carbon emission prediction: {prediction}")
        
        # Get SHAP-based feature importance
        major_features = get_shap_feature_importance(input_df)
        
        # Generate targeted recommendations based on SHAP values
        recommendations = generate_targeted_recommendations(data, major_features)
        
        # Build comprehensive insights response
        insights = {
            'major_contributing_features': major_features,
            'recommendations': recommendations
        }
        
        return jsonify({
            'carbonEmission': prediction,
            'insights': insights
        })
        
    except Exception as e:
        print(f"Error in /insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'model_loaded': model is not None,
        'shap_available': explainer is not None
    }
    return jsonify(status)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
