from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib

app = Flask(__name__)
CORS(app)

# Load model and preprocessing tools
model_path = 'lasso_best_model.pkl'
encoder_path = 'encoder_final.pkl'
scaler_path = 'scaler_final.pkl'

if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = joblib.load(f)
    print("Model loaded successfully")
else:
    print(f"Model file not found at {model_path}")

if os.path.exists(encoder_path):
    with open(encoder_path, 'rb') as f:
        encoder = pickle.load(f)
    print("Encoder loaded successfully")
else:
    print(f"Encoder file not found at {encoder_path}")

if os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Scaler loaded successfully")
else:
    print(f"Scaler file not found at {scaler_path}")

# Function to preprocess input data
def preprocess_input(data):
    print(f"Preprocessing data: {data}")  # Debugging the incoming data
    
    # Convert input data to DataFrame
    input_df = pd.DataFrame([data])
    print(f"Initial input DataFrame:\n{input_df}")  # Debugging the DataFrame

    # Identify categorical and numerical columns (based on training)
    categorical_cols = ['Body Type','Sex', 'Diet', 'How Often Shower', 'Heating Energy Source', 'Transport', 'Vehicle Type', 'Social Activity', 'Frequency of Traveling by Air', 'Waste Bag Size', 'Energy efficiency', 'Recycling', 'Cooking_With']
    numerical_cols = ['Monthly Grocery Bill', 'Vehicle Monthly Distance Km', 'Waste Bag Weekly Count', 'How Long TV PC Daily Hour', 'How Many New Clothes Monthly', 'How Long Internet Daily Hour']

    # Fill missing numerical columns with 0
    for col in numerical_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    # Fill missing categorical columns with 'None'
    for col in categorical_cols:
        if col not in input_df.columns:
            input_df[col] = 'None'

    # Handle list-type columns (Recycling, Cooking_With)
    for col in ['Recycling', 'Cooking_With']:
        if isinstance(input_df[col].iloc[0], list):
            input_df[col] = input_df[col].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    # Ensure column order
    input_df = input_df.reindex(columns=numerical_cols + categorical_cols)
    print(f"Data after reordering and filling missing values:\n{input_df}")  # Debugging the updated DataFrame

    # Standardize numerical features
    X_num = scaler.transform(input_df[numerical_cols])
    X_num_df = pd.DataFrame(X_num, columns=numerical_cols)

    # One-hot encode categorical features
    X_cat = encoder.transform(input_df[categorical_cols])
    X_cat_df = pd.DataFrame(X_cat, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine
    X_final = pd.concat([X_num_df, X_cat_df], axis=1)
    print(f"Encoded Features:\n{X_final}")  # Debugging the final encoded features
    return X_final

# Function to generate insights based on prediction
def generate_insights(data, prediction, major_features):
    print(f"Generating insights for data: {data} with prediction: {prediction}")  # Debugging the insights generation
    
    insights = {
        'prediction': prediction,
        'breakdown': calculate_breakdown(data),
        'recommendations': generate_recommendations(data, major_features)
    }
    print(f"Generated insights: {insights}")  # Debugging the insights
    return insights

# Function to calculate breakdown
def calculate_breakdown(data):
    print(f"Calculating emission breakdown for data: {data}")  # Debugging the breakdown calculation
    
    transport_emission = 5.2 if data.get('Transport') == 'Car' else (
                         3.1 if data.get('Transport') == 'Public Transport' else (
                         4.0 if data.get('Transport') == 'Motorcycle' else 2.0))

    energy_emission = 4.8 if data.get('Heating Energy Source') == 'Coal' else (
                    3.8 if data.get('Heating Energy Source') == 'Natural Gas' else (
                    3.2 if data.get('Heating Energy Source') == 'Electricity' else 3.5))

    food_emission = 1.2 if data.get('Diet') == 'Vegan' else (
                  1.5 if data.get('Diet') == 'Vegetarian' else (
                  2.1 if data.get('Diet') == 'Omnivore' else (
                  3.5 if data.get('Diet') == 'Heavy Meat Eater' else 2.1)))

    consumption_emission = 2.2 if data.get('How Many New Clothes Monthly', 0) > 5 else (
                         1.6 if data.get('How Many New Clothes Monthly', 0) > 2 else 1.4)

    waste_emission = 1.2 if data.get('Waste Bag Weekly Count', 0) > 3 else (
                   0.8 if data.get('Waste Bag Weekly Count', 0) > 1 else 0.6)

    breakdown = [
        {'name': 'Transport', 'value': transport_emission, 'average': 4.8},
        {'name': 'Home Energy', 'value': energy_emission, 'average': 4.2},
        {'name': 'Food', 'value': food_emission, 'average': 2.8},
        {'name': 'Consumption', 'value': consumption_emission, 'average': 1.9},
        {'name': 'Waste', 'value': waste_emission, 'average': 1.0}
    ]
    print(f"Emission breakdown: {breakdown}")  # Debugging the emission breakdown
    return breakdown

# Function to generate recommendations based on major contributing features
def generate_recommendations(data, major_features=None):
    print(f"Generating recommendations for data: {data}")  # Debugging the recommendations generation
    print(f"Major features for recommendations: {major_features}")  # Debugging the major features
    
    recommendations = []
    
    # If we have major contributing features, prioritize recommendations based on them
    if major_features and len(major_features) > 0:
        # Create a mapping of feature names to recommendation functions
        feature_to_recommendation = {
            'Vehicle Monthly Distance Km': generate_vehicle_distance_recommendations,
            'How Many New Clothes Monthly': generate_clothes_recommendations,
            'Waste Bag Weekly Count': generate_waste_recommendations,
            'Monthly Grocery Bill': generate_grocery_recommendations,
            'How Long Internet Daily Hour': generate_internet_recommendations,
            'How Long TV PC Daily Hour': generate_screen_time_recommendations,
            'Transport': generate_transport_recommendations,
            'Diet': generate_diet_recommendations,
            'Heating Energy Source': generate_heating_recommendations,
            'Energy efficiency': generate_energy_efficiency_recommendations,
            'Recycling': generate_recycling_recommendations
        }
        
        # Generate recommendations for each major feature
        for feature_info in major_features:
            feature_name = feature_info['feature']
            
            # Extract the base feature name from encoded features
            if '_' in feature_name and not feature_name in feature_to_recommendation:
                # Handle one-hot encoded features by extracting the base column name
                base_feature = feature_name.split('_')[0]
                if base_feature in feature_to_recommendation:
                    feature_name = base_feature
            
            # Get recommendations for this feature if we have a specific function for it
            if feature_name in feature_to_recommendation:
                feature_recommendations = feature_to_recommendation[feature_name](data)
                recommendations.extend(feature_recommendations)
    
    # If we don't have enough recommendations from major features, add generic ones
    if len(recommendations) < 3:
        # Add generic recommendations based on the data
        if data.get('Transport') == 'Car':
            recommendations.append({
                'category': 'Transport',
                'title': 'Reduce car usage',
                'description': 'Try using public transportation, biking, or walking for short trips.',
                'impact': 'high'
            })
        elif data.get('Vehicle Type') in ['Gasoline', 'Diesel']:
            recommendations.append({
                'category': 'Transport',
                'title': 'Consider a more fuel-efficient vehicle',
                'description': 'Consider an electric or hybrid vehicle.',
                'impact': 'high'
            })

        if data.get('Heating Energy Source') in ['Coal', 'Oil']:
            recommendations.append({
                'category': 'Home Energy',
                'title': 'Switch to cleaner energy',
                'description': 'Switch to natural gas or electricity for heating.',
                'impact': 'high'
            })

        if data.get('Energy efficiency') == 'No':
            recommendations.append({
                'category': 'Home Energy',
                'title': 'Use energy-efficient appliances',
                'description': 'Replace old appliances and use LED lighting.',
                'impact': 'medium'
            })

        if data.get('Diet') in ['Heavy Meat Eater', 'Omnivore']:
            recommendations.append({
                'category': 'Food',
                'title': 'Reduce meat consumption',
                'description': 'Incorporate more plant-based meals.',
                'impact': 'high'
            })

        if data.get('How Many New Clothes Monthly', 0) > 3:
            recommendations.append({
                'category': 'Consumption',
                'title': 'Buy fewer new clothes',
                'description': 'Try second-hand shopping.',
                'impact': 'medium'
            })

        if 'None' in data.get('Recycling', []) or not data.get('Recycling'):
            recommendations.append({
                'category': 'Waste',
                'title': 'Start recycling',
                'description': 'Separate recyclables from trash.',
                'impact': 'medium'
            })
        elif len(data.get('Recycling', [])) < 3:
            recommendations.append({
                'category': 'Waste',
                'title': 'Improve recycling habits',
                'description': 'Recycle more materials like glass and electronics.',
                'impact': 'low'
            })

    # If we still don't have enough recommendations, add some general ones
    if len(recommendations) < 3:
        recommendations.extend([{
            'category': 'Home Energy',
            'title': 'Reduce standby power consumption',
            'description': 'Unplug electronics when not in use.',
            'impact': 'low'
        }, {
            'category': 'Consumption',
            'title': 'Choose products with less packaging',
            'description': 'Prefer minimal or recyclable packaging.',
            'impact': 'low'
        }])

    # Remove duplicates (based on title)
    unique_recommendations = []
    seen_titles = set()
    for rec in recommendations:
        if rec['title'] not in seen_titles:
            unique_recommendations.append(rec)
            seen_titles.add(rec['title'])

    print(f"Generated recommendations: {unique_recommendations[:5]}")  # Debugging the generated recommendations
    return unique_recommendations[:5]  # Return top 5 recommendations

# Feature-specific recommendation generators
def generate_vehicle_distance_recommendations(data):
    distance = data.get('Vehicle Monthly Distance Km', 0)
    recommendations = []
    
    if distance > 500:
        recommendations.append({
            'category': 'Transport',
            'title': 'Reduce driving distance',
            'description': 'Consider carpooling, combining trips, or using public transport to reduce your monthly driving distance.',
            'impact': 'high'
        })
    
    if distance > 200:
        recommendations.append({
            'category': 'Transport',
            'title': 'Work from home when possible',
            'description': 'If your job allows, try working from home a few days a week to reduce commuting distance.',
            'impact': 'medium'
        })
        
    if data.get('Vehicle Type') in ['Gasoline', 'Diesel']:
        recommendations.append({
            'category': 'Transport',
            'title': 'Switch to an electric or hybrid vehicle',
            'description': 'Your driving distance would have a much lower impact with a more efficient vehicle.',
            'impact': 'high'
        })
    
    return recommendations

def generate_clothes_recommendations(data):
    clothes_count = data.get('How Many New Clothes Monthly', 0)
    recommendations = []
    
    if clothes_count > 3:
        recommendations.append({
            'category': 'Consumption',
            'title': 'Reduce new clothing purchases',
            'description': 'Try to limit new clothing purchases and consider second-hand options.',
            'impact': 'medium'
        })
        
    if clothes_count > 1:
        recommendations.append({
            'category': 'Consumption',
            'title': 'Choose sustainable clothing brands',
            'description': 'When buying new clothes, look for sustainable and eco-friendly brands that use organic materials.',
            'impact': 'medium'
        })
        
    recommendations.append({
        'category': 'Consumption',
        'title': 'Extend clothing lifespan',
        'description': 'Take good care of your clothes, repair them when needed, and repurpose old items.',
        'impact': 'medium'
    })
    
    return recommendations

def generate_waste_recommendations(data):
    waste_count = data.get('Waste Bag Weekly Count', 0)
    recommendations = []
    
    if waste_count > 2:
        recommendations.append({
            'category': 'Waste',
            'title': 'Reduce household waste',
            'description': 'Your waste production is significant. Try composting food scraps and buying products with less packaging.',
            'impact': 'high'
        })
    
    recycling = data.get('Recycling', [])
    if not recycling or 'None' in recycling:
        recommendations.append({
            'category': 'Waste',
            'title': 'Start recycling program',
            'description': 'Begin separating recyclables from your regular trash to significantly reduce your waste impact.',
            'impact': 'high'
        })
    elif len(recycling) < 3:
        recommendations.append({
            'category': 'Waste',
            'title': 'Expand your recycling habits',
            'description': 'Add more materials to your recycling routine, such as glass, metal, and electronics.',
            'impact': 'medium'
        })
    
    return recommendations

def generate_grocery_recommendations(data):
    grocery_bill = data.get('Monthly Grocery Bill', 0)
    recommendations = []
    
    if grocery_bill > 400:
        recommendations.append({
            'category': 'Food',
            'title': 'Reduce food waste',
            'description': 'Plan meals ahead, store food properly, and use leftovers to reduce food waste and grocery expenses.',
            'impact': 'medium'
        })
    
    diet = data.get('Diet', '')
    if diet in ['Heavy Meat Eater', 'Omnivore']:
        recommendations.append({
            'category': 'Food',
            'title': 'Shift to more plant-based foods',
            'description': 'Gradually replace some meat-based meals with plant-based alternatives to reduce your carbon footprint.',
            'impact': 'high'
        })
    
    recommendations.append({
        'category': 'Food',
        'title': 'Buy local and seasonal produce',
        'description': 'Choose locally grown, seasonal foods to reduce transportation emissions and support local farmers.',
        'impact': 'medium'
    })
    
    return recommendations

def generate_internet_recommendations(data):
    internet_hours = data.get('How Long Internet Daily Hour', 0)
    recommendations = []
    
    if internet_hours > 5:
        recommendations.append({
            'category': 'Digital',
            'title': 'Reduce streaming quality',
            'description': 'Lower the resolution of video streaming services to reduce energy consumption.',
            'impact': 'low'
        })
    
    if internet_hours > 3:
        recommendations.append({
            'category': 'Digital',
            'title': 'Use energy-efficient devices',
            'description': 'Consider energy efficiency when purchasing new digital devices and enable power-saving modes.',
            'impact': 'low'
        })
    
    return recommendations

def generate_screen_time_recommendations(data):
    screen_hours = data.get('How Long TV PC Daily Hour', 0)
    recommendations = []
    
    if screen_hours > 4:
        recommendations.append({
            'category': 'Home Energy',
            'title': 'Reduce screen time',
            'description': 'Try to limit your daily screen time and turn off devices when not in use.',
            'impact': 'low'
        })
    
    recommendations.append({
        'category': 'Home Energy',
        'title': 'Use energy-efficient displays',
        'description': 'Choose energy-efficient monitors and TVs, and adjust brightness settings to reduce power consumption.',
        'impact': 'low'
    })
    
    return recommendations

def generate_transport_recommendations(data):
    transport = data.get('Transport', '')
    recommendations = []
    
    if transport == 'Car':
        recommendations.append({
            'category': 'Transport',
            'title': 'Use alternative transportation',
            'description': 'Try public transit, biking, or walking for shorter trips instead of driving.',
            'impact': 'high'
        })
    elif transport == 'Motorcycle':
        recommendations.append({
            'category': 'Transport',
            'title': 'Consider electric alternatives',
            'description': 'Electric motorcycles and scooters produce fewer emissions than gas-powered ones.',
            'impact': 'medium'
        })
    
    return recommendations

def generate_diet_recommendations(data):
    diet = data.get('Diet', '')
    recommendations = []
    
    if diet == 'Heavy Meat Eater':
        recommendations.append({
            'category': 'Food',
            'title': 'Reduce red meat consumption',
            'description': 'Start by replacing some red meat meals with chicken, fish, or plant-based alternatives.',
            'impact': 'high'
        })
    elif diet == 'Omnivore':
        recommendations.append({
            'category': 'Food',
            'title': 'Try meatless days',
            'description': 'Implement one or two meatless days per week to reduce your carbon footprint.',
            'impact': 'medium'
        })
    
    return recommendations

def generate_heating_recommendations(data):
    heating = data.get('Heating Energy Source', '')
    recommendations = []
    
    if heating in ['Coal', 'Oil']:
        recommendations.append({
            'category': 'Home Energy',
            'title': 'Switch to cleaner heating',
            'description': 'Consider switching to natural gas, electricity, or renewable energy sources for heating.',
            'impact': 'high'
        })
    
    recommendations.append({
        'category': 'Home Energy',
        'title': 'Improve home insulation',
        'description': 'Better insulation can significantly reduce heating needs and lower your carbon footprint.',
        'impact': 'medium'
    })
    
    return recommendations

def generate_energy_efficiency_recommendations(data):
    efficiency = data.get('Energy efficiency', '')
    recommendations = []
    
    if efficiency == 'No':
        recommendations.append({
            'category': 'Home Energy',
            'title': 'Upgrade to energy-efficient appliances',
            'description': 'Replace old appliances with energy-efficient models that use less electricity.',
            'impact': 'medium'
        })
        
        recommendations.append({
            'category': 'Home Energy',
            'title': 'Install LED lighting',
            'description': 'Replace all incandescent and fluorescent bulbs with energy-efficient LED lighting.',
            'impact': 'low'
        })
    
    return recommendations

def generate_recycling_recommendations(data):
    recycling = data.get('Recycling', [])
    recommendations = []
    
    if not recycling or 'None' in recycling:
        recommendations.append({
            'category': 'Waste',
            'title': 'Start basic recycling',
            'description': 'Begin with paper and plastic recycling, which are easy to implement and widely accepted.',
            'impact': 'medium'
        })
    elif len(recycling) < 3:
        recommendations.append({
            'category': 'Waste',
            'title': 'Expand recycling practices',
            'description': 'Add more materials to your recycling routine, such as glass, metal, and electronics.',
            'impact': 'medium'
        })
    
    return recommendations

def get_major_contributing_features(input_df, model, top_n=7):
    try:
        feature_contributions = {}
        coef = model.coef_

        for i, feature in enumerate(input_df.columns):
            contribution = coef[i] * input_df.iloc[0, i]
            feature_contributions[feature] = contribution

        total_contribution = sum(abs(value) for value in feature_contributions.values())

        # Sort features by absolute contribution
        sorted_features = sorted(feature_contributions.items(), key=lambda x: abs(x[1]), reverse=True)

        major_features = []
        others_contribution = 0

        for i, (k, v) in enumerate(sorted_features):
            percentage = (abs(v) / total_contribution) * 100 if total_contribution != 0 else 0
            if i < top_n:
                major_features.append({
                    'feature': k,
                    'contribution': v,
                    'percentage': round(percentage, 2)
                })
            else:
                others_contribution += abs(v)

        if len(sorted_features) > top_n:
            others_percentage = (others_contribution / total_contribution) * 100 if total_contribution != 0 else 0
            major_features.append({
                'feature': 'Others',
                'contribution': 0,  # optional, since it's aggregated
                'percentage': round(others_percentage, 2)
            })

        print(f"Major contributing features (with 'Others'): {major_features}")
        return major_features

    except Exception as e:
        print(f"Error in get_major_contributing_features: {str(e)}")
        return []

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        print(f"Received data for prediction: {data}")  # Debugging the incoming data

        # Preprocess input
        input_df = preprocess_input(data)

        # Predict
        prediction = model.predict(input_df)[0]  # Predict and take first element (since input_df has 1 row)
        print(f"Prediction result: {prediction}")  # Debugging the prediction

        # Return prediction as JSON
        return jsonify({'prediction': prediction})
    
    except Exception as e:
        print(f"Error in /predict: {str(e)}")  # Debugging the error in prediction route
        return jsonify({'error': str(e)}), 500

@app.route('/insights', methods=['POST'])
def get_insights():
    try:
        data = request.json.get('carbonData', {})
        print(f"Received data for insights: {data}")  # Debugging

        # Preprocess input to get features ready
        input_df = preprocess_input(data)

        # Predict carbon emission
        prediction = abs(model.predict(input_df)[0])  # Annual carbon emission in tons
        print("prediction : " ,prediction)
        # Generate major contributing features
        major_features = get_major_contributing_features(input_df, model)

        # Generate targeted recommendations based on major contributing features
        recommendations = generate_recommendations(data, major_features)

        # Build the full insights response
        full_insights = {
            'major_contributing_features': major_features,
            'recommendations': recommendations
        }

        # Return JSON with carbon emission + insights
        return jsonify({
            'carbonEmission': prediction,
            'insights': full_insights
        })

    except Exception as e:
        print(f"Error in /insights: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)