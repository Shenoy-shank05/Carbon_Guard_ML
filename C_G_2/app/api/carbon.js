import axios from "axios"
import { setAuthToken } from "./auth"

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5001/api"
const FLASK_API_URL = process.env.NEXT_PUBLIC_FLASK_API_URL || "http://localhost:5000"

// Submit carbon footprint data
export const submitCarbonData = async (formData) => {
  try {
    // First, get prediction from Flask API
    console.log("Sending data to Flask API:", formData)
    const flaskResponse = await axios.post(`${FLASK_API_URL}/predict`, formData)
    console.log("Flask API response:", flaskResponse.data)

    // Check if we have a prediction
    if (flaskResponse.data && flaskResponse.data.prediction !== undefined) {
      // Get insights with the same data
      try {
        const insightsResponse = await axios.post(`${FLASK_API_URL}/insights`, { carbonData: formData })
        console.log("Flask API insights response:", insightsResponse.data)

        return {
          prediction: flaskResponse.data.prediction,
          insights: insightsResponse.data.insights,
        }
      } catch (insightsError) {
        console.error("Error getting insights from Flask API:", insightsError)
        return {
          prediction: flaskResponse.data.prediction,
          insights: generateFallbackInsights(formData),
        }
      }
    } else {
      // Generate a fallback prediction if the API doesn't return one
      console.warn("No prediction received from Flask API, using fallback value")
      return {
        prediction: calculateFallbackPrediction(formData),
        insights: generateFallbackInsights(formData),
      }
    }
  } catch (error) {
    console.error("Error in submitCarbonData:", error)
    // Generate a fallback prediction if the API call fails
    console.warn("Flask API call failed, using fallback prediction")
    return {
      prediction: calculateFallbackPrediction(formData),
      insights: generateFallbackInsights(formData),
    }
  }
}

// Save carbon data to database with proper field mapping
export const saveCarbonDataToDatabase = async (formData, carbonEmission) => {
  try {
    const token = localStorage.getItem("token")
    setAuthToken(token)

    // Map API field names to MongoDB schema field names
    const dbData = {
      bodyType: formData["Body Type"],
      sex: formData["Sex"],
      diet: formData["Diet"],
      howOftenShower: formData["How Often Shower"],
      heatingEnergySource: formData["Heating Energy Source"],
      transport: formData["Transport"],
      vehicleType: formData["Vehicle Type"],
      socialActivity: formData["Social Activity"],
      monthlyGroceryBill: formData["Monthly Grocery Bill"],
      frequencyOfTravelingByAir: formData["Frequency of Traveling by Air"],
      vehicleMonthlyDistanceKm: formData["Vehicle Monthly Distance Km"],
      wasteBagSize: formData["Waste Bag Size"],
      wasteBagWeeklyCount: formData["Waste Bag Weekly Count"],
      howLongTvPcDailyHour: formData["How Long TV PC Daily Hour"],
      howManyNewClothesMonthly: formData["How Many New Clothes Monthly"],
      howLongInternetDailyHour: formData["How Long Internet Daily Hour"],
      energyEfficiency: formData["Energy efficiency"],
      recycling: formData["Recycling"],
      cookingWith: formData["Cooking_With"],
      // Store the prediction in kg as is
      carbonEmission: carbonEmission,
    }

    console.log("Saving to database with mapped fields:", dbData)

    const response = await axios.post(`${API_URL}/carbon-data`, dbData)
    return response.data
  } catch (error) {
    console.error("Error saving to database:", error)
    throw error.response?.data || { message: "Failed to save carbon data to database" }
  }
}

// Calculate a fallback prediction if the API call fails
function calculateFallbackPrediction(formData) {
  // Simple calculation based on form data
  let emission = 1500 // Base value in kg

  // Add based on transport
  if (formData["Transport"] === "private") emission += 500
  if (formData["Transport"] === "public") emission += 200

  // Add based on diet
  if (formData["Diet"] === "omnivore") emission += 400
  if (formData["Diet"] === "vegetarian") emission -= 100
  if (formData["Diet"] === "vegan") emission -= 200

  // Add based on heating
  if (formData["Heating Energy Source"] === "coal") emission += 300
  if (formData["Heating Energy Source"] === "natural gas") emission += 200

  // Add based on vehicle distance
  emission += formData["Vehicle Monthly Distance Km"] * 0.5

  // Add random variation
  emission += Math.random() * 200

  return Number(emission.toFixed(2))
}

// Generate fallback insights if the API doesn't provide them
function generateFallbackInsights(formData) {
  // Create mock major contributing features
  const majorFeatures = [
    {
      feature: "Vehicle Monthly Distance Km",
      contribution: formData["Vehicle Monthly Distance Km"] * 0.5,
      percentage: 30,
    },
    {
      feature: "How Many New Clothes Monthly",
      contribution: formData["How Many New Clothes Monthly"] * 20,
      percentage: 25,
    },
    {
      feature: "Waste Bag Weekly Count",
      contribution: formData["Waste Bag Weekly Count"] * 30,
      percentage: 20,
    },
    {
      feature: "Monthly Grocery Bill",
      contribution: formData["Monthly Grocery Bill"] * 0.1,
      percentage: 15,
    },
    {
      feature: "How Long Internet Daily Hour",
      contribution: formData["How Long Internet Daily Hour"] * 5,
      percentage: 10,
    },
  ]

  return {
    major_contributing_features: majorFeatures,
    recommendations: [
      {
        category: "Transport",
        title: "Reduce car usage",
        description: "Try using public transportation, biking, or walking for short trips.",
        impact: "high",
      },
      {
        category: "Home Energy",
        title: "Switch to cleaner energy",
        description: "Consider switching to renewable energy sources for your home.",
        impact: "high",
      },
      {
        category: "Food",
        title: "Reduce meat consumption",
        description: "Try incorporating more plant-based meals into your diet each week.",
        impact: "medium",
      },
      {
        category: "Consumption",
        title: "Buy fewer new clothes",
        description: "Consider second-hand shopping or extending the life of your current wardrobe.",
        impact: "medium",
      },
      {
        category: "Waste",
        title: "Improve recycling habits",
        description: "Expand your recycling to include more materials like glass and electronics.",
        impact: "low",
      },
    ],
  }
}

// Get all carbon footprint data for user
export const getCarbonData = async () => {
  try {
    const token = localStorage.getItem("token")
    setAuthToken(token)

    const response = await axios.get(`${API_URL}/carbon-data`)
    return response.data
  } catch (error) {
    console.error("Error fetching carbon data:", error)
    // Return empty array as fallback
    return []
  }
}

// Get latest carbon footprint data with insights
export const getLatestCarbonData = async () => {
  try {
    const token = localStorage.getItem("token")
    setAuthToken(token)

    const response = await axios.get(`${API_URL}/carbon-data/latest`)

    // If we have data, try to get insights with properly formatted field names
    if (response.data && response.data.latestData) {
      try {
        // Format the data with the field names the model expects
        const formattedData = {
          "Body Type": response.data.latestData.bodyType,
          Sex: response.data.latestData.sex,
          Diet: response.data.latestData.diet,
          "How Often Shower": response.data.latestData.howOftenShower,
          "Heating Energy Source": response.data.latestData.heatingEnergySource,
          Transport: response.data.latestData.transport,
          "Vehicle Type": response.data.latestData.vehicleType,
          "Social Activity": response.data.latestData.socialActivity,
          "Monthly Grocery Bill": response.data.latestData.monthlyGroceryBill,
          "Frequency of Traveling by Air": response.data.latestData.frequencyOfTravelingByAir,
          "Vehicle Monthly Distance Km": response.data.latestData.vehicleMonthlyDistanceKm,
          "Waste Bag Size": response.data.latestData.wasteBagSize,
          "Waste Bag Weekly Count": response.data.latestData.wasteBagWeeklyCount,
          "How Long TV PC Daily Hour": response.data.latestData.howLongTvPcDailyHour,
          "How Many New Clothes Monthly": response.data.latestData.howManyNewClothesMonthly,
          "How Long Internet Daily Hour": response.data.latestData.howLongInternetDailyHour,
          "Energy efficiency": response.data.latestData.energyEfficiency,
          Recycling: response.data.latestData.recycling,
          Cooking_With: response.data.latestData.cookingWith,
        }

        // Get insights directly from Flask API with properly formatted data
        console.log("Sending formatted data to Flask API for insights:", formattedData)
        const insightsResponse = await axios.post(`${FLASK_API_URL}/insights`, { carbonData: formattedData })

        if (insightsResponse.data && insightsResponse.data.insights) {
          // Replace the insights in the response with the ones from Flask API
          response.data.insights = insightsResponse.data.insights
        }
      } catch (insightsError) {
        console.error("Error getting insights from Flask API:", insightsError)
        // Keep the original insights if available, or use fallback
        if (!response.data.insights) {
          response.data.insights = generateFallbackInsights({
            Transport: response.data.latestData.transport,
            "Heating Energy Source": response.data.latestData.heatingEnergySource,
            Diet: response.data.latestData.diet,
            "How Many New Clothes Monthly": response.data.latestData.howManyNewClothesMonthly,
            "Waste Bag Weekly Count": response.data.latestData.wasteBagWeeklyCount,
            "Monthly Grocery Bill": response.data.latestData.monthlyGroceryBill,
            "Vehicle Monthly Distance Km": response.data.latestData.vehicleMonthlyDistanceKm,
            "How Long Internet Daily Hour": response.data.latestData.howLongInternetDailyHour,
          })
        }
      }
    }

    return response.data
  } catch (error) {
    console.error("Error fetching latest carbon data:", error)
    // Return fallback data
    return {
      latestData: {
        carbonEmission: 1250, // in kg
        date: new Date().toISOString(),
      },
      previousData: {
        carbonEmission: 1420, // in kg
      },
      changePercentage: -12,
      insights: generateFallbackInsights({
        Transport: "public",
        "Heating Energy Source": "electricity",
        Diet: "omnivore",
        "How Many New Clothes Monthly": 2,
        "Waste Bag Weekly Count": 1,
        "Monthly Grocery Bill": 300,
        "Vehicle Monthly Distance Km": 200,
        "How Long Internet Daily Hour": 4,
      }),
    }
  }
}

// Convert kg to tons
export const kgToTons = (kg) => {
  return kg / 1000
}
