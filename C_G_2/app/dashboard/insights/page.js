"use client"

import { useState, useEffect } from "react"
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts"
import { ArrowDown, ArrowUp, Leaf, AlertTriangle, Info, FileInput } from "lucide-react"
import Link from "next/link"
import { getCarbonData, getLatestCarbonData, kgToTons } from "../../api/carbon"

export default function Insights() {
  const [userData, setUserData] = useState(null)
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    const fetchData = async () => {
      try {
        // Get all carbon data for trends
        const allData = await getCarbonData()

        // Get latest data with insights
        const latestData = await getLatestCarbonData()

        // Process monthly trend data
        const monthlyTrend = processMonthlyTrend(allData)

        // Transform data for UI
        const transformedData = {
          carbonFootprint: {
            // Convert kg to tons
            current: kgToTons(latestData.latestData?.carbonEmission || 0),
            previous: kgToTons(latestData.previousData?.carbonEmission || 0),
            change: latestData.changePercentage || 0,
            globalAverage: 1.35, // Static value for now (monthly in tons)
            countryAverage: 1.31, // Static value for now (monthly in tons)
          },
          // Use the major contributing features for the breakdown if available
          breakdown: latestData.insights?.major_contributing_features
            ? latestData.insights.major_contributing_features.map((item) => ({
                name: formatFeatureName(item.feature),
                value: Math.abs(kgToTons(item.contribution)), // Convert to tons and ensure positive
                percentage: item.percentage,
              }))
            : [],
          monthlyTrend: monthlyTrend,
          recommendations: latestData.insights?.recommendations || [],
        }

        setUserData(transformedData)
      } catch (err) {
        console.error("Error fetching data:", err)
        setError("Failed to load insights data. Please try again later.")
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  // Add a helper function to format feature names for display
  const formatFeatureName = (featureName) => {
    // Convert camelCase or snake_case to Title Case with spaces
    return featureName
      .replace(/([A-Z])/g, " $1") // Insert space before capital letters
      .replace(/_/g, " ") // Replace underscores with spaces
      .replace(/^\w/, (c) => c.toUpperCase()) // Capitalize first letter
      .trim()
  }

  // Update the processMonthlyTrend function to work with kg values
  const processMonthlyTrend = (data) => {
    if (!data || data.length === 0) {
      // Return mock data if no real data available
      return [
        { month: "Jan", emission: 0.12 }, // Monthly values in tons
        { month: "Feb", emission: 0.115 },
        { month: "Mar", emission: 0.11 },
        { month: "Apr", emission: 0.11 },
        { month: "May", emission: 0.105 },
        { month: "Jun", emission: 0.1 },
      ]
    }

    // Sort data by date
    const sortedData = [...data].sort((a, b) => new Date(a.date) - new Date(b.date))

    // Group by month and year
    const monthlyData = {}
    sortedData.forEach((entry) => {
      const date = new Date(entry.date)
      const monthYear = `${date.getFullYear()}-${date.getMonth() + 1}`
      const monthName = date.toLocaleString("default", { month: "short" })

      // Convert kg to tons
      const emissionInTons = kgToTons(entry.carbonEmission)

      if (!monthlyData[monthYear]) {
        monthlyData[monthYear] = {
          month: monthName,
          emission: emissionInTons,
          count: 1,
        }
      } else {
        monthlyData[monthYear] = {
          month: monthName,
          emission:
            (monthlyData[monthYear].emission * monthlyData[monthYear].count + emissionInTons) /
            (monthlyData[monthYear].count + 1),
          count: monthlyData[monthYear].count + 1,
        }
      }
    })

    // Convert to array and take last 6 months (or all if less than 6)
    const result = Object.values(monthlyData)
    return result.slice(-6)
  }

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-emerald-500"></div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-7xl mx-auto">
        <div className="bg-red-50 text-red-600 p-4 rounded-lg mb-6">{error}</div>
        <Link
          href="/dashboard/carbon-form"
          className="bg-emerald-600 text-white py-2 px-4 rounded-md hover:bg-emerald-500 inline-flex items-center"
        >
          <FileInput className="h-4 w-4 mr-2" />
          Submit Your Carbon Data
        </Link>
      </div>
    )
  }

  // Prepare data for pie chart
  const pieData = userData.breakdown.map((item) => ({
    name: item.name,
    value: item.value,
    percentage: item.percentage,
  }))

  // Colors for pie chart
  const COLORS = ["#16a34a", "#22c55e", "#4ade80", "#86efac", "#bbf7d0"]

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-2xl font-bold">Carbon Footprint Insights</h1>
        <p className="text-gray-600">Detailed analysis of your environmental impact</p>
      </div>

      {/* Overview cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Your Monthly Carbon Footprint</h3>
          <div className="flex items-end">
            <span className="text-3xl font-bold">{userData.carbonFootprint.current.toFixed(2)}</span>
            <span className="text-gray-500 ml-1 mb-1">tons CO₂/month</span>
          </div>
          <div className="mt-2 flex items-center">
            <span
              className={`text-sm font-medium ${userData.carbonFootprint.change < 0 ? "text-green-600" : "text-red-600"}`}
            >
              {userData.carbonFootprint.change.toFixed(2)}%
            </span>
            {userData.carbonFootprint.change < 0 ? (
              <ArrowDown className="h-4 w-4 text-green-600 ml-1" />
            ) : (
              <ArrowUp className="h-4 w-4 text-red-600 ml-1" />
            )}
            <span className="text-gray-500 text-sm ml-1">vs previous</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Global Monthly Average</h3>
          <div className="flex items-end">
            <span className="text-3xl font-bold">{userData.carbonFootprint.globalAverage.toFixed(2)}</span>
            <span className="text-gray-500 ml-1 mb-1">tons CO₂/month</span>
          </div>
          <div className="mt-2 flex items-center">
            <span
              className={`text-sm font-medium ${userData.carbonFootprint.current < userData.carbonFootprint.globalAverage ? "text-green-600" : "text-red-600"}`}
            >
              {(
                ((userData.carbonFootprint.current - userData.carbonFootprint.globalAverage) /
                  userData.carbonFootprint.globalAverage) *
                100
              ).toFixed(2)}
              %
            </span>
            {userData.carbonFootprint.current < userData.carbonFootprint.globalAverage ? (
              <ArrowDown className="h-4 w-4 text-green-600 ml-1" />
            ) : (
              <ArrowUp className="h-4 w-4 text-red-600 ml-1" />
            )}
            <span className="text-gray-500 text-sm ml-1">your footprint</span>
          </div>
        </div>

        <div className="bg-white p-6 rounded-xl border border-gray-200">
          <h3 className="text-sm font-medium text-gray-500 mb-2">Country Monthly Average</h3>
          <div className="flex items-end">
            <span className="text-3xl font-bold">{userData.carbonFootprint.countryAverage.toFixed(2)}</span>
            <span className="text-gray-500 ml-1 mb-1">tons CO₂/month</span>
          </div>
          <div className="mt-2 flex items-center">
            <span
              className={`text-sm font-medium ${userData.carbonFootprint.current < userData.carbonFootprint.countryAverage ? "text-green-600" : "text-red-600"}`}
            >
              {(
                ((userData.carbonFootprint.current - userData.carbonFootprint.countryAverage) /
                  userData.carbonFootprint.countryAverage) *
                100
              ).toFixed(2)}
              %
            </span>
            {userData.carbonFootprint.current < userData.carbonFootprint.countryAverage ? (
              <ArrowDown className="h-4 w-4 text-green-600 ml-1" />
            ) : (
              <ArrowUp className="h-4 w-4 text-red-600 ml-1" />
            )}
            <span className="text-gray-500 text-sm ml-1">your footprint</span>
          </div>
        </div>
      </div>

      {/* Major Contributing Factors Pie Chart - Full width */}
      <div className="bg-white p-6 rounded-xl border border-gray-200 mb-8">
        <h2 className="text-xl font-semibold mb-6">Major Contributing Factors</h2>
        <div className="h-96">
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={pieData}
                cx="50%"
                cy="50%"
                labelLine={true}
                outerRadius={130}
                fill="#8884d8"
                dataKey="value"
                label={({ name, percent }) => `${name} (${(percent * 100).toFixed(2)}%)`}
              >
                {pieData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value, name, props) => [
                  `${value.toFixed(3)} tons CO₂/month (${props.payload.percentage.toFixed(2)}%)`,
                  name,
                ]}
              />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Monthly trend */}
      <div className="bg-white p-6 rounded-xl border border-gray-200 mb-8">
        <h2 className="text-xl font-semibold mb-6">Monthly Emission Trend</h2>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={userData.monthlyTrend} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis label={{ value: "tons CO₂/month", angle: -90, position: "insideLeft" }} />
              <Tooltip formatter={(value) => [`${value.toFixed(3)} tons CO₂/month`, "Monthly Emission"]} />
              <Bar dataKey="emission" name="Monthly Emission" fill="#16a34a" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Recommendations */}
      <div className="bg-white p-6 rounded-xl border border-gray-200">
        <h2 className="text-xl font-semibold mb-6">Recommendations to Reduce Your Footprint</h2>
        <div className="space-y-6">
          {userData.recommendations.map((recommendation, index) => (
            <div key={index} className="border-l-4 border-emerald-500 pl-4 py-1">
              <div className="flex items-center mb-2">
                <span
                  className={`inline-flex items-center justify-center rounded-full p-1 mr-2 ${
                    recommendation.impact === "high"
                      ? "bg-red-100 text-red-600"
                      : recommendation.impact === "medium"
                        ? "bg-yellow-100 text-yellow-600"
                        : "bg-blue-100 text-blue-600"
                  }`}
                >
                  {recommendation.impact === "high" ? (
                    <AlertTriangle className="h-4 w-4" />
                  ) : recommendation.impact === "medium" ? (
                    <Info className="h-4 w-4" />
                  ) : (
                    <Leaf className="h-4 w-4" />
                  )}
                </span>
                <h3 className="font-semibold">{recommendation.title}</h3>
                <span className="ml-2 text-xs px-2 py-1 bg-gray-100 rounded-full text-gray-600">
                  {recommendation.category}
                </span>
              </div>
              <p className="text-gray-600 ml-7">{recommendation.description}</p>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
