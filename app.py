import sys
import importlib
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import re
import streamlit as st
import os
import base64
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
from crewai import Crew, Agent, Task, Process,LLM
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from textwrap import dedent
import pandas as pd
import plotly.express as px
from tools.flightAgent import FlightsFinderTool
from tools.HotelAgent import HotelsFinderTool
import networkx as nx
import matplotlib.pyplot as plt


load_dotenv()

# Set API keys from environment variables
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")

# Initialize toolss
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Set page config with wider layout and custom theme
st.set_page_config(
    page_title="✨ AI Travel Planner ✈️",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with enhanced styling
# Custom CSS - Removed any animation effects
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
        margin-bottom: 1rem;
    }
    .card {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .highlight {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .success-message {
        background-color: #e7f5e7;
        color: #2e7d32;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
    /* Removed any animation-related CSS */
</style>
""", unsafe_allow_html=True)

class TravelPlanningCrew:
    def __init__(self, destination, departure_airport, arrival_airport, 
                 outbound_date, return_date, 
                 num_travelers, hotel_city, rooms, adults, children, hotel_class, preferences, budget, special_requirements):
        self.destination = destination
        self.departure_airport = departure_airport
        self.arrival_airport = arrival_airport
        self.outbound_date = outbound_date
        self.return_date = return_date
        self.num_travelers = num_travelers
        self.rooms = rooms
        self.adults = adults
        self.children = children
        self.hotel_class = hotel_class
        self.hotel_city = hotel_city
        self.preferences = preferences
        self.budget = budget
        self.special_requirements = special_requirements
        
        # Initialize LLM
        self.llm=LLM(model='gemini/gemini-1.5-flash',api_key=os.environ["GOOGLE_API_KEY"],num_retries=2)
        # self.llm=LLM(model="openai/gpt-4o-mini",temperature=0.7,api_key=os.environ["OPENAI_API_KEY"])

        # Temp file paths for storing unverified content
        self.temp_output_dir = "temp_outputs"
        os.makedirs(self.temp_output_dir, exist_ok=True)
        
        self.temp_destination_file = f"{self.temp_output_dir}/temp_destination_guide.md"
        self.temp_flight_file = f"{self.temp_output_dir}/temp_flight_options.md"
        self.temp_hotel_file = f"{self.temp_output_dir}/temp_hotel_recommendations.md"
        self.temp_itinerary_file = f"{self.temp_output_dir}/temp_itinerary_recommendations.md"

                # Initialize empty temp files
        for file_path in [self.temp_destination_file, self.temp_flight_file, 
                         self.temp_hotel_file, self.temp_itinerary_file]:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("# Placeholder content\nThis file will be updated by the AI agent.")

    def create_agents(self):
        # Travel Destination Researcher Agent
        destination_researcher = Agent(
            role='Travel Destination Specialist',
            goal='Provide comprehensive insights about the destination, including best areas to stay, local attractions, and travel tips',
            backstory=dedent("""
            You are a seasoned travel researcher with extensive knowledge of global destinations. 
            Your expertise lies in uncovering hidden gems, understanding local cultures, 
            and providing actionable travel advice that goes beyond typical tourist information.
            You have visited over 100 countries and have written for major travel publications.
            """),
            llm=self.llm,
            tools=[search_tool],
            max_iter=5,
            verbose=True,
            allow_delegation=False
        )

        # Flight Booking Specialist Agent
        flight_specialist = Agent(
            role='Flight Booking Expert',
            goal='Find the most optimal and cost-effective flight options',
            backstory=dedent("""
            You are a flight booking professional with a keen eye for finding the best 
            travel routes, prices, and convenient flight times. Your mission is to 
            secure the most comfortable and economical flights for travelers.
            You have 15 years of experience in the airline industry and know how to 
            find the best deals and most comfortable routes.
            """),
            llm=self.llm,
            max_iter=5,
            tools=[FlightsFinderTool()],
            verbose=True
        )

        # Hotel Booking Specialist Agent
        hotel_specialist = Agent(
            role='Hotel Booking Strategist',
            goal='Identify and recommend the best accommodation options',
            backstory=dedent("""
            You are a hotel booking expert who understands the nuanced needs of travelers. 
            You excel at finding accommodations that balance comfort, location, amenities, 
            and budget to create the perfect stay experience.
            You've personally stayed in over 500 hotels worldwide and know exactly what makes
            a hotel exceptional for different types of travelers.
            """),
            llm=self.llm,
            tools=[HotelsFinderTool()],
            max_iter=5,
            verbose=True
        )

        itinerary_specialist = Agent(
            role="Travel Itinerary Specialist",
            goal="Create detailed, personalized travel itineraries based on traveler preferences and constraints",
            backstory=dedent("""
            You are an experienced travel planner with deep knowledge of global 
            destinations, local attractions, and optimal travel logistics. You excel at creating 
            balanced itineraries that match travelers' interests while accounting for practical
            considerations like travel time, budget constraints, and local conditions.
            You have planned over 1,000 successful trips for clients with diverse needs and preferences.
            """),
            llm=self.llm,
            tools=[search_tool],
            max_iter=5,
            verbose=True,
            allow_delegation=False
        )

        verification_specialist = Agent(
            role="Travel Content Verification Specialist",
            goal="Verify and enhance travel information while strictly maintaining content category boundaries",
            backstory=dedent("""
            You are a meticulous travel content reviewer with an eye for detail and quality.
            Your expertise lies in ensuring information is accurate, comprehensive, and presented
            in a user-friendly format. You verify facts, improve formatting, enhance descriptions,
            and ensure all content meets the highest standards before being presented to travelers.
            
            You are especially focused on ensuring content is appropriately structured with clear
            headings, bullet points where appropriate, and consistent formatting. Most importantly,
            you maintain strict boundaries between different content types and NEVER mix destination,
            flight, hotel, and itinerary information. Each document you verify must contain ONLY
            information relevant to its specific category.
            """),
            llm=self.llm,
            verbose=True,
            max_iter=5,
            allow_delegation=False
        )



        return destination_researcher, flight_specialist, hotel_specialist ,itinerary_specialist,verification_specialist

    def create_tasks(self, agents):
        destination_researcher, flight_specialist, hotel_specialist ,itinerary_specialist,verification_specialist= agents
        


        # Destination Research Task
        destination_research_task = Task(
            description=dedent(f"""
            Conduct comprehensive research on {self.destination}. 
            Provide detailed insights including:
            - Top attractions and must-visit locations with brief descriptions
            - Best neighborhoods for staying with pros and cons of each
            - Local culture, customs, and etiquette tips
            - Transportation options within the destination
            - Estimated budget recommendations for different spending levels
            - Seasonal considerations and best time to visit
            - Safety tips and health recommendations
            - Local cuisine and must-try dishes
            
            Your guide should be visually appealing with clear sections and headers.
            Include specific recommendations rather than generic advice.
            
            RESTRICTIONS:
            - Verify all information is factually accurate and up-to-date
            - Avoid generic travel advice that could apply to any destination
            - Ensure all recommendations are appropriate for the specified number of travelers ({self.num_travelers})
            - Consider the specified preferences: {self.preferences}
            - Format the guide in well-structured Markdown with clear headings and subheadings
            """),
            agent=destination_researcher,
            expected_output="A detailed travel guide with comprehensive destination insights ",
            output_file=self.temp_destination_file
        )

        # Flight Booking Task
        flight_booking_task = Task(
            description=dedent(f"""
            Find the best flight options from {self.departure_airport} to {self.arrival_airport}
            for {self.num_travelers} travelers on the following dates:
            - Outbound: {self.outbound_date}
            - Return: {self.return_date}
            
            Consider factors like:
            - Lowest prices and best value (not just the cheapest)
            - Convenient flight times (avoid very early or late flights)
            - Minimal layovers and total travel time
            - Airline reputation and comfort
            - Baggage allowance policies
            
            Present the results in a clean, easy-to-read Markdown table format with these columns:
            - Airline & Flight Numbers
            - Departure & Arrival Times (with dates)
            - Duration (including layovers)
            - Number of Stops
            - Price per person
            - Total price for all travelers
            - Notable features (baggage policy, meal service, etc.)
            
            RESTRICTIONS:
            - Include at least 3 but no more than 5 flight options
            - Ensure all prices are reasonably current (note if they are estimates)
            - Flag any red-eye flights or unusually long layovers
            - Verify that all flight options can accommodate {self.num_travelers} travelers
            - Clearly mark the best value option and explain why
            """),
            agent=flight_specialist,
            expected_output="A comprehensive list of best flight options with pricing and details ",
            output_file=self.temp_flight_file
        )

         # Hotel Booking Task
        hotel_booking_task = Task(
            description=dedent(f"""
            Find the best hotel options in {self.hotel_city} for the following criteria:
            - Check-in date: {self.outbound_date}
            - Check-out date: {self.return_date}
            - Number of rooms: {self.rooms}
            - Adults: {self.adults}
            - Children: {self.children}
            - Hotel class: {self.hotel_class} stars
            - Budget level: {self.budget}
            - Special requirements: {self.special_requirements}
            
            Focus your research on:
            - Location proximity to main attractions and neighborhoods
            - Value for money within the specified budget level
            - Amenities that match traveler preferences 
            - Recent guest reviews and satisfaction ratings
            - Room types that can accommodate the specified number of travelers
            - Child-friendly facilities if children are traveling
            
            For each hotel, provide:
            - Name and star rating
            - Brief description and unique selling points
            - Location details and nearby attractions
            - Room types available and recommendations
            - Key amenities and services
            - Price range with any special offers
            - Guest rating summary from multiple sources
            - Pros and cons based on recent guest reviews
            
            RESTRICTIONS:
            - Recommend exactly 5 hotels that represent different neighborhoods or styles
            - Verify all hotels are actually available for the specified dates
            - Ensure all recommended hotels can accommodate the specified number of travelers
            - Include at least one option that specifically addresses the special requirements
            - Format information in attractive Markdown with clear sections for each hotel
            - Clearly indicate which hotel represents the best overall value
            - Double-check that all prices are within the specified budget level
            """),
            agent=hotel_specialist,
            expected_output="Top 5 hotel recommendations with detailed descriptions ",
            output_file=self.temp_hotel_file
        )
     
                # Calculate the number of days for the itinerary
        date_format = "%Y-%m-%d"
        start_date = datetime.strptime(self.outbound_date, date_format)
        end_date = datetime.strptime(self.return_date, date_format)
        days = (end_date - start_date).days

        # Itinerary Task
        itinerary_task = Task(
            description=dedent(f"""
            Create a detailed {days}-day travel itinerary for {self.destination} based on the following parameters:
            
            - Traveler preferences: {self.preferences}
            - Budget level: {self.budget}
            - Travel dates: {self.outbound_date} to {self.return_date}
            - Number of travelers: {self.adults} adults and {self.children} children
            - Special requirements: {self.special_requirements}
            
            The itinerary should include:
            1. A day-by-day breakdown with timestamped activities
            2. Specific attraction names, not generic recommendations
            3. Estimated costs for each activity and meal
            4. Transportation recommendations between locations with options and times
            5. Restaurant suggestions for each meal with cuisine type and price range
            6. Alternative activities for bad weather or closures
            7. Rest periods and downtime, especially for families with children
            8. A balance of activities that match all stated preferences
            9. Tips for local etiquette and customs at each location
            
            RESTRICTIONS:
            - Ensure the pace is appropriate for families if children are traveling
            - Verify all attractions are open on the days they're scheduled
            - Include at least one activity daily that addresses each stated preference
            - Don't overpack the schedule - allow reasonable travel time between locations
            - Ensure all recommended activities fit within the stated budget level
            - Include specific meeting points and addresses, not just attraction names
            - Double-check that special requirements are addressed throughout the itinerary
            - Format the itinerary with clear day headings, timeframes, and sections
            """),
            agent=itinerary_specialist,
            expected_output="A comprehensive day-by-day travel itinerary in markdown format with all requested elements.",
            output_file=self.temp_itinerary_file
        )

               # Destination Guide Verification
        verify_destination_task = Task(
            description=dedent(f"""
            Perform a thorough verification of the destination guide for {self.destination}.
            
            Review the content in {self.temp_destination_file} and check for:
            1. Factual accuracy of all information
            2. Completeness of content based on the original requirements
            3. Relevance to the traveler's preferences: {self.preferences}
            4. Quality of recommendations and practical usefulness
            5. Proper formatting and organization
            6. Any missing critical information about the destination

            IMPORTANT: This document must ONLY contain destination information. 
            DO NOT include ANY flight details, hotel recommendations, or itinerary schedules.
            The destination guide should ONLY cover:
            - Attractions and points of interest
            - Local culture and customs
            - Neighborhoods and areas
            - Transportation within the destination
            - Safety tips and local information
                    
           If you find ANY flight, hotel, or itinerary content in this file, REMOVE IT IMMEDIATELY.
    
           If issues are found:
            - Correct factual errors
            - Add missing destination information
            - Improve formatting and structure
            - Enhance recommendations to better match traveler preferences
            
            Create a polished final version that meets all quality standards and ONLY contains destination guide information.
            """),
            agent=verification_specialist,
            expected_output="A verified and enhanced destination guide with ONLY destination information",
            output_file="destination_guide.md"
        )

        # Flight Options Verification
        verify_flights_task = Task(
            description=dedent(f"""
            Verify the flight options presented for the route from {self.departure_airport} to {self.arrival_airport}.
            
            Review the content in {self.temp_flight_file} and check for:
            1. Reasonable pricing and current availability
            2. Accuracy of flight times and durations
            3. Completeness of information for each flight option
            4. Clarity of presentation and formatting
            5. Appropriateness for {self.num_travelers} travelers
            6. Proper highlighting of the best value options
            
            IMPORTANT: This document must ONLY contain flight information.
            DO NOT include ANY destination guide content, hotel recommendations, or itinerary schedules.
            The flight options should ONLY cover:
            - Airline details and flight numbers
            - Departure and arrival times
            - Prices and fees
            - Layover information
            - Flight amenities and policies
            
            If you find ANY destination, hotel, or itinerary content in this file, REMOVE IT IMMEDIATELY.
            
            If issues are found:
            - Correct any unrealistic prices or flight details
            - Improve the table formatting for better readability
            - Ensure all required flight information is included
            - Add clear recommendations based on traveler needs
            - Remove any markdown code block indicators
            
            Create a polished final version that a traveler could use for booking decisions, containing ONLY flight information.
            """),
            agent=verification_specialist,
            expected_output="Verified and enhanced flight options  with ONLY flight information",
            output_file="flight_options.md"
        )

        verify_hotels_task = Task(
            description=dedent(f"""
            Verify the hotel recommendations for {self.hotel_city}.
            
            Review the content in {self.temp_hotel_file} and check for:
            1. Accuracy of hotel information and amenities
            2. Appropriateness for {self.adults} adults and {self.children} children
            3. Alignment with the requested {self.hotel_class}-star standard
            4. Suitability for the stated budget level: {self.budget}
            5. Addressing of special requirements: {self.special_requirements}
            6. Quality and completeness of the recommendations
            
            IMPORTANT: This document must ONLY contain hotel information.
            DO NOT include ANY destination guide content, flight options, or itinerary schedules.
            The hotel recommendations should ONLY cover:
            - Hotel names and ratings
            - Room types and prices
            - Hotel locations and amenities
            - Guest reviews and ratings
            - Special offers or amenities
            
            If you find ANY destination, flight, or itinerary content in this file, REMOVE IT IMMEDIATELY.
            
            If issues are found:
            - Update or correct hotel information
            - Ensure all hotels can accommodate the specified travelers
            - Verify price ranges are appropriate for the budget level
            - Improve formatting and presentation
            - Add any missing critical details about each property
            - Remove any markdown code block indicators
            
            Create a polished final version with verified, reliable hotel recommendations containing ONLY hotel information.
            """),
            agent=verification_specialist,
            expected_output="Verified and enhanced hotel recommendations  with ONLY hotel information",
            output_file="hotel_recommendations.md"
    )
        verify_itinerary_task = Task(
            description=dedent(f"""
            Verify the {days}-day itinerary for {self.destination}.
            
            Review the content in {self.temp_itinerary_file} and check for:
            1. Logical flow and realistic timing of activities
            2. Balanced inclusion of all stated preferences: {self.preferences}
            3. Appropriateness for {self.adults} adults and {self.children} children
            4. Realistic transportation times between activities
            5. Accuracy of opening hours and availability
            6. Alignment with the stated budget level: {self.budget}
            7. Inclusion of contingency plans and alternatives
            8. Addressing of special requirements: {self.special_requirements}
            
            IMPORTANT: This document must ONLY contain itinerary information.
            DO NOT include ANY general destination guide content, flight options, or hotel recommendations.
            The itinerary should ONLY cover:
            - Day-by-day schedule of activities
            - Time-specific plans and activities
            - Recommended restaurants and meals
            - Transportation between activities
            - Estimated costs for activities
            
            If you find ANY general destination information, flight details, or hotel listings in this file, REMOVE IT IMMEDIATELY.
            
            If issues are found:
            - Adjust timing to be more realistic
            - Balance activities better across stated preferences
            - Ensure child-friendly considerations if children are traveling
            - Add missing details for clarity and usefulness
            - Improve formatting and day-by-day structure
            - Remove any markdown code block indicators
            
            Create a polished final version that represents a realistic, enjoyable, and well-balanced itinerary containing ONLY day-by-day activity plans.
            """),
            agent=verification_specialist,
            expected_output="A verified and enhanced travel itinerary  with ONLY itinerary information",
            output_file="itinerary_recommendations.md"
    )



        return [destination_research_task,
                flight_booking_task,
                hotel_booking_task,itinerary_task,verify_destination_task,
                verify_flights_task,
                verify_hotels_task,
                verify_itinerary_task]

    def run(self, progress_bar):
    # Clear any existing output files to ensure fresh start
        output_files = {
            "destination_guide": "destination_guide.md",
            "flight_options": "flight_options.md",
            "hotel_recommendations": "hotel_recommendations.md",
            "itinerary_recommendations": "itinerary_recommendations.md"
        }
        
        for file_path in output_files.values():
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Create and run crew
        agents = self.create_agents()
        tasks = self.create_tasks(agents)
        
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )

        # Simulate progress
        for i in range(1, 101):
            progress_bar.progress(i/100)
            time.sleep(0.05)

        # Run the crew
        crew.kickoff()
       
        # Add small delay to ensure files are written
        time.sleep(1)

                # Verify final output files exist, if not create with error message
        for output_name, file_path in output_files.items():
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# {output_name.replace('_', ' ').title()}\n\n")
                    f.write("Sorry, there was an issue generating this content. ")
                    f.write("Please try again or adjust your parameters.")
        
        return output_files

def display_markdown_file(file_path, default_content=""):
    """Helper function to display markdown content from a file with retry logic"""
    max_retries = 3
    retry_delay = 0.5
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    if content.strip():  # Only return if file has content
                        return content
            time.sleep(retry_delay)  # Wait before retrying
        except Exception as e:
            st.warning(f"Attempt {attempt + 1} failed for {file_path}: {str(e)}")
            time.sleep(retry_delay)
    
    return default_content

def display_agent_cards():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="card pulse">
            <h3><span class="emoji">🌍</span> Destination Specialist</h3>
            <p>Researches your destination to provide insights on attractions,local culture, and travel tips.</p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="font-size: 2rem;">🌴</span>
                <span style="font-size: 2rem;">🏛️</span>
                <span style="font-size: 2rem;">🍜</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class="card pulse" style="animation-delay: 0.2s;">
            <h3><span class="emoji">✈️</span> Flight Expert</h3>
            <p>Finds the most optimal and cost-effective flight options for your journey.</p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="font-size: 2rem;">🛫</span>
                <span style="font-size: 2rem;">💺</span>
                <span style="font-size: 2rem;">🛄</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="card pulse" style="animation-delay: 0.4s;">
            <h3><span class="emoji">🏨</span> Hotel Strategist</h3>
            <p>Recommends the best accommodations based on your preferences and needs.</p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="font-size: 2rem;">🛏️</span>
                <span style="font-size: 2rem;">🏊</span>
                <span style="font-size: 2rem;">🌟</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="card pulse" style="animation-delay: 0.6s;">
            <h3><span class="emoji">📅</span> Itinerary Planner</h3>
            <p>Creates a detailed day-by-day schedule tailored to your preferences.</p>
            <div style="text-align: center; margin-top: 20px;">
                <span style="font-size: 2rem;">🗺️</span>
                <span style="font-size: 2rem;">⏰</span>
                <span style="font-size: 2rem;">🍽️</span>
            </div>
        </div>
        """, unsafe_allow_html=True)


def main():
    # Header with animated gradient
    st.markdown("""
    <div style="text-align: center;">
        <h1 class="main-header floating">✨ AI Travel Planner ✈️</h1>
        <p style="font-size: 1.3rem; color: #636e72; margin-bottom: 2rem;">
        Your personal AI travel crew to plan the perfect trip
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display animated agent cards
    display_agent_cards()
    
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
    
    # Sidebar for inputs with improved styling
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #6B73FF 0%, #000DFF 100%); 
                   padding: 15px; 
                   border-radius: 10px; 
                   color: white;
                   margin-bottom: 20px;">
            <h2 style="color: white; margin: 0;">✈️ Trip Details</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Destination details
        st.markdown('### 🌍 Destination')
        destination = st.text_input("Destination", "Bangkok, Thailand", key="destination")
        hotel_city = st.text_input("Hotel City", "Bangkok", key="hotel_city")
        
        # Flight details
        st.markdown('### ✈️ Flight Information')
        col1, col2 = st.columns(2)
        with col1:
            departure_airport = st.text_input("Departure Airport (Code)", "HYD", key="departure")
        with col2:
            arrival_airport = st.text_input("Arrival Airport (Code)", "BKK", key="arrival")
        
        # Date selection
        today = datetime.now().date()
        tomorrow = today + timedelta(days=1)
        next_week = today + timedelta(days=7)
        
        col1, col2 = st.columns(2)
        with col1:
            outbound_date = st.date_input("Departure Date", tomorrow + timedelta(days=100), key="outbound")
        with col2:
            return_date = st.date_input("Return Date", tomorrow + timedelta(days=107), key="return")
            
        
        # Traveler details
        st.markdown('### 👪 Travelers')
        col1, col2, col3 = st.columns(3)
        with col1:
            adults = st.number_input("Adults", min_value=1, value=2, key="adults")
        with col2:
            children = st.number_input("Children", min_value=0, value=1, key="children")
        with col3:
            rooms = st.number_input("Rooms", min_value=1, value=1, key="rooms")
        
        num_travelers = adults + children
        
        # Hotel preferences
        st.markdown('### 🏨 Hotel Preferences')
        hotel_class = st.slider("Hotel Class (Stars)", min_value=1, max_value=5, value=4, key="stars")
        
        # Itinerary preferences
        st.markdown('### 📅 Itinerary Preferences')
        preferences = st.text_area("Travel Interests", 
                                  "historical sites, local cuisine, nature, shopping", 
                                  help="What activities, attractions, or experiences are you interested in?",
                                  key="preferences")
        
        budget_options = ["Budget", "Mid-range", "Luxury"]
        budget = st.selectbox("Budget Level", budget_options, index=1, key="budget")
        
        special_requirements = st.text_area("Special Requirements", 
                                         "Child-friendly activities, Transport & Connectivity", 
                                         help="Any special needs or requirements for your trip?",
                                         key="requirements")

        st.markdown("""
        <div style="margin-top: 30px; text-align: center;">
            <small>Powered by AI Travel Agents </small>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    if st.button('✨ Create My Travel Plan', use_container_width=True, key="generate"):
        # Format dates as strings
        outbound_date_str = outbound_date.strftime("%Y-%m-%d")
        return_date_str = return_date.strftime("%Y-%m-%d")
        
        # Show progress while calculating with animated header
        st.markdown("""
        <div style="text-align: center; margin-bottom: 30px;">
            <h2 class="sub-header" style="border: none;">Building your personalized travel plan...</h2>
            <div style="font-size: 3rem;" class="floating">
                ✈️ 🌍 🏨 📅
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        progress_bar = st.progress(0)
        
        # Create tabs for organized results with custom styling
        tab1, tab2, tab3, tab4 = st.tabs([
            "🌍 Destination Guide", 
            "✈️ Flight Options", 
            "🏨 Hotel Recommendations", 
            "📅 Itinerary"
        ])
        
        with st.spinner('Our AI agents crew are working on your perfect travel plan please be patient...'):
            travel_crew = TravelPlanningCrew(
                destination=destination,
                departure_airport=departure_airport,
                arrival_airport=arrival_airport,
                outbound_date=outbound_date_str,
                return_date=return_date_str,
                num_travelers=num_travelers,
                hotel_city=hotel_city,
                rooms=rooms,
                adults=adults,
                children=children,
                hotel_class=hotel_class,
                preferences=preferences,
                budget=budget,
                special_requirements=special_requirements
            )
            
            try:
                # Run the crew and get results
                result_files = travel_crew.run(progress_bar)
                
                # Process the results with enhanced tab styling
                with tab1:  # Destination Guide
                    st.markdown("""
                    <div class="tab-content">
                        <h3 style="color: #6B73FF; border-bottom: 2px solid #dfe6e9; padding-bottom: 10px;">
                            🌍 Destination Guide
                        </h3>
                    """, unsafe_allow_html=True)
                    content = display_markdown_file(
                        result_files["destination_guide"],
                        "Destination information will be displayed here"
                    )
                    st.markdown(content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab2:  # Flight Options
                    st.markdown("""
                    <div class="tab-content">
                        <h3 style="color: #FFA726; border-bottom: 2px solid #dfe6e9; padding-bottom: 10px;">
                            ✈️ Flight Options
                        </h3>
                    """, unsafe_allow_html=True)
                    content = display_markdown_file(
                        result_files["flight_options"],
                        "Flight options will be displayed here"
                    )
                    st.markdown(content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                with tab3:  # Hotel Recommendations
                    st.markdown("""
                    <div class="tab-content">
                        <h3 style="color: #66BB6A; border-bottom: 2px solid #dfe6e9; padding-bottom: 10px;">
                            🏨 Hotel Recommendations
                        </h3>
                    """, unsafe_allow_html=True)
                    content = display_markdown_file(
                        result_files["hotel_recommendations"],
                        "Hotel recommendations will be displayed here"
                    )
                    st.markdown(content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                with tab4:  # Itinerary
                    st.markdown("""
                    <div class="tab-content">
                        <h3 style="color: #AB47BC; border-bottom: 2px solid #dfe6e9; padding-bottom: 10px;">
                            📅 Personalized Itinerary
                        </h3>
                    """, unsafe_allow_html=True)
                    content = display_markdown_file(
                        result_files["itinerary_recommendations"],
                        "Your day-by-day itinerary will be displayed here"
                    )
                    st.markdown(content, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                
                # Success message with animation
                st.markdown("""
                <div class="success-message" style="animation: fadeIn 1s ease-in-out;">
                    <h3 style="margin: 0; display: flex; align-items: center;">
                        <span style="margin-right: 10px;">🎉</span> Your travel plan is ready!
                    </h3>
                    <p style="margin: 10px 0 0 0;">Explore the tabs above to see your personalized recommendations.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Add a download button for the full plan
                col1, col2, col3 = st.columns([1,2,1])
                with col2:
                    st.download_button(
                        label="📥 Download Full Travel Plan",
                        data=create_zip_file(result_files),
                        file_name=f"{destination}_travel_plan.zip",
                        mime="application/zip",
                        use_container_width=True
                    )
                
            except Exception as e:
                # Log the error for debugging
                import logging
                logging.error(f"Travel plan generation error: {str(e)}")
                
                # Show a more specific error message to the user
                error_type = type(e).__name__
                st.error(f"Error while generating travel plan: {error_type}")
                
                # Show more details in an expander for technical users
                with st.expander("Technical Details"):
                    st.code(str(e))
                
                # Provide a helpful message before showing fallback data
                st.warning("We couldn't generate a personalized travel plan based on your inputs.")
                
  

def create_zip_file(result_files):
    """Create a zip file containing all travel plan documents"""
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
        for file_name, file_path in result_files.items():
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    zip_file.writestr(f"{file_name}.md", file.read())
    
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


if __name__ == "__main__":
    main()
