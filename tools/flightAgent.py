import os
from typing import Optional
from crewai.tools import BaseTool
import serpapi
from pydantic import  Field
from serpapi import GoogleSearch
api_key = os.getenv("serpapi")



class FlightsFinderTool(BaseTool):

    name: str = "Flights Finder"
    description: str = "Find flights using the Google Flights engine"

    
    def _run(
        self, 
        departure_airport: Optional[str] = Field(description='Departure airport code (IATA)'),
        arrival_airport: Optional[str] = Field(description='Arrival airport code (IATA)'),
        outbound_date: Optional[str] = Field(description='Outbound date in YYYY-MM-DD format'),
        return_date: Optional[str] = Field(description='Return date in YYYY-MM-DD format'),
        adults: int = 1,  # Default to 1
        children: int = 0,  # Defaultt to 0
        infants_in_seat: int = 0,  # Default to 0
        infants_on_lap: int = 0,  # Default to 0
        stops:int=0
    ) -> dict:

        params = {
            'engine': 'google_flights',
            'departure_id': departure_airport,
            'arrival_id': arrival_airport,
            'outbound_date': outbound_date,
            'return_date': return_date,
            'currency': 'INR',
            'adults': adults,
            'children': children,
            'infants_in_seat': infants_in_seat,
            'infants_on_lap': infants_on_lap,
            'stops':stops,
            'hl': 'en',
            'gl': 'us',
            'api_key': api_key
        }
        
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            results = results['best_flights']
        except Exception as e:
            results = str(e)

        return results
    
