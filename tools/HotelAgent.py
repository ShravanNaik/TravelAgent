import os
from typing import Optional, List, Dict
from crewai.tools import BaseTool
from pydantic import  Field
import serpapi
from serpapi import GoogleSearch
api_key = os.getenv("serpapi")

class HotelsFinderTool(BaseTool):
    name: str = "Hotels Finder"
    description: str = "Find hotels using the Google Hotels engine."

    def _run(self, 
            q: str = Field(description='Location of the hotel'),
            check_in_date: str = Field(description='Check-in date. The format is YYYY-MM-DD. e.g. 2024-06-22'),
            check_out_date: str = Field(description='Check-out date. The format is YYYY-MM-DD. e.g. 2024-06-28'),
            #sort_by: Optional[str] = Field(default=8, description='Parameter is used for sorting the results. Default is sort by highest rating'),
            adults: Optional[int] = Field(default=2, description='Number of adults. Default to 1.'),
            children: Optional[int] = Field(default=0, description='Number of children. Default to 0.'),
            rooms: Optional[int] = Field(default=0, description='Number of rooms. Default to 1.'),
            rating:Optional[int]=Field(default=8,description='Parameter is used for filtering the results to certain rating.'),
            hotel_class: Optional[str] = Field(default=4,description='Parameter defines to include only certain hotel class in the results. for example- 2,3,4')) -> str:
        """
        Search for hotels based on provided parameters
        
        Args:
            q (str): Destination or location for hotel search
            check_in_date (str): Check-in date in YYYY-MM-DD format
            check_out_date (str): Check-out date in YYYY-MM-DD format
            sort_by (Optional[str]): Sorting criteria for hotels. Defaults to '8' (best match)
            adults (Optional[int]): Number of adult travelers. Defaults to 1
            children (Optional[int]): Number of children. Defaults to 0
            rooms (Optional[int]): Number of rooms. Defaults to 1
            hotel_class (Optional[str]): Preferred hotel class/rating. Defaults to None
        
        Returns:
            str: Hotel search results
        """
        # Remove None values from params to avoid validation issues
        params = {
        'api_key': os.environ.get('serpapi'),
        'engine': 'google_hotels',
        'hl': 'en',
        'gl': 'us',
        'q': q,
        'check_in_date': check_in_date,
        'check_out_date': check_out_date,
        'currency': 'INR',
        'adults': adults,
        'children': children,
        'bedrooms': rooms,
        'rating':rating,
        'hotel_class':hotel_class ,
        }
        try:
            search = GoogleSearch(params)
            results = search.get_dict()
            results = results['properties']
        except Exception as e:
            results = str(e)

        return results
    