#!/usr/bin/env python3
"""
Flight Booking Example - Core Tools and Agents

This example demonstrates the new object-based create_function_tool API
and showcases functional composition patterns for building complex agents.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from pydantic import BaseModel, Field

from jaf import (
    Agent,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    create_function_tool,
    generate_run_id,
    generate_trace_id,
    run,
)
from jaf.core.tool_results import ToolResponse, ToolResult
from jaf.providers.model import make_litellm_provider


# Data models for flight booking
@dataclass
class Flight:
    """Flight information."""
    flight_number: str
    origin: str
    destination: str
    departure_time: datetime
    arrival_time: datetime
    price: float
    airline: str
    seats_available: int
    aircraft_type: str


@dataclass
class Booking:
    """Flight booking information."""
    booking_id: str
    flight: Flight
    passenger_name: str
    seat_number: str
    booking_status: str
    total_cost: float


# Tool argument models
class FlightSearchArgs(BaseModel):
    """Arguments for flight search."""
    origin: str = Field(description="Origin airport code (e.g., 'LAX')")
    destination: str = Field(description="Destination airport code (e.g., 'JFK')")
    departure_date: str = Field(description="Departure date in YYYY-MM-DD format")
    passengers: int = Field(default=1, description="Number of passengers")
    class_preference: Optional[str] = Field(default="economy", description="Preferred class: economy, business, first")


class SeatAvailabilityArgs(BaseModel):
    """Arguments for seat availability check."""
    flight_number: str = Field(description="Flight number to check")
    passenger_count: int = Field(default=1, description="Number of seats needed")


class BookFlightArgs(BaseModel):
    """Arguments for flight booking."""
    flight_number: str = Field(description="Flight number to book")
    passenger_name: str = Field(description="Full name of the passenger")
    seat_preference: Optional[str] = Field(default="window", description="Seat preference: window, aisle, middle")


class FlightStatusArgs(BaseModel):
    """Arguments for flight status check."""
    flight_number: str = Field(description="Flight number to check status")


class CancelBookingArgs(BaseModel):
    """Arguments for booking cancellation."""
    booking_id: str = Field(description="Booking ID to cancel")


# Mock data for demonstration
MOCK_FLIGHTS = [
    Flight("AA101", "LAX", "JFK", datetime.now() + timedelta(days=1, hours=8), 
           datetime.now() + timedelta(days=1, hours=14), 299.99, "American Airlines", 45, "Boeing 737"),
    Flight("UA205", "LAX", "JFK", datetime.now() + timedelta(days=1, hours=10), 
           datetime.now() + timedelta(days=1, hours=16), 349.99, "United Airlines", 32, "Airbus A320"),
    Flight("DL308", "LAX", "JFK", datetime.now() + timedelta(days=1, hours=14), 
           datetime.now() + timedelta(days=1, hours=20), 279.99, "Delta Airlines", 18, "Boeing 757"),
    Flight("BA401", "LAX", "LHR", datetime.now() + timedelta(days=2, hours=9), 
           datetime.now() + timedelta(days=3, hours=2), 899.99, "British Airways", 28, "Boeing 777"),
    Flight("LH505", "LAX", "FRA", datetime.now() + timedelta(days=2, hours=11), 
           datetime.now() + timedelta(days=3, hours=6), 1199.99, "Lufthansa", 15, "Airbus A380"),
]

MOCK_BOOKINGS: Dict[str, Booking] = {}


# Flight booking tool implementations using new API
async def search_flights_execute(args: FlightSearchArgs, context: Any) -> ToolResult:
    """Search for flights between origin and destination."""
    try:
        # Filter flights based on origin and destination
        matching_flights = [
            f for f in MOCK_FLIGHTS 
            if f.origin.lower() == args.origin.lower() and 
               f.destination.lower() == args.destination.lower() and
               f.seats_available >= args.passengers
        ]
        
        if not matching_flights:
            return ToolResponse.success(
                f"No flights found from {args.origin} to {args.destination} on {args.departure_date} for {args.passengers} passenger(s)."
            )
        
        # Format flight results
        flight_info = []
        for flight in matching_flights[:5]:  # Limit to top 5 results
            flight_info.append({
                "flight_number": flight.flight_number,
                "airline": flight.airline,
                "departure": flight.departure_time.strftime("%Y-%m-%d %H:%M"),
                "arrival": flight.arrival_time.strftime("%Y-%m-%d %H:%M"),
                "price": f"${flight.price}",
                "seats_available": flight.seats_available,
                "aircraft": flight.aircraft_type
            })
        
        result = {
            "search_criteria": {
                "origin": args.origin.upper(),
                "destination": args.destination.upper(),
                "date": args.departure_date,
                "passengers": args.passengers,
                "class": args.class_preference
            },
            "flights": flight_info,
            "total_found": len(matching_flights)
        }
        
        return ToolResponse.success(result)
        
    except Exception as e:
        return ToolResponse.error(f"Error searching flights: {str(e)}")


async def check_seat_availability_execute(args: SeatAvailabilityArgs, context: Any) -> ToolResult:
    """Check seat availability for a specific flight."""
    try:
        # Find the flight
        flight = next((f for f in MOCK_FLIGHTS if f.flight_number == args.flight_number), None)
        
        if not flight:
            return ToolResponse.validation_error(f"Flight {args.flight_number} not found.")
        
        available = flight.seats_available >= args.passenger_count
        
        result = {
            "flight_number": flight.flight_number,
            "airline": flight.airline,
            "requested_seats": args.passenger_count,
            "seats_available": flight.seats_available,
            "availability": "Available" if available else "Not Available",
            "price_per_seat": flight.price
        }
        
        return ToolResponse.success(result)
        
    except Exception as e:
        return ToolResponse.error(f"Error checking seat availability: {str(e)}")


async def book_flight_execute(args: BookFlightArgs, context: Any) -> ToolResult:
    """Book a flight for a passenger."""
    try:
        # Find the flight
        flight = next((f for f in MOCK_FLIGHTS if f.flight_number == args.flight_number), None)
        
        if not flight:
            return ToolResponse.validation_error(f"Flight {args.flight_number} not found.")
        
        if flight.seats_available < 1:
            return ToolResponse.validation_error(f"No seats available on flight {args.flight_number}.")
        
        # Generate booking ID and seat number
        booking_id = f"BK{datetime.now().strftime('%Y%m%d%H%M%S')}"
        seat_number = f"{flight.seats_available}A"  # Simple seat assignment
        
        # Create booking
        booking = Booking(
            booking_id=booking_id,
            flight=flight,
            passenger_name=args.passenger_name,
            seat_number=seat_number,
            booking_status="Confirmed",
            total_cost=flight.price
        )
        
        # Store booking and update seat availability
        MOCK_BOOKINGS[booking_id] = booking
        flight.seats_available -= 1
        
        result = {
            "booking_id": booking_id,
            "status": "Confirmed",
            "flight": {
                "flight_number": flight.flight_number,
                "airline": flight.airline,
                "route": f"{flight.origin} → {flight.destination}",
                "departure": flight.departure_time.strftime("%Y-%m-%d %H:%M"),
                "arrival": flight.arrival_time.strftime("%Y-%m-%d %H:%M")
            },
            "passenger": args.passenger_name,
            "seat": seat_number,
            "total_cost": f"${flight.price}",
            "confirmation": "Your flight has been successfully booked!"
        }
        
        return ToolResponse.success(result)
        
    except Exception as e:
        return ToolResponse.error(f"Error booking flight: {str(e)}")


async def check_flight_status_execute(args: FlightStatusArgs, context: Any) -> ToolResult:
    """Check the status of a flight."""
    try:
        flight = next((f for f in MOCK_FLIGHTS if f.flight_number == args.flight_number), None)
        
        if not flight:
            return ToolResponse.validation_error(f"Flight {args.flight_number} not found.")
        
        # Mock flight status
        now = datetime.now()
        if flight.departure_time > now + timedelta(hours=2):
            status = "On Time"
        elif flight.departure_time > now:
            status = "Boarding"
        else:
            status = "Departed"
        
        result = {
            "flight_number": flight.flight_number,
            "airline": flight.airline,
            "route": f"{flight.origin} → {flight.destination}",
            "scheduled_departure": flight.departure_time.strftime("%Y-%m-%d %H:%M"),
            "scheduled_arrival": flight.arrival_time.strftime("%Y-%m-%d %H:%M"),
            "status": status,
            "aircraft": flight.aircraft_type,
            "gate": "A15"  # Mock gate
        }
        
        return ToolResponse.success(result)
        
    except Exception as e:
        return ToolResponse.error(f"Error checking flight status: {str(e)}")


async def cancel_booking_execute(args: CancelBookingArgs, context: Any) -> ToolResult:
    """Cancel a flight booking."""
    try:
        booking = MOCK_BOOKINGS.get(args.booking_id)
        
        if not booking:
            return ToolResponse.validation_error(f"Booking {args.booking_id} not found.")
        
        if booking.booking_status == "Cancelled":
            return ToolResponse.validation_error(f"Booking {args.booking_id} is already cancelled.")
        
        # Update booking status and restore seat availability
        booking.booking_status = "Cancelled"
        booking.flight.seats_available += 1
        
        result = {
            "booking_id": args.booking_id,
            "status": "Cancelled",
            "flight_number": booking.flight.flight_number,
            "passenger": booking.passenger_name,
            "refund_amount": f"${booking.total_cost}",
            "message": "Your booking has been successfully cancelled. Refund will be processed within 3-5 business days."
        }
        
        return ToolResponse.success(result)
        
    except Exception as e:
        return ToolResponse.error(f"Error cancelling booking: {str(e)}")


# Create tools using the new object-based API
search_flights_tool = create_function_tool({
    'name': 'search_flights',
    'description': 'Search for available flights between origin and destination',
    'execute': search_flights_execute,
    'parameters': FlightSearchArgs,
    'metadata': {'category': 'flight_search', 'priority': 'high'}
})

check_seat_availability_tool = create_function_tool({
    'name': 'check_seat_availability',
    'description': 'Check seat availability for a specific flight',
    'execute': check_seat_availability_execute,
    'parameters': SeatAvailabilityArgs,
    'metadata': {'category': 'availability', 'priority': 'medium'}
})

book_flight_tool = create_function_tool({
    'name': 'book_flight',
    'description': 'Book a flight for a passenger',
    'execute': book_flight_execute,
    'parameters': BookFlightArgs,
    'metadata': {'category': 'booking', 'priority': 'high'}
})

check_flight_status_tool = create_function_tool({
    'name': 'check_flight_status',
    'description': 'Check the status of a flight',
    'execute': check_flight_status_execute,
    'parameters': FlightStatusArgs,
    'metadata': {'category': 'status', 'priority': 'low'}
})

cancel_booking_tool = create_function_tool({
    'name': 'cancel_booking',
    'description': 'Cancel a flight booking',
    'execute': cancel_booking_execute,
    'parameters': CancelBookingArgs,
    'metadata': {'category': 'booking', 'priority': 'medium'}
})


# Create a comprehensive flight booking agent
def flight_agent_instructions(state: RunState) -> str:
    """Instructions for the flight booking agent."""
    return """You are a helpful flight booking assistant. I can help you with:

1. **Search Flights**: Find available flights between airports
2. **Check Availability**: Verify seat availability for specific flights  
3. **Book Flights**: Reserve flights for passengers
4. **Flight Status**: Check current flight status and updates
5. **Cancel Bookings**: Cancel existing flight reservations

Please provide clear information about your travel needs, including:
- Origin and destination airports (use 3-letter codes like LAX, JFK)
- Travel dates
- Number of passengers
- Any specific preferences

I'll help you find the best options and guide you through the booking process!"""


flight_booking_agent = Agent(
    name="FlightBookingAgent",
    instructions=flight_agent_instructions,
    tools=[
        search_flights_tool,
        check_seat_availability_tool,
        book_flight_tool,
        check_flight_status_tool,
        cancel_booking_tool
    ],
    model_config=ModelConfig(
        name="gemini-2.0-flash",
        temperature=0.1,
        max_tokens=1000
    )
)


# Example usage and testing
async def main():
    """Demonstrate the flight booking system."""
    print("🛫 Flight Booking System Demo")
    print("=" * 50)
    
    # Create a mock model provider for testing
    class MockModelProvider:
        async def get_completion(self, state, agent, config):
            return {
                'message': {
                    'content': 'I can help you search for flights. Please provide your origin and destination airports.',
                    'tool_calls': None
                }
            }
    
    # Test the agent
    initial_state = RunState(
        run_id=generate_run_id(),
        trace_id=generate_trace_id(),
        messages=[Message(role='user', content="I want to book a flight from LAX to JFK")],
        current_agent_name="FlightBookingAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=MockModelProvider(),
        max_turns=3
    )
    
    result = await run(initial_state, config)
    
    print(f"✅ Agent Run Result: {result.outcome.status}")
    if hasattr(result.outcome, 'output') and result.outcome.output:
        print(f"📋 Final Output: {result.outcome.output}")
    
    # Test individual tools
    print("\n🔧 Testing Individual Tools:")
    
    # Test flight search
    search_result = await search_flights_execute(
        FlightSearchArgs(origin="LAX", destination="JFK", departure_date="2024-01-15"),
        {}
    )
    print(f"✈️ Flight Search: {search_result.status}")
    
    # Test seat availability
    availability_result = await check_seat_availability_execute(
        SeatAvailabilityArgs(flight_number="AA101", passenger_count=1),
        {}
    )
    print(f"💺 Seat Availability: {availability_result.status}")
    
    print("\n🎉 Flight booking demo completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())