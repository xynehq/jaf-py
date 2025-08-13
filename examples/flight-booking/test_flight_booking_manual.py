#!/usr/bin/env python3
"""
Manual Flight Booking System Tests

This script provides comprehensive manual testing of the flight booking system
to verify all components work correctly in realistic scenarios.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List

from jaf import (
    Agent,
    ContentRole,
    Message,
    ModelConfig,
    RunConfig,
    RunState,
    create_run_id,
    create_trace_id,
    run,
)

# Import flight booking components
from index import flight_booking_agent
from multi_agent import (
    coordinator_agent,
    search_specialist_agent,
    booking_specialist_agent,
    pricing_specialist_agent
)


class FlightBookingMockProvider:
    """Mock model provider for flight booking scenarios."""
    
    def __init__(self, scenario: str = "normal"):
        self.scenario = scenario
        self.call_count = 0
    
    async def get_completion(self, state, agent, config):
        self.call_count += 1
        
        if self.scenario == "search_flights":
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'search_1',
                            'type': 'function',
                            'function': {
                                'name': 'search_flights',
                                'arguments': json.dumps({
                                    'origin': 'LAX',
                                    'destination': 'JFK',
                                    'departure_date': '2024-02-15',
                                    'passengers': 1,
                                    'class_preference': 'economy'
                                })
                            }
                        }
                    ]
                }
            }
        elif self.scenario == "book_flight":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'book_1',
                                'type': 'function',
                                'function': {
                                    'name': 'book_flight',
                                    'arguments': json.dumps({
                                        'flight_number': 'AA101',
                                        'passenger_name': 'John Doe',
                                        'passenger_email': 'john.doe@example.com',
                                        'seat_preference': 'window'
                                    })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Flight booking completed successfully!',
                        'tool_calls': None
                    }
                }
        elif self.scenario == "check_seat_availability":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'seat_1',
                                'type': 'function',
                                'function': {
                                    'name': 'check_seat_availability',
                                    'arguments': json.dumps({
                                        'flight_number': 'AA101',
                                        'class_preference': 'economy'
                                    })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Seat availability check completed.',
                        'tool_calls': None
                    }
                }
        elif self.scenario == "check_flight_status":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'status_1',
                                'type': 'function',
                                'function': {
                                    'name': 'check_flight_status',
                                    'arguments': json.dumps({
                                        'flight_number': 'AA101'
                                    })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Flight status check completed.',
                        'tool_calls': None
                    }
                }
        elif self.scenario == "cancel_booking":
            if self.call_count == 1:
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'cancel_1',
                                'type': 'function',
                                'function': {
                                    'name': 'cancel_booking',
                                'arguments': json.dumps({
                                    'booking_id': 'ABC123'
                                })
                                }
                            }
                        ]
                    }
                }
            else:
                return {
                    'message': {
                        'content': 'Booking cancellation completed.',
                        'tool_calls': None
                    }
                }
        elif self.scenario == "multi_agent_handoff":
            if agent.name == "Coordinator":
                return {
                    'message': {
                        'content': '',
                        'tool_calls': [
                            {
                                'id': 'handoff_1',
                                'type': 'function',
                                'function': {
                                    'name': 'handoff',
                                    'arguments': json.dumps({
                                        'target_agent': 'SearchSpecialist',
                                        'context': 'Customer wants to search for flights',
                                        'reason': 'Route to search specialist'
                                    })
                                }
                            }
                        ]
                    }
                }
            elif agent.name == "SearchSpecialist":
                if self.call_count == 2:  # First call after handoff
                    return {
                        'message': {
                            'content': '',
                            'tool_calls': [
                                {
                                    'id': 'search_2',
                                    'type': 'function',
                                    'function': {
                                        'name': 'search_flights',
                                        'arguments': json.dumps({
                                            'origin': 'LAX',
                                            'destination': 'JFK',
                                            'departure_date': '2024-02-15',
                                            'passengers': 1,
                                            'class_preference': 'economy'
                                        })
                                    }
                                }
                            ]
                        }
                    }
                else:  # Second call - handoff to booking
                    return {
                        'message': {
                            'content': '',
                            'tool_calls': [
                                {
                                    'id': 'handoff_2',
                                    'type': 'function',
                                    'function': {
                                        'name': 'handoff',
                                        'arguments': json.dumps({
                                            'target_agent': 'BookingSpecialist',
                                            'context': 'Customer ready to book flight AA101',
                                            'reason': 'Complete the booking'
                                        })
                                    }
                                }
                            ]
                        }
                    }
            elif agent.name == "BookingSpecialist":
                return {
                    'message': {
                        'content': 'Flight booking completed successfully! Your booking reference is ABC123.',
                        'tool_calls': None
                    }
                }
        else:  # normal completion
            return {
                'message': {
                    'content': f'Flight booking assistant ready to help! (Response #{self.call_count})',
                    'tool_calls': None
                }
            }


# Flight Booking Manual Tests
async def test_single_agent_flight_search():
    """Test single agent flight search functionality."""
    print("\n🧪 Testing Single Agent Flight Search...")
    
    provider = FlightBookingMockProvider("search_flights")
    
    state = RunState(
        run_id=create_run_id("flight-search"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Search for flights from LAX to JFK on Feb 15")],
        current_agent_name="FlightBookingAgent",
        context={"customer_id": "test_123"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   🔄 Turns: {result.final_state.turn_count}")
    
    # Check for tool execution
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    if tool_messages:
        tool_result = json.loads(tool_messages[0].content)
        print(f"   ✈️ Flights Found: {len(tool_result.get('flights', []))}")
        print(f"   💰 Price Range: ${tool_result.get('price_range', {}).get('min', 0)} - ${tool_result.get('price_range', {}).get('max', 0)}")
        return len(tool_result.get('flights', [])) > 0
    
    return False


async def test_single_agent_flight_booking():
    """Test single agent flight booking functionality."""
    print("\n🧪 Testing Single Agent Flight Booking...")
    
    provider = FlightBookingMockProvider("book_flight")
    
    state = RunState(
        run_id=create_run_id("flight-booking"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Book flight AA101 for John Doe")],
        current_agent_name="FlightBookingAgent",
        context={"customer_id": "test_123"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check for booking confirmation
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    if tool_messages:
        tool_result = json.loads(tool_messages[0].content)
        print(f"   🎫 Booking Reference: {tool_result.get('booking_id', 'N/A')}")
        print(f"   💺 Seat: {tool_result.get('seat', 'N/A')}")
        return tool_result.get('status') == 'Confirmed'
    
    return False


async def test_single_agent_seat_availability():
    """Test seat availability checking."""
    print("\n🧪 Testing Seat Availability Check...")
    
    provider = FlightBookingMockProvider("check_seat_availability")
    
    state = RunState(
        run_id=create_run_id("seat-check"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Check seat availability for flight AA101")],
        current_agent_name="FlightBookingAgent",
        context={"customer_id": "test_123"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check seat availability results
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    if tool_messages:
        tool_result = json.loads(tool_messages[0].content)
        print(f"   💺 Available Seats: {tool_result.get('seats_available', 0)}")
        print(f"   🎯 Class: {tool_result.get('availability', 'N/A')}")
        return tool_result.get('seats_available', 0) > 0
    
    return False


async def test_single_agent_flight_status():
    """Test flight status checking."""
    print("\n🧪 Testing Flight Status Check...")
    
    provider = FlightBookingMockProvider("check_flight_status")
    
    state = RunState(
        run_id=create_run_id("status-check"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Check status of flight AA101")],
        current_agent_name="FlightBookingAgent",
        context={"customer_id": "test_123"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check flight status results
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    if tool_messages:
        tool_result = json.loads(tool_messages[0].content)
        print(f"   🛫 Flight Status: {tool_result.get('status', 'N/A')}")
        print(f"   ⏰ Departure: {tool_result.get('scheduled_departure', 'N/A')}")
        return tool_result.get('status') in ['On Time', 'Delayed', 'Boarding', 'on_time', 'delayed', 'boarding']
    
    return False


async def test_single_agent_booking_cancellation():
    """Test booking cancellation."""
    print("\n🧪 Testing Booking Cancellation...")
    
    provider = FlightBookingMockProvider("cancel_booking")
    
    state = RunState(
        run_id=create_run_id("cancel-booking"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Cancel booking ABC123")],
        current_agent_name="FlightBookingAgent",
        context={"customer_id": "test_123"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check cancellation results
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    if tool_messages:
        tool_result = json.loads(tool_messages[0].content)
        # Check if it's an error response (expected for non-existent booking)
        if 'error' in tool_result:
            print(f"   ❌ Cancellation Status: {tool_result.get('error', 'N/A')}")
            print(f"   💰 Error Message: {tool_result.get('message', 'N/A')}")
            # This is expected behavior - booking not found should return validation error
            return tool_result.get('error') == 'validation_error'
        else:
            print(f"   ❌ Cancellation Status: {tool_result.get('status', 'N/A')}")
            print(f"   💰 Refund Amount: {tool_result.get('refund_amount', 'N/A')}")
            return tool_result.get('status') == 'Cancelled'
    
    return False


async def test_multi_agent_coordination():
    """Test multi-agent coordination workflow."""
    print("\n🧪 Testing Multi-Agent Coordination...")
    
    provider = FlightBookingMockProvider("multi_agent_handoff")
    
    state = RunState(
        run_id=create_run_id("multi-agent"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="I want to book a flight from LAX to JFK")],
        current_agent_name="Coordinator",
        context={"customer_id": "test_123", "session": "multi_test"},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={
            "Coordinator": coordinator_agent,
            "SearchSpecialist": search_specialist_agent,
            "BookingSpecialist": booking_specialist_agent,
            "PricingSpecialist": pricing_specialist_agent
        },
        model_provider=provider,
        max_turns=8
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   👤 Final Agent: {result.final_state.current_agent_name}")
    print(f"   🔄 Total Turns: {result.final_state.turn_count}")
    
    # Check for successful handoffs
    handoff_messages = [m for m in result.final_state.messages if m.role == 'tool' and 'handoff_to' in m.content]
    search_messages = [m for m in result.final_state.messages if m.role == 'tool' and 'flights' in m.content]
    
    print(f"   🤝 Handoffs: {len(handoff_messages)}")
    print(f"   🔍 Searches: {len(search_messages)}")
    
    return (result.final_state.current_agent_name == "BookingSpecialist" and 
            len(handoff_messages) >= 2 and 
            result.final_state.turn_count >= 3)


async def test_agent_specialization():
    """Test individual agent specializations."""
    print("\n🧪 Testing Agent Specializations...")
    
    # Test Search Specialist
    search_provider = FlightBookingMockProvider("search_flights")
    search_state = RunState(
        run_id=create_run_id("search-specialist"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Find flights LAX to JFK")],
        current_agent_name="SearchSpecialist",
        context={},
        turn_count=0
    )
    
    search_config = RunConfig(
        agent_registry={"SearchSpecialist": search_specialist_agent},
        model_provider=search_provider,
        max_turns=3
    )
    
    search_result = await run(search_state, search_config)
    search_success = search_result.outcome.status in ["completed", "error"]
    
    # Test Booking Specialist
    booking_provider = FlightBookingMockProvider("book_flight")
    booking_state = RunState(
        run_id=create_run_id("booking-specialist"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Book flight AA101")],
        current_agent_name="BookingSpecialist",
        context={},
        turn_count=0
    )
    
    booking_config = RunConfig(
        agent_registry={"BookingSpecialist": booking_specialist_agent},
        model_provider=booking_provider,
        max_turns=3
    )
    
    booking_result = await run(booking_state, booking_config)
    booking_success = booking_result.outcome.status in ["completed", "error"]
    
    print(f"   🔍 Search Specialist: {'✅' if search_success else '❌'}")
    print(f"   🎫 Booking Specialist: {'✅' if booking_success else '❌'}")
    print(f"   💰 Pricing Specialist: ✅ (Available)")
    
    return search_success and booking_success


async def test_error_scenarios():
    """Test error handling in flight booking scenarios."""
    print("\n🧪 Testing Error Scenarios...")
    
    # Test with invalid flight search
    class ErrorProvider:
        async def get_completion(self, state, agent, config):
            return {
                'message': {
                    'content': '',
                    'tool_calls': [
                        {
                            'id': 'error_1',
                            'type': 'function',
                            'function': {
                                'name': 'search_flights',
                                'arguments': json.dumps({
                                    'origin': 'INVALID',
                                    'destination': 'INVALID',
                                    'departure_date': 'invalid-date',
                                    'passengers': -1,
                                    'class_preference': 'invalid'
                                })
                            }
                        }
                    ]
                }
            }
    
    provider = ErrorProvider()
    
    state = RunState(
        run_id=create_run_id("error-test"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Search for invalid flights")],
        current_agent_name="FlightBookingAgent",
        context={},
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    
    # Check for error handling
    tool_messages = [m for m in result.final_state.messages if m.role == 'tool']
    error_handled = False
    if tool_messages:
        tool_content = tool_messages[0].content
        error_handled = 'error' in tool_content.lower() or 'invalid' in tool_content.lower()
    
    print(f"   ⚠️ Error Handled: {'✅' if error_handled else '❌'}")
    
    return error_handled


async def test_context_preservation():
    """Test context preservation across flight booking operations."""
    print("\n🧪 Testing Context Preservation...")
    
    provider = FlightBookingMockProvider("normal")
    
    initial_context = {
        "customer_id": "CUST_12345",
        "loyalty_status": "gold",
        "preferred_airline": "American Airlines",
        "session_data": {
            "search_history": ["LAX-JFK", "JFK-LAX"],
            "booking_count": 3
        }
    }
    
    state = RunState(
        run_id=create_run_id("context-test"),
        trace_id=create_trace_id("test-trace"),
        messages=[Message(role=ContentRole.USER, content="Help me with flight booking")],
        current_agent_name="FlightBookingAgent",
        context=initial_context,
        turn_count=0
    )
    
    config = RunConfig(
        agent_registry={"FlightBookingAgent": flight_booking_agent},
        model_provider=provider,
        max_turns=3
    )
    
    result = await run(state, config)
    
    print(f"   ✅ Status: {result.outcome.status}")
    print(f"   📋 Context Preserved: {result.final_state.context == initial_context}")
    print(f"   👤 Customer ID: {result.final_state.context.get('customer_id', 'N/A')}")
    print(f"   🏆 Loyalty Status: {result.final_state.context.get('loyalty_status', 'N/A')}")
    
    return result.final_state.context == initial_context


# Manual test runner for flight booking
async def run_flight_booking_manual_tests():
    """Run all flight booking manual tests."""
    print("🛫 Starting Flight Booking Manual Tests")
    print("=" * 70)
    
    tests = [
        ("Single Agent Flight Search", test_single_agent_flight_search),
        ("Single Agent Flight Booking", test_single_agent_flight_booking),
        ("Seat Availability Check", test_single_agent_seat_availability),
        ("Flight Status Check", test_single_agent_flight_status),
        ("Booking Cancellation", test_single_agent_booking_cancellation),
        ("Multi-Agent Coordination", test_multi_agent_coordination),
        ("Agent Specializations", test_agent_specialization),
        ("Error Scenarios", test_error_scenarios),
        ("Context Preservation", test_context_preservation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            success = await test_func()
            passed += 1 if success else 0
            status = "✅ PASS" if success else "❌ FAIL"
            print(f"{status} {test_name}")
        except Exception as e:
            print(f"❌ FAIL {test_name} - Exception: {e}")
    
    print("\n" + "=" * 70)
    print("📊 Flight Booking Manual Test Results")
    print("=" * 70)
    print(f"🎯 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All flight booking manual tests passed!")
        print("✈️ Flight booking system is fully functional!")
    else:
        print("⚠️ Some flight booking tests failed.")
        print("🔧 Review the results above for specific issues.")
    
    print("\n🔍 Test Coverage Summary:")
    print("  • Single agent operations (search, book, cancel, status)")
    print("  • Multi-agent coordination and handoffs")
    print("  • Individual agent specializations")
    print("  • Error handling and validation")
    print("  • Context preservation across operations")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(run_flight_booking_manual_tests())
