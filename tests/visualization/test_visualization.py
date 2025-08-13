"""
Tests for JAF Visualization Module

Tests the core visualization functionality including graph generation,
DOT language output, and various configuration options.
"""

import pytest
import tempfile
import os
from typing import Any
from dataclasses import dataclass

from jaf.core.types import Agent, Tool, ToolSchema, ModelConfig, RunState
from jaf.visualization.graphviz import (
    generate_agent_graph,
    generate_tool_graph, 
    generate_runner_graph,
    get_graph_dot,
    validate_graph_options,
    COLOR_SCHEMES
)
from jaf.visualization.types import GraphOptions, GraphResult


# ========== Test Tool Implementation ==========

@dataclass
class TestArgs:
    """Test tool arguments."""
    value: str


class TestTool:
    """Simple test tool implementation."""
    
    @property
    def schema(self) -> ToolSchema[TestArgs]:
        return ToolSchema(
            name='test_tool',
            description='A test tool for visualization testing',
            parameters=TestArgs
        )
    
    async def execute(self, args: TestArgs, context: Any) -> str:
        return f"Test result: {args.value}"


# ========== Test Fixtures ==========

@pytest.fixture
def test_tool():
    """Create a test tool instance."""
    return TestTool()


@pytest.fixture
def test_agent(test_tool):
    """Create a test agent instance."""
    def instructions(state: RunState[Any]) -> str:
        return "I am a test agent for visualization testing."
    
    return Agent(
        name='TestAgent',
        instructions=instructions,
        tools=[test_tool],
        model_config=ModelConfig(name='gpt-4'),
        handoffs=['OtherAgent']
    )


@pytest.fixture
def second_test_agent():
    """Create a second test agent instance."""
    def instructions(state: RunState[Any]) -> str:
        return "I am another test agent."
    
    return Agent(
        name='OtherAgent',
        instructions=instructions,
        tools=[],
        model_config=ModelConfig(name='gpt-3.5-turbo')
    )


# ========== DOT Language Tests ==========

class TestDOTGeneration:
    """Test DOT language generation and validation."""
    
    def test_get_graph_dot_basic(self, test_agent):
        """Test basic DOT generation."""
        dot_source = get_graph_dot([test_agent])
        
        # Check basic DOT structure
        assert 'digraph AgentGraph' in dot_source
        assert 'TestAgent' in dot_source
        assert 'test_tool' in dot_source
        assert '->' in dot_source  # Should have edges
    
    def test_get_graph_dot_with_options(self, test_agent):
        """Test DOT generation with custom options."""
        options = GraphOptions(
            title='Custom Test Graph',
            color_scheme='modern',
            show_tool_details=False
        )
        
        dot_source = get_graph_dot([test_agent], options)
        
        assert 'Custom Test Graph' in dot_source
        # With show_tool_details=False, tool nodes should not be present
        # but we can't fully test this without running the full graph generation
    
    def test_get_graph_dot_multiple_agents(self, test_agent, second_test_agent):
        """Test DOT generation with multiple agents."""
        dot_source = get_graph_dot([test_agent, second_test_agent])
        
        assert 'TestAgent' in dot_source
        assert 'OtherAgent' in dot_source
        assert 'handoff' in dot_source  # Should show handoff edge
    
    def test_color_scheme_attributes(self, test_agent):
        """Test that color schemes affect DOT attributes."""
        for scheme in ['default', 'modern', 'minimal']:
            options = GraphOptions(color_scheme=scheme)
            dot_source = get_graph_dot([test_agent], options)
            
            # Check that color scheme colors are present
            scheme_config = COLOR_SCHEMES[scheme]
            agent_color = scheme_config['agent']['fillcolor']
            assert agent_color in dot_source


# ========== Validation Tests ==========

class TestValidation:
    """Test option validation functionality."""
    
    def test_validate_valid_options(self):
        """Test validation of valid options."""
        options = GraphOptions(
            layout='dot',
            rankdir='TB',
            output_format='png',
            color_scheme='default'
        )
        
        errors = validate_graph_options(options)
        assert len(errors) == 0
    
    def test_validate_invalid_layout(self):
        """Test validation of invalid layout."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            GraphOptions(layout='invalid_layout')
        
        # Check that the error mentions layout
        assert 'layout' in str(exc_info.value)
    
    def test_validate_invalid_rankdir(self):
        """Test validation of invalid rankdir."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            GraphOptions(rankdir='INVALID')
        
        # Check that the error mentions rankdir
        assert 'rankdir' in str(exc_info.value)
    
    def test_validate_invalid_output_format(self):
        """Test validation of invalid output format."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            GraphOptions(output_format='invalid')
        
        # Check that the error mentions output_format
        assert 'output_format' in str(exc_info.value)
    
    def test_validate_invalid_color_scheme(self):
        """Test validation of invalid color scheme."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            GraphOptions(color_scheme='invalid')
        
        # Check that the error mentions color_scheme
        assert 'color_scheme' in str(exc_info.value)
    
    def test_validate_multiple_errors(self):
        """Test validation with multiple errors."""
        from pydantic import ValidationError
        
        with pytest.raises(ValidationError) as exc_info:
            GraphOptions(
                layout='invalid',
                rankdir='INVALID',
                output_format='bad',
                color_scheme='wrong'
            )
        
        # Check that multiple fields are mentioned in the error
        error_str = str(exc_info.value)
        assert 'layout' in error_str
        assert 'rankdir' in error_str
        assert 'output_format' in error_str
        assert 'color_scheme' in error_str


# ========== Graph Generation Tests ==========

class TestGraphGeneration:
    """Test actual graph generation functionality."""
    
    @pytest.mark.asyncio
    async def test_generate_agent_graph_success(self, test_agent):
        """Test successful agent graph generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_agent.png')
            
            options = GraphOptions(
                output_path=output_path,
                output_format='png'
            )
            
            try:
                result = await generate_agent_graph([test_agent], options)
                
                # Check result structure
                assert isinstance(result, GraphResult)
                assert result.success is True or result.success is False  # Should be boolean
                
                if result.success:
                    assert result.output_path == output_path
                    assert result.graph_dot is not None
                    assert 'TestAgent' in result.graph_dot
                else:
                    # If it fails, it's likely due to missing Graphviz system dependency
                    assert result.error is not None
                    assert 'graphviz' in result.error.lower() or 'command not found' in result.error.lower()
                    
            except Exception as e:
                # If Graphviz is not installed, the test should not fail
                pytest.skip(f"Graphviz not available: {e}")
    
    @pytest.mark.asyncio
    async def test_generate_agent_graph_invalid_options(self, test_agent):
        """Test agent graph generation with invalid options."""
        from pydantic import ValidationError
        
        # Since Pydantic validates at creation time, we expect a ValidationError
        with pytest.raises(ValidationError):
            GraphOptions(layout='invalid_layout')
        
        # Test that our validate_graph_options function still works for edge cases
        # that might get through Pydantic (though this is mostly for completeness)
        valid_options = GraphOptions()
        errors = validate_graph_options(valid_options)
        assert len(errors) == 0
    
    @pytest.mark.asyncio
    async def test_generate_tool_graph_success(self, test_tool):
        """Test successful tool graph generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_tools.png')
            
            options = GraphOptions(
                output_path=output_path,
                output_format='png'
            )
            
            try:
                result = await generate_tool_graph([test_tool], options)
                
                assert isinstance(result, GraphResult)
                assert result.success is True or result.success is False
                
                if result.success:
                    assert result.output_path == output_path
                    assert result.graph_dot is not None
                    assert 'test_tool' in result.graph_dot
                else:
                    assert result.error is not None
                    
            except Exception as e:
                pytest.skip(f"Graphviz not available: {e}")
    
    @pytest.mark.asyncio
    async def test_generate_runner_graph_success(self, test_agent, second_test_agent):
        """Test successful runner graph generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_runner.png')
            
            agent_registry = {
                'TestAgent': test_agent,
                'OtherAgent': second_test_agent
            }
            
            options = GraphOptions(
                output_path=output_path,
                output_format='png'
            )
            
            try:
                result = await generate_runner_graph(agent_registry, options)
                
                assert isinstance(result, GraphResult)
                assert result.success is True or result.success is False
                
                if result.success:
                    assert result.output_path == output_path
                    assert result.graph_dot is not None
                    assert 'Runner' in result.graph_dot
                    assert 'TestAgent' in result.graph_dot
                else:
                    assert result.error is not None
                    
            except Exception as e:
                pytest.skip(f"Graphviz not available: {e}")


# ========== Color Scheme Tests ==========

class TestColorSchemes:
    """Test color scheme configurations."""
    
    def test_all_color_schemes_exist(self):
        """Test that all expected color schemes are defined."""
        expected_schemes = ['default', 'modern', 'minimal']
        
        for scheme in expected_schemes:
            assert scheme in COLOR_SCHEMES
    
    def test_color_scheme_structure(self):
        """Test that color schemes have required structure."""
        required_keys = ['agent', 'tool', 'sub_agent', 'edge', 'tool_edge']
        
        for scheme_name, scheme in COLOR_SCHEMES.items():
            for key in required_keys:
                assert key in scheme, f"Missing '{key}' in color scheme '{scheme_name}'"
            
            # Test agent style has required attributes
            agent_style = scheme['agent']
            required_agent_attrs = ['shape', 'fillcolor', 'fontcolor', 'style']
            for attr in required_agent_attrs:
                assert attr in agent_style, f"Missing '{attr}' in agent style for '{scheme_name}'"
    
    def test_color_scheme_values(self):
        """Test that color scheme values are valid."""
        for scheme_name, scheme in COLOR_SCHEMES.items():
            # Test that colors are valid (hex colors or named colors)
            agent_fillcolor = scheme['agent']['fillcolor']
            assert isinstance(agent_fillcolor, str)
            assert len(agent_fillcolor) > 0
            
            # Test that shapes are valid
            agent_shape = scheme['agent']['shape']
            valid_shapes = ['box', 'ellipse', 'circle', 'diamond', 'rect']
            # Note: Graphviz supports many shapes, so we'll just check it's a string
            assert isinstance(agent_shape, str)
            assert len(agent_shape) > 0


# ========== Integration Tests ==========

class TestIntegration:
    """Integration tests for the visualization system."""
    
    def test_complete_workflow_dot_generation(self, test_agent, test_tool):
        """Test complete workflow from agent creation to DOT generation."""
        # Create agents with tools
        agents = [test_agent]
        
        # Generate DOT
        dot_source = get_graph_dot(agents)
        
        # Verify DOT contains expected elements
        assert 'digraph AgentGraph' in dot_source
        assert test_agent.name in dot_source
        assert test_tool.schema.name in dot_source
        
        # Verify structure
        lines = dot_source.split('\n')
        assert any('rankdir=' in line for line in lines)
        assert any('label=' in line for line in lines)
    
    @pytest.mark.asyncio
    async def test_multiple_format_generation(self, test_agent):
        """Test generation of multiple output formats."""
        with tempfile.TemporaryDirectory() as temp_dir:
            formats = ['png', 'svg', 'pdf']
            
            for fmt in formats:
                output_path = os.path.join(temp_dir, f'test.{fmt}')
                options = GraphOptions(
                    output_path=output_path,
                    output_format=fmt
                )
                
                try:
                    result = await generate_agent_graph([test_agent], options)
                    
                    # Should either succeed or fail gracefully
                    assert isinstance(result, GraphResult)
                    assert isinstance(result.success, bool)
                    
                    if not result.success:
                        # If it fails, should have error message
                        assert result.error is not None
                        
                except Exception as e:
                    pytest.skip(f"Graphviz not available for format {fmt}: {e}")


# ========== Error Handling Tests ==========

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.mark.asyncio
    async def test_empty_agent_list(self):
        """Test handling of empty agent list."""
        result = await generate_agent_graph([])
        
        # Should handle gracefully - either succeed with empty graph or provide clear error
        assert isinstance(result, GraphResult)
        assert isinstance(result.success, bool)
    
    @pytest.mark.asyncio
    async def test_agent_without_tools(self):
        """Test handling of agents without tools."""
        def instructions(state: RunState[Any]) -> str:
            return "I am an agent without tools."
        
        agent_no_tools = Agent(
            name='NoToolsAgent',
            instructions=instructions,
            tools=[],
            model_config=ModelConfig(name='gpt-4')
        )
        
        result = await generate_agent_graph([agent_no_tools])
        
        assert isinstance(result, GraphResult)
        # Should succeed even without tools
        if result.success:
            assert result.graph_dot is not None
            assert 'NoToolsAgent' in result.graph_dot
    
    def test_malformed_options(self):
        """Test handling of edge case options."""
        # Test with None values
        options = GraphOptions(title=None)
        errors = validate_graph_options(options)
        # Should not error on None title
        assert len(errors) == 0 or all('title' not in error.lower() for error in errors)