"""Integration tests for MCP server and LM Studio connectivity.

These tests verify the integration between the MCP server and LM Studio.
They require either a running LM Studio instance or use mocks for CI.
"""

import json
import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# Mark all tests in this module as integration and mcp tests
pytestmark = [pytest.mark.integration, pytest.mark.mcp]


class TestLMStudioIntegration:
    """Integration tests for LM Studio client."""

    @pytest.mark.asyncio
    async def test_client_lifecycle(self, lm_studio_settings, mock_httpx_client):
        """Test client creation, usage, and cleanup lifecycle."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        async with LMStudioClient(settings=lm_studio_settings) as client:
            client._http_client = mock_httpx_client

            # Test completion
            result = await client.complete([
                {"role": "user", "content": "Test message"}
            ])

            assert result.content == "Test response content"

        # Client should be cleaned up after context exit

    @pytest.mark.asyncio
    async def test_complete_with_all_parameters(self, lm_studio_settings, mock_httpx_client):
        """Test completion with all optional parameters."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient

        client = LMStudioClient(settings=lm_studio_settings)
        client._http_client = mock_httpx_client

        result = await client.complete(
            messages=[
                {"role": "user", "content": "Analyze this data"},
            ],
            system="You are a helpful calibration assistant.",
            max_tokens=2048,
            temperature=0.3,
            top_p=0.95,
        )

        assert result.content == "Test response content"

        # Verify request was made with correct parameters
        call_args = mock_httpx_client.post.call_args
        payload = call_args.kwargs.get("json", {})

        assert payload["max_tokens"] == 2048
        assert payload["temperature"] == 0.3
        assert payload["top_p"] == 0.95
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "system"

    @pytest.mark.asyncio
    async def test_retry_on_transient_failure(self, lm_studio_settings):
        """Test retry logic on transient failures."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient, LMStudioConnectionError
        import httpx

        # Create client with fast retry for testing
        lm_studio_settings.max_retries = 2
        lm_studio_settings.retry_delay_seconds = 0.01

        client = LMStudioClient(settings=lm_studio_settings)

        # Mock that fails first, then succeeds
        call_count = 0

        async def mock_request(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise httpx.ConnectError("Connection failed")
            return MagicMock(
                status_code=200,
                json=lambda: {
                    "choices": [{"message": {"content": "Success after retry"}}],
                    "model": "test",
                },
            )

        mock_client = AsyncMock()
        mock_client.post = mock_request
        client._http_client = mock_client

        result = await client.complete([{"role": "user", "content": "Test"}])

        assert result.content == "Success after retry"
        assert call_count == 2  # First failed, second succeeded


class TestMCPServerIntegration:
    """Integration tests for MCP server with tools and resources."""

    @pytest.mark.asyncio
    async def test_full_workflow_analyze_and_generate(
        self, mcp_server, sample_densities
    ):
        """Test a full workflow: analyze densities, then generate curve."""
        # Step 1: Analyze densities
        analyze_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "analyze_densities",
                "arguments": {"densities": sample_densities},
            },
        })

        response = await mcp_server.handle_message(analyze_msg)
        result = json.loads(response)
        analysis = json.loads(result["result"]["content"][0]["text"])

        assert analysis["is_monotonic"] is True
        # Quality depends on linearity - monotonic data can still be "fair" if deviation is high
        assert analysis["quality"] in ["good", "fair"]

        # Step 2: Generate linearization curve
        generate_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "generate_linearization_curve",
                "arguments": {
                    "densities": sample_densities,
                    "curve_name": "Test Print Curve",
                    "curve_type": "linear",
                },
            },
        })

        response = await mcp_server.handle_message(generate_msg)
        result = json.loads(response)
        curve = json.loads(result["result"]["content"][0]["text"])

        assert curve["name"] == "Test Print Curve"
        assert len(curve["output_values"]) == 256

    @pytest.mark.asyncio
    async def test_resource_and_tool_combination(self, mcp_server):
        """Test using resources to get info, then calling tools."""
        # Get chemistry reference
        ref_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "ptpd://reference/chemistry"},
        })

        response = await mcp_server.handle_message(ref_msg)
        result = json.loads(response)
        chemistry = json.loads(result["result"]["contents"][0]["text"])

        assert "metals" in chemistry
        assert "platinum" in chemistry["metals"]

        # Calculate chemistry for a print
        calc_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "calculate_chemistry",
                "arguments": {
                    "width_inches": 8,
                    "height_inches": 10,
                    "metal_ratio": 0.6,  # 60% platinum
                    "contrast_drops": 3,
                },
            },
        })

        response = await mcp_server.handle_message(calc_msg)
        result = json.loads(response)
        calculation = json.loads(result["result"]["content"][0]["text"])

        assert calculation["metal_ratio"]["platinum_percent"] == 60.0
        assert calculation["drops"]["contrast_agent"] == 3

    @pytest.mark.asyncio
    async def test_server_initialization_sequence(self, mcp_server):
        """Test proper server initialization sequence."""
        # Initialize
        init_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0.0",
                },
            },
        })

        response = await mcp_server.handle_message(init_msg)
        result = json.loads(response)

        assert "serverInfo" in result["result"]
        assert "capabilities" in result["result"]
        assert mcp_server._initialized is True

        # List tools
        tools_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
        })

        response = await mcp_server.handle_message(tools_msg)
        result = json.loads(response)

        assert len(result["result"]["tools"]) > 0

        # List resources
        resources_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "resources/list",
        })

        response = await mcp_server.handle_message(resources_msg)
        result = json.loads(response)

        assert len(result["result"]["resources"]) > 0


class TestEndToEndWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.asyncio
    async def test_calibration_workflow(self, mcp_server, sample_densities):
        """Test complete calibration workflow from analysis to curve export."""
        # 1. Get system info
        info_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "ptpd://system/info"},
        })
        response = await mcp_server.handle_message(info_msg)
        info = json.loads(response)
        assert "PTPD Calibration System" in info["result"]["contents"][0]["text"]

        # 2. Analyze densities
        analyze_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "analyze_densities",
                "arguments": {"densities": sample_densities},
            },
        })
        response = await mcp_server.handle_message(analyze_msg)
        analysis_result = json.loads(response)
        analysis = json.loads(analysis_result["result"]["content"][0]["text"])
        assert analysis["quality"] in ["good", "fair", "poor"]

        # 3. Generate curve
        generate_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "generate_linearization_curve",
                "arguments": {
                    "densities": sample_densities,
                    "curve_name": "Workflow Test",
                    "curve_type": "linear",
                    "interpolation": "pchip",
                },
            },
        })
        response = await mcp_server.handle_message(generate_msg)
        curve_result = json.loads(response)
        curve = json.loads(curve_result["result"]["content"][0]["text"])

        # 4. Format as QTR
        qtr_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "format_qtr_curve",
                "arguments": {
                    "input_values": curve["input_values"],
                    "output_values": curve["output_values"],
                    "curve_name": curve["name"],
                },
            },
        })
        response = await mcp_server.handle_message(qtr_msg)
        qtr_result = json.loads(response)
        qtr_content = qtr_result["result"]["content"][0]["text"]

        assert "[Gray]" in qtr_content
        assert "Workflow Test" in qtr_content

    @pytest.mark.asyncio
    async def test_exposure_planning_workflow(self, mcp_server):
        """Test exposure planning workflow."""
        # 1. Get paper profiles
        papers_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "resources/read",
            "params": {"uri": "ptpd://reference/papers"},
        })
        response = await mcp_server.handle_message(papers_msg)
        papers = json.loads(response)
        paper_data = json.loads(papers["result"]["contents"][0]["text"])

        # Get first paper name
        paper_name = paper_data["papers"][0]["name"]

        # 2. Suggest exposure
        exposure_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "suggest_exposure",
                "arguments": {
                    "paper_type": paper_name,
                    "uv_source": "led",
                    "metal_ratio": 0.5,
                },
            },
        })
        response = await mcp_server.handle_message(exposure_msg)
        exposure_result = json.loads(response)
        exposure = json.loads(exposure_result["result"]["content"][0]["text"])

        assert "suggested_exposure_seconds" in exposure
        assert len(exposure["exposure_bracket"]) == 5

        # 3. Calculate chemistry
        chem_msg = json.dumps({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "calculate_chemistry",
                "arguments": {
                    "width_inches": 11,
                    "height_inches": 14,
                    "metal_ratio": 0.5,
                },
            },
        })
        response = await mcp_server.handle_message(chem_msg)
        chem_result = json.loads(response)
        chemistry = json.loads(chem_result["result"]["content"][0]["text"])

        assert "volumes_ml" in chemistry
        assert "drops" in chemistry


@pytest.mark.skipif(
    os.environ.get("LM_STUDIO_HOST") is None,
    reason="LM Studio not configured - set LM_STUDIO_HOST to run"
)
class TestLiveIntegration:
    """Live integration tests requiring a running LM Studio instance.

    These tests are skipped by default and only run when LM_STUDIO_HOST
    environment variable is set.
    """

    @pytest.mark.asyncio
    async def test_live_connection(self):
        """Test live connection to LM Studio."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient
        from ptpd_calibration.mcp.config import LMStudioSettings

        host = os.environ.get("LM_STUDIO_HOST", "localhost")
        port = int(os.environ.get("LM_STUDIO_PORT", "1234"))

        settings = LMStudioSettings(host=host, port=port)
        client = LMStudioClient(settings=settings)

        is_healthy = await client.health_check()
        assert is_healthy is True

        await client.close()

    @pytest.mark.asyncio
    async def test_live_models_list(self):
        """Test listing models from live LM Studio."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient
        from ptpd_calibration.mcp.config import LMStudioSettings

        host = os.environ.get("LM_STUDIO_HOST", "localhost")
        port = int(os.environ.get("LM_STUDIO_PORT", "1234"))

        settings = LMStudioSettings(host=host, port=port)
        client = LMStudioClient(settings=settings)

        models = await client.list_models()
        assert len(models) > 0

        await client.close()

    @pytest.mark.asyncio
    async def test_live_completion(self):
        """Test completion with live LM Studio."""
        from ptpd_calibration.mcp.lm_studio import LMStudioClient
        from ptpd_calibration.mcp.config import LMStudioSettings

        host = os.environ.get("LM_STUDIO_HOST", "localhost")
        port = int(os.environ.get("LM_STUDIO_PORT", "1234"))

        settings = LMStudioSettings(
            host=host,
            port=port,
            max_tokens=100,
            temperature=0.0,  # Deterministic for testing
        )
        client = LMStudioClient(settings=settings)

        result = await client.complete([
            {"role": "user", "content": "Reply with just the word 'hello'"},
        ])

        assert len(result.content) > 0
        assert result.model is not None

        await client.close()
