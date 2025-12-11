"""Tests for MCP resources."""

import pytest
import json


class TestResourceContent:
    """Tests for ResourceContent."""

    def test_text_content(self):
        """Test text content creation."""
        from ptpd_calibration.mcp.resources import ResourceContent

        content = ResourceContent(
            uri="test://resource",
            mime_type="text/plain",
            text="Hello, World!",
        )

        assert content.uri == "test://resource"
        assert content.mime_type == "text/plain"
        assert content.text == "Hello, World!"

    def test_to_mcp_format_text(self):
        """Test MCP format conversion for text content."""
        from ptpd_calibration.mcp.resources import ResourceContent

        content = ResourceContent(
            uri="test://resource",
            mime_type="application/json",
            text='{"key": "value"}',
        )

        mcp_format = content.to_mcp_format()

        assert mcp_format["uri"] == "test://resource"
        assert mcp_format["mimeType"] == "application/json"
        assert mcp_format["text"] == '{"key": "value"}'

    def test_to_mcp_format_blob(self):
        """Test MCP format conversion for binary content."""
        from ptpd_calibration.mcp.resources import ResourceContent
        import base64

        blob = b"Binary data here"
        content = ResourceContent(
            uri="test://binary",
            mime_type="application/octet-stream",
            blob=blob,
        )

        mcp_format = content.to_mcp_format()

        assert mcp_format["uri"] == "test://binary"
        assert mcp_format["blob"] == base64.b64encode(blob).decode("utf-8")


class TestMCPResource:
    """Tests for MCPResource."""

    def test_to_mcp_format(self):
        """Test MCP format conversion."""
        from ptpd_calibration.mcp.resources import MCPResource

        resource = MCPResource(
            uri="ptpd://test/resource",
            name="Test Resource",
            description="A test resource",
            mime_type="application/json",
        )

        mcp_format = resource.to_mcp_format()

        assert mcp_format["uri"] == "ptpd://test/resource"
        assert mcp_format["name"] == "Test Resource"
        assert mcp_format["description"] == "A test resource"
        assert mcp_format["mimeType"] == "application/json"

    def test_to_mcp_format_with_annotations(self):
        """Test MCP format with annotations."""
        from ptpd_calibration.mcp.resources import MCPResource

        resource = MCPResource(
            uri="ptpd://test/resource",
            name="Test Resource",
            description="A test resource",
            annotations={"priority": "high", "category": "test"},
        )

        mcp_format = resource.to_mcp_format()

        assert mcp_format["annotations"]["priority"] == "high"
        assert mcp_format["annotations"]["category"] == "test"

    @pytest.mark.asyncio
    async def test_read_with_handler(self):
        """Test reading resource with handler."""
        from ptpd_calibration.mcp.resources import MCPResource, ResourceContent

        def my_handler():
            return ResourceContent(
                uri="test://resource",
                mime_type="text/plain",
                text="Handler response",
            )

        resource = MCPResource(
            uri="test://resource",
            name="Test",
            description="Test resource",
            handler=my_handler,
        )

        content = await resource.read()

        assert content.text == "Handler response"

    @pytest.mark.asyncio
    async def test_read_without_handler_raises(self):
        """Test reading resource without handler raises error."""
        from ptpd_calibration.mcp.resources import MCPResource

        resource = MCPResource(
            uri="test://resource",
            name="Test",
            description="No handler",
        )

        with pytest.raises(ValueError, match="No handler defined"):
            await resource.read()

    @pytest.mark.asyncio
    async def test_read_handler_returns_dict(self):
        """Test reading when handler returns dict."""
        from ptpd_calibration.mcp.resources import MCPResource

        def dict_handler():
            return {"key": "value", "count": 42}

        resource = MCPResource(
            uri="test://resource",
            name="Test",
            description="Dict handler",
            handler=dict_handler,
        )

        content = await resource.read()

        assert content.mime_type == "application/json"
        data = json.loads(content.text)
        assert data["key"] == "value"


class TestResourceRegistry:
    """Tests for ResourceRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving resources."""
        from ptpd_calibration.mcp.resources import ResourceRegistry, MCPResource

        registry = ResourceRegistry()

        resource = MCPResource(
            uri="ptpd://test/resource",
            name="Test",
            description="Test resource",
        )

        registry.register(resource)

        retrieved = registry.get("ptpd://test/resource")
        assert retrieved is not None
        assert retrieved.name == "Test"

    def test_get_nonexistent(self):
        """Test getting non-existent resource."""
        from ptpd_calibration.mcp.resources import ResourceRegistry

        registry = ResourceRegistry()

        assert registry.get("nonexistent://uri") is None

    def test_unregister(self):
        """Test unregistering resource."""
        from ptpd_calibration.mcp.resources import ResourceRegistry, MCPResource

        registry = ResourceRegistry()

        resource = MCPResource(
            uri="ptpd://test/resource",
            name="Test",
            description="Test resource",
        )

        registry.register(resource)
        assert registry.get("ptpd://test/resource") is not None

        result = registry.unregister("ptpd://test/resource")
        assert result is True
        assert registry.get("ptpd://test/resource") is None

    def test_template_matching(self):
        """Test template URI matching."""
        from ptpd_calibration.mcp.resources import ResourceRegistry, MCPResource

        registry = ResourceRegistry()

        resource = MCPResource(
            uri="ptpd://calibrations/{id}",
            name="Calibration",
            description="Get calibration by ID",
        )

        registry.register(resource)

        # Should match template
        retrieved = registry.get("ptpd://calibrations/abc-123")
        assert retrieved is not None
        assert retrieved.name == "Calibration"

    def test_list_resources(self):
        """Test listing all resources."""
        from ptpd_calibration.mcp.resources import ResourceRegistry, MCPResource

        registry = ResourceRegistry()

        registry.register(MCPResource(
            uri="ptpd://resource1",
            name="Resource 1",
            description="First resource",
        ))

        registry.register(MCPResource(
            uri="ptpd://resource2",
            name="Resource 2",
            description="Second resource",
        ))

        resources = registry.list_resources()
        assert len(resources) == 2

    def test_to_mcp_format(self):
        """Test converting registry to MCP format."""
        from ptpd_calibration.mcp.resources import ResourceRegistry, MCPResource

        registry = ResourceRegistry()

        registry.register(MCPResource(
            uri="ptpd://resource1",
            name="Resource 1",
            description="First resource",
        ))

        mcp_format = registry.to_mcp_format()

        assert len(mcp_format) == 1
        assert mcp_format[0]["uri"] == "ptpd://resource1"


class TestCreateCalibrationResources:
    """Tests for create_calibration_resources factory."""

    def test_creates_standard_resources(self, resource_registry):
        """Test that standard resources are created."""
        resources = resource_registry.list_resources()

        # Check some expected resources exist
        uris = [r.uri for r in resources]
        assert "ptpd://system/info" in uris
        assert "ptpd://capabilities" in uris
        assert "ptpd://reference/chemistry" in uris
        assert "ptpd://reference/papers" in uris
        assert "ptpd://reference/curve-types" in uris


class TestSystemInfoResource:
    """Tests for system info resource."""

    @pytest.mark.asyncio
    async def test_system_info_content(self, resource_registry):
        """Test system info resource content."""
        resource = resource_registry.get("ptpd://system/info")
        content = await resource.read()

        assert content.mime_type == "application/json"

        data = json.loads(content.text)
        assert data["name"] == "PTPD Calibration System"
        assert "features" in data
        assert "MCP server" in data["features"]


class TestCapabilitiesResource:
    """Tests for capabilities resource."""

    @pytest.mark.asyncio
    async def test_capabilities_content(self, resource_registry):
        """Test capabilities resource content."""
        resource = resource_registry.get("ptpd://capabilities")
        content = await resource.read()

        data = json.loads(content.text)
        assert "detection" in data
        assert "curves" in data
        assert data["curves"]["available"] is True


class TestChemistryReferenceResource:
    """Tests for chemistry reference resource."""

    @pytest.mark.asyncio
    async def test_chemistry_reference_content(self, resource_registry):
        """Test chemistry reference content."""
        resource = resource_registry.get("ptpd://reference/chemistry")
        content = await resource.read()

        data = json.loads(content.text)
        assert "metals" in data
        assert "platinum" in data["metals"]
        assert "palladium" in data["metals"]
        assert "contrast_agents" in data
        assert "developers" in data


class TestPaperProfilesResource:
    """Tests for paper profiles resource."""

    @pytest.mark.asyncio
    async def test_paper_profiles_content(self, resource_registry):
        """Test paper profiles content."""
        resource = resource_registry.get("ptpd://reference/papers")
        content = await resource.read()

        data = json.loads(content.text)
        assert "papers" in data
        assert len(data["papers"]) > 0

        # Check a known paper exists
        paper_names = [p["name"] for p in data["papers"]]
        assert "Arches Platine" in paper_names


class TestCurveTypesResource:
    """Tests for curve types resource."""

    @pytest.mark.asyncio
    async def test_curve_types_content(self, resource_registry):
        """Test curve types content."""
        resource = resource_registry.get("ptpd://reference/curve-types")
        content = await resource.read()

        data = json.loads(content.text)
        assert "curve_types" in data
        assert "interpolation_methods" in data
        assert "export_formats" in data

        # Check expected curve types
        type_names = [t["name"] for t in data["curve_types"]]
        assert "linear" in type_names
        assert "paper_white" in type_names
