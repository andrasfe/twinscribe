"""Unit tests for twinscribe.models.components module.

Tests cover:
- ComponentLocation model validation and methods
- ParameterDoc model validation
- ReturnDoc and ExceptionDoc models
- ComponentDocumentation model
- Component model with all edge cases
"""

from datetime import datetime

import pytest
from pydantic import ValidationError

from twinscribe.models.base import ComponentType
from twinscribe.models.components import (
    Component,
    ComponentDocumentation,
    ComponentLocation,
    ExceptionDoc,
    ParameterDoc,
    ReturnDoc,
)


class TestComponentLocation:
    """Tests for ComponentLocation model."""

    def test_valid_location(self):
        """Test creating a valid component location."""
        loc = ComponentLocation(
            file_path="src/utils/helpers.py",
            line_start=10,
            line_end=25,
        )
        assert loc.file_path == "src/utils/helpers.py"
        assert loc.line_start == 10
        assert loc.line_end == 25
        assert loc.column_start is None
        assert loc.column_end is None

    def test_location_with_columns(self):
        """Test location with column information."""
        loc = ComponentLocation(
            file_path="main.py",
            line_start=1,
            line_end=1,
            column_start=0,
            column_end=50,
        )
        assert loc.column_start == 0
        assert loc.column_end == 50

    def test_line_end_must_be_after_start(self):
        """Test that line_end >= line_start is enforced."""
        with pytest.raises(ValidationError) as exc_info:
            ComponentLocation(
                file_path="test.py",
                line_start=50,
                line_end=10,
            )
        assert "line_end must be >= line_start" in str(exc_info.value)

    def test_line_numbers_must_be_positive(self):
        """Test that line numbers must be >= 1."""
        with pytest.raises(ValidationError):
            ComponentLocation(
                file_path="test.py",
                line_start=0,
                line_end=5,
            )

    def test_to_reference_method(self):
        """Test the to_reference() method."""
        loc = ComponentLocation(
            file_path="src/module.py",
            line_start=100,
            line_end=150,
        )
        assert loc.to_reference() == "src/module.py:100-150"

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        loc = ComponentLocation(
            file_path="test.py",
            line_start=1,
            line_end=10,
            column_start=0,
            column_end=80,
        )
        json_str = loc.model_dump_json()
        restored = ComponentLocation.model_validate_json(json_str)
        assert restored == loc


class TestParameterDoc:
    """Tests for ParameterDoc model."""

    def test_valid_parameter(self):
        """Test creating a valid parameter doc."""
        param = ParameterDoc(
            name="value",
            type="str",
            description="The input value",
            default="None",
            required=False,
        )
        assert param.name == "value"
        assert param.type == "str"
        assert param.description == "The input value"
        assert param.default == "None"
        assert param.required is False

    def test_minimal_parameter(self):
        """Test parameter with only required fields."""
        param = ParameterDoc(name="x")
        assert param.name == "x"
        assert param.type is None
        assert param.description == ""
        assert param.default is None
        assert param.required is True

    def test_args_and_kwargs_names(self):
        """Test that *args and **kwargs are valid names."""
        args_param = ParameterDoc(name="*args")
        kwargs_param = ParameterDoc(name="**kwargs")
        assert args_param.name == "*args"
        assert kwargs_param.name == "**kwargs"

    def test_empty_name_rejected(self):
        """Test that empty parameter name is rejected."""
        with pytest.raises(ValidationError):
            ParameterDoc(name="")

    def test_invalid_identifier_name(self):
        """Test that invalid Python identifiers are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ParameterDoc(name="123invalid")
        assert "not a valid parameter name" in str(exc_info.value)

    def test_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        param = ParameterDoc(
            name="data",
            type="dict[str, Any]",
            description="Input data",
            default="{}",
            required=True,
        )
        json_str = param.model_dump_json()
        restored = ParameterDoc.model_validate_json(json_str)
        assert restored == param


class TestReturnDoc:
    """Tests for ReturnDoc model."""

    def test_valid_return_doc(self):
        """Test creating a valid return doc."""
        ret = ReturnDoc(
            type="str",
            description="The processed result",
        )
        assert ret.type == "str"
        assert ret.description == "The processed result"

    def test_minimal_return_doc(self):
        """Test return doc with defaults."""
        ret = ReturnDoc()
        assert ret.type is None
        assert ret.description == ""

    def test_complex_type(self):
        """Test with complex type annotation."""
        ret = ReturnDoc(
            type="dict[str, list[tuple[int, str]]]",
            description="Nested data structure",
        )
        assert "dict" in ret.type


class TestExceptionDoc:
    """Tests for ExceptionDoc model."""

    def test_valid_exception_doc(self):
        """Test creating a valid exception doc."""
        exc = ExceptionDoc(
            type="ValueError",
            condition="When input is negative",
        )
        assert exc.type == "ValueError"
        assert exc.condition == "When input is negative"

    def test_empty_type_rejected(self):
        """Test that empty exception type is rejected."""
        with pytest.raises(ValidationError):
            ExceptionDoc(type="")

    def test_minimal_exception_doc(self):
        """Test exception doc with only type."""
        exc = ExceptionDoc(type="RuntimeError")
        assert exc.type == "RuntimeError"
        assert exc.condition == ""


class TestComponentDocumentation:
    """Tests for ComponentDocumentation model."""

    def test_complete_documentation(self):
        """Test creating complete documentation."""
        doc = ComponentDocumentation(
            summary="Process input data.",
            description="Detailed description of the processing logic.",
            parameters=[
                ParameterDoc(name="data", type="dict", description="Input"),
                ParameterDoc(name="timeout", type="float", default="30.0"),
            ],
            returns=ReturnDoc(type="bool", description="Success status"),
            raises=[
                ExceptionDoc(type="ValueError", condition="Invalid input"),
                ExceptionDoc(type="TimeoutError", condition="Timeout exceeded"),
            ],
            examples=["result = process({'key': 'value'})"],
            notes="This function is thread-safe.",
            see_also=["other_function", "related_module"],
        )
        assert doc.summary == "Process input data."
        assert len(doc.parameters) == 2
        assert doc.returns is not None
        assert len(doc.raises) == 2
        assert len(doc.examples) == 1
        assert doc.notes is not None
        assert len(doc.see_also) == 2

    def test_empty_documentation(self):
        """Test documentation with all defaults."""
        doc = ComponentDocumentation()
        assert doc.summary == ""
        assert doc.description == ""
        assert doc.parameters == []
        assert doc.returns is None
        assert doc.raises == []
        assert doc.examples == []
        assert doc.notes is None
        assert doc.see_also == []

    def test_summary_max_length(self):
        """Test that summary has a max length."""
        long_summary = "x" * 201
        with pytest.raises(ValidationError):
            ComponentDocumentation(summary=long_summary)

    def test_json_roundtrip(self):
        """Test JSON serialization with nested models."""
        doc = ComponentDocumentation(
            summary="Test function",
            parameters=[ParameterDoc(name="arg1", type="int")],
            returns=ReturnDoc(type="str"),
            raises=[ExceptionDoc(type="ValueError")],
        )
        json_str = doc.model_dump_json()
        restored = ComponentDocumentation.model_validate_json(json_str)
        assert restored.summary == doc.summary
        assert len(restored.parameters) == 1
        assert restored.returns is not None


class TestComponent:
    """Tests for Component model."""

    def test_valid_component(self):
        """Test creating a valid component."""
        comp = Component(
            component_id="mypackage.utils.helper",
            name="helper",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="src/utils.py",
                line_start=10,
                line_end=20,
            ),
        )
        assert comp.component_id == "mypackage.utils.helper"
        assert comp.name == "helper"
        assert comp.type == ComponentType.FUNCTION
        assert comp.is_public is True

    def test_component_with_all_fields(self):
        """Test component with all optional fields."""
        comp = Component(
            component_id="pkg.Module.MyClass.method",
            name="method",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="module.py",
                line_start=50,
                line_end=75,
            ),
            signature="def method(self, arg: str) -> int:",
            parent_id="pkg.Module.MyClass",
            dependencies=["pkg.other", "pkg.utils"],
            existing_docstring="Original docstring.",
            is_public=True,
        )
        assert comp.parent_id == "pkg.Module.MyClass"
        assert len(comp.dependencies) == 2
        assert comp.existing_docstring == "Original docstring."

    def test_private_component_detection(self):
        """Test that private components are detected by name."""
        comp = Component(
            component_id="pkg._private_helper",
            name="_private_helper",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="pkg.py",
                line_start=1,
                line_end=5,
            ),
        )
        assert comp.is_public is False

    def test_module_path_extraction(self):
        """Test the module_path property."""
        comp = Component(
            component_id="mypackage.submodule.ClassName.method",
            name="method",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="test.py",
                line_start=1,
                line_end=1,
            ),
        )
        assert comp.module_path == "mypackage.submodule.ClassName"

    def test_single_segment_component_id(self):
        """Test component with single-segment ID."""
        comp = Component(
            component_id="standalone",
            name="standalone",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="main.py",
                line_start=1,
                line_end=10,
            ),
        )
        assert comp.module_path == ""

    def test_empty_component_id_rejected(self):
        """Test that empty component_id is rejected."""
        with pytest.raises(ValidationError):
            Component(
                component_id="",
                name="test",
                type=ComponentType.FUNCTION,
                location=ComponentLocation(
                    file_path="test.py",
                    line_start=1,
                    line_end=1,
                ),
            )

    def test_created_at_default(self):
        """Test that created_at defaults to current time."""
        before = datetime.utcnow()
        comp = Component(
            component_id="test.func",
            name="func",
            type=ComponentType.FUNCTION,
            location=ComponentLocation(
                file_path="test.py",
                line_start=1,
                line_end=1,
            ),
        )
        after = datetime.utcnow()
        assert before <= comp.created_at <= after

    def test_all_component_types(self):
        """Test all component types can be used."""
        for comp_type in ComponentType:
            comp = Component(
                component_id=f"test.{comp_type.value}",
                name=comp_type.value,
                type=comp_type,
                location=ComponentLocation(
                    file_path="test.py",
                    line_start=1,
                    line_end=1,
                ),
            )
            assert comp.type == comp_type

    def test_json_serialization(self):
        """Test full JSON serialization."""
        comp = Component(
            component_id="pkg.mod.Class.method",
            name="method",
            type=ComponentType.METHOD,
            location=ComponentLocation(
                file_path="src/pkg/mod.py",
                line_start=100,
                line_end=150,
            ),
            parent_id="pkg.mod.Class",
            dependencies=["pkg.utils"],
        )
        json_str = comp.model_dump_json()
        restored = Component.model_validate_json(json_str)
        assert restored.component_id == comp.component_id
        assert restored.type == comp.type
        assert restored.location.file_path == comp.location.file_path
