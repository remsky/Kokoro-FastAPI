"""
Pinecone Control Plane API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-04
Contact: support@pinecone.io
"""

from pinecone.openapi_support.model_utils import (  # noqa: F401
    PineconeApiTypeError,
    ModelComposed,
    ModelNormal,
    ModelSimple,
    OpenApiModel,
    cached_property,
    change_keys_js_to_python,
    convert_js_args_to_python_args,
    date,
    datetime,
    file_type,
    none_type,
    validate_get_composed_info,
)
from pinecone.openapi_support.exceptions import PineconeApiAttributeError


def lazy_import():
    from pinecone.core.openapi.db_control.model.index_tags import IndexTags

    globals()["IndexTags"] = IndexTags


from typing import Dict, Literal, Tuple, Set, Any, Type, TypeVar
from pinecone.openapi_support import PropertyValidationTypedDict, cached_class_property

T = TypeVar("T", bound="BackupModel")


class BackupModel(ModelNormal):
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.

    Attributes:
      allowed_values (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          with a capitalized key describing the allowed value and an allowed
          value. These dicts store the allowed enum values.
      attribute_map (dict): The key is attribute name
          and the value is json key in definition.
      discriminator_value_class_map (dict): A dict to go from the discriminator
          variable value to the discriminator class name.
      validations (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          that stores validations for max_length, min_length, max_items,
          min_items, exclusive_maximum, inclusive_maximum, exclusive_minimum,
          inclusive_minimum, and regex.
      additional_properties_type (tuple): A tuple of classes accepted
          as additional properties values.
    """

    _data_store: Dict[str, Any]
    _check_type: bool

    allowed_values: Dict[Tuple[str, ...], Dict[str, Any]] = {
        ("metric",): {"COSINE": "cosine", "EUCLIDEAN": "euclidean", "DOTPRODUCT": "dotproduct"}
    }

    validations: Dict[Tuple[str, ...], PropertyValidationTypedDict] = {
        ("dimension",): {"inclusive_maximum": 20000, "inclusive_minimum": 1}
    }

    @cached_class_property
    def additional_properties_type(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
        lazy_import()
        return (bool, dict, float, int, list, str, none_type)  # noqa: E501

    _nullable = False

    @cached_class_property
    def openapi_types(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded

        Returns
            openapi_types (dict): The key is attribute name
                and the value is attribute type.
        """
        lazy_import()
        return {
            "backup_id": (str,),  # noqa: E501
            "source_index_name": (str,),  # noqa: E501
            "source_index_id": (str,),  # noqa: E501
            "status": (str,),  # noqa: E501
            "cloud": (str,),  # noqa: E501
            "region": (str,),  # noqa: E501
            "name": (str,),  # noqa: E501
            "description": (str,),  # noqa: E501
            "dimension": (int,),  # noqa: E501
            "metric": (str,),  # noqa: E501
            "record_count": (int,),  # noqa: E501
            "namespace_count": (int,),  # noqa: E501
            "size_bytes": (int,),  # noqa: E501
            "tags": (IndexTags,),  # noqa: E501
            "created_at": (str,),  # noqa: E501
        }

    @cached_class_property
    def discriminator(cls):
        return None

    attribute_map: Dict[str, str] = {
        "backup_id": "backup_id",  # noqa: E501
        "source_index_name": "source_index_name",  # noqa: E501
        "source_index_id": "source_index_id",  # noqa: E501
        "status": "status",  # noqa: E501
        "cloud": "cloud",  # noqa: E501
        "region": "region",  # noqa: E501
        "name": "name",  # noqa: E501
        "description": "description",  # noqa: E501
        "dimension": "dimension",  # noqa: E501
        "metric": "metric",  # noqa: E501
        "record_count": "record_count",  # noqa: E501
        "namespace_count": "namespace_count",  # noqa: E501
        "size_bytes": "size_bytes",  # noqa: E501
        "tags": "tags",  # noqa: E501
        "created_at": "created_at",  # noqa: E501
    }

    read_only_vars: Set[str] = set([])

    _composed_schemas: Dict[Literal["allOf", "oneOf", "anyOf"], Any] = {}

    @classmethod
    @convert_js_args_to_python_args
    def _from_openapi_data(
        cls: Type[T],
        backup_id,
        source_index_name,
        source_index_id,
        status,
        cloud,
        region,
        *args,
        **kwargs,
    ) -> T:  # noqa: E501
        """BackupModel - a model defined in OpenAPI

        Args:
            backup_id (str): Unique identifier for the backup.
            source_index_name (str): Name of the index from which the backup was taken.
            source_index_id (str): ID of the index.
            status (str): Current status of the backup (e.g., Initializing, Ready, Failed).
            cloud (str): Cloud provider where the backup is stored.
            region (str): Cloud region where the backup is stored.

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            name (str): Optional user-defined name for the backup. [optional]  # noqa: E501
            description (str): Optional description providing context for the backup. [optional]  # noqa: E501
            dimension (int): The dimensions of the vectors to be inserted in the index. [optional]  # noqa: E501
            metric (str): The distance metric to be used for similarity search. You can use 'euclidean', 'cosine', or 'dotproduct'. If the 'vector_type' is 'sparse', the metric must be 'dotproduct'. If the `vector_type` is `dense`, the metric defaults to 'cosine'. [optional]  # noqa: E501
            record_count (int): Total number of records in the backup. [optional]  # noqa: E501
            namespace_count (int): Number of namespaces in the backup. [optional]  # noqa: E501
            size_bytes (int): Size of the backup in bytes. [optional]  # noqa: E501
            tags (IndexTags): [optional]  # noqa: E501
            created_at (str): Timestamp when the backup was created. [optional]  # noqa: E501
        """

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", False)
        _enforce_validations = kwargs.pop("_enforce_validations", False)
        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _path_to_item = kwargs.pop("_path_to_item", ())
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        self = super(OpenApiModel, cls).__new__(cls)

        if args:
            raise PineconeApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                % (args, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.backup_id = backup_id
        self.source_index_name = source_index_name
        self.source_index_id = source_index_id
        self.status = status
        self.cloud = cloud
        self.region = region
        for var_name, var_value in kwargs.items():
            if (
                var_name not in self.attribute_map
                and self._configuration is not None
                and self._configuration.discard_unknown_keys
                and self.additional_properties_type is None
            ):
                # discard variable.
                continue
            setattr(self, var_name, var_value)
        return self

    required_properties = set(
        [
            "_enforce_allowed_values",
            "_enforce_validations",
            "_data_store",
            "_check_type",
            "_spec_property_naming",
            "_path_to_item",
            "_configuration",
            "_visited_composed_classes",
        ]
    )

    @convert_js_args_to_python_args
    def __init__(
        self, backup_id, source_index_name, source_index_id, status, cloud, region, *args, **kwargs
    ) -> None:  # noqa: E501
        """BackupModel - a model defined in OpenAPI

        Args:
            backup_id (str): Unique identifier for the backup.
            source_index_name (str): Name of the index from which the backup was taken.
            source_index_id (str): ID of the index.
            status (str): Current status of the backup (e.g., Initializing, Ready, Failed).
            cloud (str): Cloud provider where the backup is stored.
            region (str): Cloud region where the backup is stored.

        Keyword Args:
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
            name (str): Optional user-defined name for the backup. [optional]  # noqa: E501
            description (str): Optional description providing context for the backup. [optional]  # noqa: E501
            dimension (int): The dimensions of the vectors to be inserted in the index. [optional]  # noqa: E501
            metric (str): The distance metric to be used for similarity search. You can use 'euclidean', 'cosine', or 'dotproduct'. If the 'vector_type' is 'sparse', the metric must be 'dotproduct'. If the `vector_type` is `dense`, the metric defaults to 'cosine'. [optional]  # noqa: E501
            record_count (int): Total number of records in the backup. [optional]  # noqa: E501
            namespace_count (int): Number of namespaces in the backup. [optional]  # noqa: E501
            size_bytes (int): Size of the backup in bytes. [optional]  # noqa: E501
            tags (IndexTags): [optional]  # noqa: E501
            created_at (str): Timestamp when the backup was created. [optional]  # noqa: E501
        """

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", True)
        _enforce_validations = kwargs.pop("_enforce_validations", True)
        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _path_to_item = kwargs.pop("_path_to_item", ())
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        if args:
            raise PineconeApiTypeError(
                "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                % (args, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        self._data_store = {}
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)

        self.backup_id = backup_id
        self.source_index_name = source_index_name
        self.source_index_id = source_index_id
        self.status = status
        self.cloud = cloud
        self.region = region
        for var_name, var_value in kwargs.items():
            if (
                var_name not in self.attribute_map
                and self._configuration is not None
                and self._configuration.discard_unknown_keys
                and self.additional_properties_type is None
            ):
                # discard variable.
                continue
            setattr(self, var_name, var_value)
            if var_name in self.read_only_vars:
                raise PineconeApiAttributeError(
                    f"`{var_name}` is a read-only attribute. Use `from_openapi_data` to instantiate "
                    f"class with read only attributes."
                )
