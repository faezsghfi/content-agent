import uuid
from abc import ABC
from typing import Generic, Type, TypeVar

from loguru import logger
from pydantic import UUID4, BaseModel, Field
from pymongo import errors

from content_agent.exceptions import ImproperlyConfigured
from content_agent.crawling_data.raw_database.mongo import connection
from content_agent.settings import settings


# Create a connection to the MongoDB database.
# The database name is read from the application settings.
# For example, the database can be called "twin".
_database = connection.get_database(settings.DATABASE_NAME)

# Define a generic type variable T.
# T represents the type of document that inherits from NoSQLBaseDocument.
# "bound" means that T can only be NoSQLBaseDocument or one of its subclasses.
#
# For example:
# T = ArticleDocument
# T = RepositoryDocument
T = TypeVar("T", bound="NoSQLBaseDocument")


class NoSQLBaseDocument(BaseModel, Generic[T], ABC):
    """
    Base class for all MongoDB documents in the application.

    This class combines:
    - Pydantic BaseModel -> validates and structures the data.
    - Generic[T] -> allows the class to work with different document types.
    - ABC -> makes this an abstract base class.

    All document classes such as ArticleDocument,
    PostDocument, and RepositoryDocument inherit from this class.
    """


    # Unique identifier for every document.
    #
    # UUID4 is used instead of a normal integer ID.
    # default_factory generates a new UUID automatically
    # whenever a new document object is created.
    id: UUID4 = Field(default_factory=uuid.uuid4)

    def __eq__(self, value: object) -> bool:
        """
        Compare two document objects.

        Two documents are considered equal when:
        1. They are instances of the same class.
        2. They have the same ID.
        """    
        # If the other object is not the same type as this object,
        # they cannot be considered equal.
        if not isinstance(value, self.__class__):
            return False

        # Compare the unique IDs of the two objects.
        return self.id == value.id

    def __hash__(self) -> int:
        """
        Return a hash value based on the document ID.

        This allows document objects to be used in:
        - sets
        - dictionaries as keys
        """
        return hash(self.id)

    @classmethod
    def from_mongo(cls: Type[T], data: dict) -> T:
        """
        Convert a MongoDB document into a Python object.

        MongoDB uses "_id" for the document ID,
        while our Python models use "id".

        This method changes "_id" to "id"
        and creates an instance of the document class.
        """


        # Make sure MongoDB returned some data.
        if not data:
            raise ValueError("Data is empty.")


        # Remove "_id" from the MongoDB dictionary
        # and store its value in the variable "id".
        id = data.pop("_id")


        # Add the ID back using the name "id"
        # and create an instance of the current class.
        #
        # "cls" means the actual class calling this method.
        # For example, if ArticleDocument calls this method,
        # cls will be ArticleDocument.
        return cls(**dict(data, id=id))

    def to_mongo(self: T, **kwargs) -> dict:
        """
        Convert a Python document object into a MongoDB document.

        MongoDB expects the identifier to be stored as "_id",
        while our Python model uses "id".
        """

        # Decide whether fields that were not explicitly set
        # should be excluded from the output.
        exclude_unset = kwargs.pop("exclude_unset", False)
        by_alias = kwargs.pop("by_alias", True) # Use field aliases when converting the model to a dictionary.


        # Convert the Pydantic model into a Python dictionary.
        parsed = self.model_dump(exclude_unset=exclude_unset, by_alias=by_alias, **kwargs)


        # If the dictionary contains "id" but not "_id",
        # rename "id" to MongoDB's "_id".
        if "_id" not in parsed and "id" in parsed:
            parsed["_id"] = str(parsed.pop("id"))


        # Convert any UUID values into strings.
        # This makes them easier to store and serialize.
        for key, value in parsed.items():
            if isinstance(value, uuid.UUID):
                parsed[key] = str(value)

        # Return the dictionary that can be stored in MongoDB.
        return parsed

    def model_dump(self: T, **kwargs) -> dict:

        """
        Convert the Pydantic model into a dictionary.

        This method overrides Pydantic's default model_dump()
        so that UUID values are automatically converted to strings.
        """


        dict_ = super().model_dump(**kwargs) # Call Pydantic's original model_dump() method.


         # Check every field in the resulting dictionary.
        for key, value in dict_.items():
            # Convert UUID objects into strings.
            if isinstance(value, uuid.UUID):
                dict_[key] = str(value)

        # Return the converted dictionary.
        return dict_

    def save(self: T, **kwargs) -> T | None:

        """
        Save the current document object into MongoDB.

        Returns:
        - self -> if the document is successfully inserted.
        - None -> if the insertion fails.
        """


        # Get the MongoDB collection associated with this document class.
        #
        # For example:
        # ArticleDocument -> "articles"
        # RepositoryDocument -> "repositories"
        collection = _database[self.get_collection_name()]
        try:
            # Convert the Python object into a MongoDB-compatible dictionary
            # and insert it into the collection.
            collection.insert_one(self.to_mongo(**kwargs))

            return self
        except errors.WriteError:
            logger.exception("Failed to insert document.")

            return None

    @classmethod
    def get_or_create(cls: Type[T], **filter_options) -> T:

        """
        Find a document using the given filters.

        If the document exists:
            return the existing document.

        If it does not exist:
            create a new document and save it.
        """

        # Get the collection associated with the current document class.
        collection = _database[cls.get_collection_name()]
        try:
            # Search MongoDB for a document matching the filters.
            #
            # Example:
            # filter_options = {"username": "Alice"}
            #
            # MongoDB will search for:
            # {"username": "Alice"}
            instance = collection.find_one(filter_options)

            # If a matching document exists,
            # convert it from MongoDB format into a Python object.
            if instance:
                return cls.from_mongo(instance)


            # If no document was found,
            # create a new instance using the filter values.
            new_instance = cls(**filter_options)

             # Save the new document into MongoDB.
            new_instance = new_instance.save()

            return new_instance
        except errors.OperationFailure:
            logger.exception(f"Failed to retrieve document with filter options: {filter_options}")

            # Re-raise the exception so the caller knows
            # that the database operation failed.
            raise

    @classmethod
    def bulk_insert(cls: Type[T], documents: list[T], **kwargs) -> bool:
        """
        Insert multiple documents into MongoDB at once.

        This is more efficient than calling save()
        separately for every document.
        """

        # Get the collection associated with the document class.
        collection = _database[cls.get_collection_name()]
        try:
            # Convert every Python document into a MongoDB dictionary
            # and insert all of them at once.
            collection.insert_many(doc.to_mongo(**kwargs) for doc in documents)

            # Return True when the insertion succeeds.
            return True
        except (errors.WriteError, errors.BulkWriteError):
            logger.error(f"Failed to insert documents of type {cls.__name__}")

            return False

    @classmethod
    def find(cls: Type[T], **filter_options) -> T | None:

        """
        Find one document in MongoDB using the given filters.

        Returns:
        - A Python document object if a match is found.
        - None if no document is found.
        """

        # Get the collection associated with this document class.
        collection = _database[cls.get_collection_name()]
        try:
            # Search MongoDB for the first document
            # matching the given filters.
            instance = collection.find_one(filter_options)

            # If a document is found,
            # convert it into a Python object.
            if instance:
                return cls.from_mongo(instance)

            return None
        except errors.OperationFailure:
            logger.error("Failed to retrieve document")

            return None

    @classmethod
    def bulk_find(cls: Type[T], **filter_options) -> list[T]:

        """
        Find multiple documents in MongoDB using the given filters.

        Returns a list of Python document objects.
        """

        # Get the collection associated with this document class.
        collection = _database[cls.get_collection_name()]
        try:
            # Find all documents matching the filter.
            instances = collection.find(filter_options)

            # Convert every MongoDB document into a Python object.
            #
            # cls.from_mongo(instance)
            # converts each MongoDB dictionary into
            # an instance of the current document class.
            return [document for instance in instances if (document := cls.from_mongo(instance)) is not None]
        except errors.OperationFailure:
            logger.error("Failed to retrieve documents")

            return []

    @classmethod
    def get_collection_name(cls: Type[T]) -> str:

        """
        Return the MongoDB collection name for the current document class.

        Each child document class must define a nested Settings class
        containing a "name" attribute.

        Example:

        class ArticleDocument(NoSQLBaseDocument):
            class Settings:
                name = "articles"
        """

        # Check whether the subclass has a Settings class
        # and whether Settings contains a "name" attribute.
        if not hasattr(cls, "Settings") or not hasattr(cls.Settings, "name"):
            raise ImproperlyConfigured(
                "Document should define an Settings configuration class with the name of the collection."
            )

        return cls.Settings.name
