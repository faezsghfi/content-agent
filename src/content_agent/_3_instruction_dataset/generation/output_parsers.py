from langchain_core.output_parsers import PydanticOutputParser

# Custom output parser that can handle both a single object
# and a list of objects returned by the language model.
class ListPydanticOutputParser(PydanticOutputParser):
    def _parse_obj(self, obj: dict | list):
        if isinstance(obj, list):
            return [super(ListPydanticOutputParser, self)._parse_obj(obj_) for obj_ in obj]
        else:
            return super(ListPydanticOutputParser, self)._parse_obj(obj)
