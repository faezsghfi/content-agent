from langchain_core.output_parsers import PydanticOutputParser

# Parses and validates language-model outputs into structured dataset records.

class ListPydanticOutputParser(PydanticOutputParser):
    def _parse_obj(self, obj: dict | list):
        if isinstance(obj, list):
            return [super(ListPydanticOutputParser, self)._parse_obj(obj_) for obj_ in obj]
        else:
            return super(ListPydanticOutputParser, self)._parse_obj(obj)
