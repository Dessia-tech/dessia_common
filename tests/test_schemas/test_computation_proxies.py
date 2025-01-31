from dessia_common.schemas.core import OptionalProperty,SchemaAttribute
from typing import List, Optional

import unittest
from parameterized import parameterized


ATTRIBUTE = SchemaAttribute(name="optional_list", default_value=None)


class TestFaulty(unittest.TestCase):
    @parameterized.expand([
        (OptionalProperty(annotation=Optional[List[int]], attribute=ATTRIBUTE),)
    ])
    def test_schema_check(self, schema):
        self.assertEqual(schema.args, (int,))
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["pythonTyping"], "List[int]")


if __name__ == '__main__':
    unittest.main(verbosity=2)
