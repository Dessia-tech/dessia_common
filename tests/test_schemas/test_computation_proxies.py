from dessia_common.schemas.core import OptionalProperty
from typing import List, Optional

import unittest
from parameterized import parameterized


class TestFaulty(unittest.TestCase):
    @parameterized.expand([
        (OptionalProperty(annotation=Optional[List[int]], attribute="optional_list", definition_default=None),)
    ])
    def test_schema_check(self, schema):
        self.assertEqual(schema.args, (int,))
        computed_schema = schema.to_dict()
        self.assertEqual(computed_schema["python_typing"], "List[int]")


if __name__ == '__main__':
    unittest.main(verbosity=2)
