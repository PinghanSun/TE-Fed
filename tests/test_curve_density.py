"""Long curves retain observations while keeping labels and markers readable."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree

from tefed.reporting import _svg_lines


class CurveDensityTests(unittest.TestCase):
    def test_300_observations_all_participate_in_line_with_sparse_markers(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'curve.svg'
            points = [(i, .5 + (i % 17) / 100) for i in range(1, 301)]
            _svg_lines(path, 'Training curve', [{'label': 'TE-Fed', 'points': points}], y_label='AP')
            root = ElementTree.parse(path).getroot()
            ns = {'s': 'http://www.w3.org/2000/svg'}
            line = root.find('s:polyline', ns)
            self.assertEqual(len(line.attrib['points'].split()), 300)
            self.assertLessEqual(len(root.findall('s:circle', ns)), 7)
            ticks = [node for node in root.findall('s:text', ns) if node.attrib.get('y') == '409']
            self.assertLessEqual(len(ticks), 7)
            self.assertEqual(ticks[0].text, '1')
            self.assertEqual(ticks[-1].text, '300')
