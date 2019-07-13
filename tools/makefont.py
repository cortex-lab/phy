#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create a multi-channel signed distance field map.
Use https://github.com/Chlumsky/msdfgen/

You need in this directory:

* msdfgen executable
* SourceCodePro-Regular.ttf

Just run this script to (re)create the font map directly in phy/plot/static/ in npy.gz format.

"""


import gzip
import os
from pathlib import Path

import imageio
import numpy as np

from phy.plot.visuals import FONT_MAP_SIZE, FONT_MAP_PATH, SDF_SIZE, FONT_MAP_CHARS, GLYPH_SIZE


class FontMapGenerator(object):
    """Generate a SDF font map for a monospace font, with a given uniform glyph size.

    """
    def __init__(self):
        self.rows, self.cols = FONT_MAP_SIZE
        self.font_map_output = FONT_MAP_PATH
        self.glyph_output = Path(__file__).parent / '_tmp.png'
        self.font = Path(__file__).parent / 'SourceCodePro-Regular.ttf'
        self.msdfgen_path = Path(__file__).parent / 'msdfgen'
        self.size = SDF_SIZE
        self.width, self.height = GLYPH_SIZE
        self.chars = FONT_MAP_CHARS

    def _iter_table(self):
        for i in range(self.rows):
            for j in range(self.cols):
                yield i, j

    def _get_char_number(self, i, j):
        return 32 + i * 16 + j

    def _get_glyph_range(self, i, j):
        x = j * self.width
        y = i * self.height
        return x, x + self.width, y, y + self.height

    def _get_cmd(self, char_number):
        """Command that generates a glyph signed distance field PNG to be used in the font map."""
        return (
            f'{self.msdfgen_path} msdf -font {self.font} {char_number} -o {self.glyph_output} '
            f'-size {self.width} {self.height} '
            '-pxrange 4 -scale 3.9 -translate 0.5 4')

    def _get_glyph_array(self, char_number):
        """Return the NumPy array with a glyph, by calling the msdfgen tool."""
        cmd = self._get_cmd(char_number)
        os.system(cmd)
        assert self.glyph_output.exists()
        return imageio.imread(self.glyph_output)

    def _make_font_map(self):
        """Create the font map by putting together all used glyphs."""
        Z = np.zeros((self.height * self.rows, self.width * self.cols, 3), dtype=np.uint8)
        for i, j in self._iter_table():
            char_number = self._get_char_number(i, j)
            x0, x1, y0, y1 = self._get_glyph_range(i, j)
            glyph = self._get_glyph_array(char_number)
            Z[y0:y1, x0:x1, :] = np.atleast_3d(glyph)[:, :, :3]
        return Z

    def make(self):
        """Create the font map and save it in phy/plot/static."""
        Z = self._make_font_map()
        with gzip.open(str(self.font_map_output), 'wb') as f:
            np.save(f, Z)
        os.remove(self.glyph_output)


if __name__ == '__main__':
    fmg = FontMapGenerator()
    fmg.make()
