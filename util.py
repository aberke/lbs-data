"""
Utilities.
"""

import unittest


# ******************
# Shapefile utilities
# ******************
# Note there are files of differing geographic granuality: census tracts, block groups, blocks.
# They are named accordingly:  shapefiles/[state]/[level]_[regions]
# level: (county|tract)
# region: (e.g. middlesex_norfolk_suffolk)
# e.g. /shapefiles/ma/blockgroup_middlesex_norfolk_suffolk.shp

DEFAULT_SHAPEFILE_DIR = "./shapefiles/"


def get_shapefile_filename(level, regions=None):
    """ Constructs filename for shapefile for given level and regions
    Args:
        (str) level of census area
        [(str)] optional list of regions [reg1, reg2, ...]
    Returns filename in format 'level[_reg1_reg2]' for each region in regions list, 
        where the output regions are sorted.
        If no regions included, returns 'level'
    """
    filename = level
    if (regions and len(regions)):
        regions = [r.lower() for r in regions]
        regions = sorted(regions)
        filename = filename + "_"  + "_".join(regions)
    return filename + '.shp'

def get_shapefile_filepath(state, level, regions=None, shapefile_dir=DEFAULT_SHAPEFILE_DIR):
    return shapefile_dir + state + '/' + get_shapefile_filename(level, regions)



# ***********
# Tests
# ***********

class ShapefileTest(unittest.TestCase):

    def test_get_shapefile_filename(self):
        self.assertEqual(
            get_shapefile_filename("county", ["norfolk", "Middlesex", "suffolk"]),
            "county_middlesex_norfolk_suffolk.shp")
        self.assertEqual(get_shapefile_filename("blockgroup"),
            "blockgroup.shp")

    def test_get_shapefile_filepath(self):
        self.assertEqual(
            get_shapefile_filepath("ma", "tract"),
            DEFAULT_SHAPEFILE_DIR + "ma/tract.shp")



if __name__ == '__main__':
    unittest.main()
