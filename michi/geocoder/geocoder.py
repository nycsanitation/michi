import itertools
import pickle
import re
from warnings import warn

import geopandas as gp
from geosupport import Geosupport, GeosupportError
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point
from shapely.ops import transform
from rtree import index as rtree_index
import pyproj

from ..config import MICHI_HOME
from ..utils.lion import load_lion_gdf
from ..utils.utils import drop_consecutive_duplicates, method_file_cache
from .errors import (
    IntersectionNotFoundError,
    NotAStreetError,
    StreetNameNotFoundError,
    StreetStretchNotFoundError,
    WrongWayError
)
from .globals import BOROUGHS, BOROUGH_CODES
from .network import (
    build_monodirectional_network, drop_internal_nodes,
    build_directional_network, default_cost_function, build_segment_network
)
from .street_stretch import StreetStretch


def _add_connector_segments(lion):
    """
    Add attribute `connector_segment` which means that the segment connect
    roadbed segments with a generic segment on a divided street.

    Changes `lion` in place.
    """
    generic_nodes = set()
    roadbed_nodes = set()

    # Segment IDs to exclude
    exceptions = ['0182982', '0283378']

    for endpoint in ['from', 'to']:
        generic_nodes.update(lion[
            (lion['node_level_%s' % endpoint] == '*') &
            (lion['segment_type'].isin(['G', 'B', 'T']))
        ][['node_id_%s' % endpoint]]['node_id_%s' % endpoint])

        roadbed_nodes.update(lion[
            (lion['segment_type'].isin(['R', 'T']))
        ][['node_id_%s' % endpoint]]['node_id_%s' % endpoint])

    lion['connector_segment'] = (
        ~lion['segment_id'].isin(exceptions) &
        ((
            lion['node_id_to'].isin(generic_nodes) &
            lion['node_id_from'].isin(roadbed_nodes)
        ) | (
            lion['node_id_to'].isin(roadbed_nodes) &
            lion['node_id_from'].isin(generic_nodes)
        ))
    )

def _handle_special_addresses(lion):
    """
    When there are special address codes/names, ensure that there is a duplicate
    row with the special name and code as the primary.

    Note: Only for special address type 'P' - addressable place names
    """

    special = lion[
        (lion['special_address_type'].isin(['P', 'B', 'G'])) &
        (lion['street'] != lion['special_address_street_name'])
    ].drop(columns=['street', 'street_code'])

    special['street'] = special['special_address_street_name']
    special['street_code'] = special['special_address_street_code']
    special['special_address_street_code'] = ""
    special['special_address_street_name'] = ""

    lion = pd.concat([lion, special], sort=True).reset_index(drop=True)

    return lion

def _clean_lion_df(lion, crs):
    """
    Load LION from SDE and then modify it for Geocoder.

    Parameters
    ----------
    lion_version: str
        The version of LION in YY{ABCD} format such as 19A, 18D, etc.
    crs: str
        The coordinate reference system (CRS) to convert the geometry to as a
        EPSG code such as 'epsg:2263' or 'epsg:4326'.
    """
    # Strip extra whitespace from strings
    for col, dtype in lion.dtypes.reset_index().values:
        if (dtype == 'O') and (col is not 'geometry'):
            lion[col] = lion[col].str.strip()

    _add_connector_segments(lion)

    lion = lion[
        ((lion['node_level_from'] != '*') | (lion['node_level_from'] != '*')) &
        ~pd.isnull(lion['physical_id'])
    ].copy()
    lion['physical_id'] = lion['physical_id'].astype(int).astype(str)
    lion['node_to'] = lion['node_id_to'] + lion['node_level_to']
    lion['node_from'] = lion['node_id_from'] + lion['node_level_from']
    lion['borough_code'] = lion['street_code'].str[0]
    lion = lion.fillna({'number_travel_lanes': lion['number_total_lanes']})
    lion['number_travel_lanes'] = (
        lion['number_travel_lanes'].str.replace(r'^\s*$', '1').astype(int)
    )

    lion = _handle_special_addresses(lion)

    lion = lion.drop_duplicates(
        subset=[col for col in lion.columns if col is not "geometry"]
    )

    # Merge streets that _should_ have a single street code
    # Keep the original street code as 'original_street_code'
    lion = _create_generic_street_codes(lion)

    lion['len'] = lion['geometry'].length
    if crs:
        lion = lion.to_crs({'init': crs})

    return lion

def _get_nodes_df(lion_df):
    columns = ['segment_id', 'street', 'street_code', 'borough_code']
    streets = lion_df[
        lion_df['traffic_direction'].isin(['W', 'A', 'T'])
    ].copy()

    streets['geometry_from'] = streets['geometry'].apply(
        lambda g: Point(g.coords[0])
    )
    streets['geometry_to'] = streets['geometry'].apply(
        lambda g: Point(g.coords[-1])
    )

    return pd.concat([
        streets[columns + ['node_to', 'geometry_to']].rename(
            columns={'node_to': 'node', 'geometry_to': 'geometry'}
        ),
        streets[columns + ['node_from', 'geometry_from']].rename(
            columns={'node_from': 'node', 'geometry_from': 'geometry'}
        )
    ]).drop_duplicates(columns + ['node'])

def _get_cscl_segments_df(lion):
    """
    Given the LION dataframe, return a dataframe where each row is a CSCL
    segment instead of a LION segment.

    Parameters
    ----------
    lion : pandas.DataFrame
        The geocoder's lion_df

    Returns
    -------
    pandas.DataFrame
    """
    def get_traffic_direction(group):
        """
        Get a single traffic direction (T, W or A) for a physical_id and
        handle when there is more than one traffic direction for a single
        physical ID. If there is more than one, return the most common. If
        There are two tied, return 'T' for two-way.
        """
        # If there's more than one traffic direction and there's a tie for the
        # most segments with a traffic direction, then return 'T' for two-way.
        # i.e there are 7 lion segmets or the physical_id, three have 'A',
        # three have 'W' and one has 'T', return 'T'.
        # `group['len']` is the number of segments with a given traffic
        # direction. See lines below this function for details.
        if (len(group) > 1) and (group['len'].iloc[0] == group['len'].iloc[1]):
            return 'T'
        else:
            return group['traffic_direction'].iloc[0]

    # Create a dataframe which has a count of the number of each traffic
    # direction per physical_id. Then apply `get_traffic_direction` to each
    # physical_id group to handle any cases where there are multiple directions
    # on a single physical_id. In the end, get a Series of traffic directions
    # with physical_id as the index.
    traffic_directions = lion.drop_duplicates([
        'segment_id', 'traffic_direction'
    ]).groupby([
        'physical_id', 'traffic_direction'
    ]).count()['len'].sort_values(
        ascending=False
    ).reset_index().groupby(
        'physical_id'
    ).apply(get_traffic_direction)

    def get_street_street_code_pairs(group):
        """
        Get a list of tuples of (street name, street code) for each physical_id
        sorted in descending order by the number of segments on that physical_id
        with that pair.
        """
        return group.groupby([
            'street', 'street_code'
        ]).count().sort_values('len', ascending=False).index.tolist()

    def get_segment_type(group):
        """
        Get the segment type of the physical_id.
        """
        types = group['segment_type'].tolist()

        # If any of the segments are 'T' (Terminator) or 'R' (Roadbed),
        # return that value.
        if len(types) > 1:
            for t in ['T', 'R']:
                if t in types:
                    return t

        # Otherwise, just return the first value.
        return types[0]

    # Create the dataframe of cscl_segments with each row summarizing the LION
    # segments that compose the physical_id.
    df = pd.concat([
        # A set of the street codes on the segment
        lion.groupby('physical_id').apply(
            lambda g: set(g['street_code'])
        ).rename('street_code'),

        # A set of the street names
        lion.groupby('physical_id').apply(
            lambda g: set(g['street'])
        ).rename('street'),

        # A list of tuples of the street/street code pairs
        lion.groupby('physical_id').apply(
            get_street_street_code_pairs
        ).rename('street_street_code_pair'),

        # A set of the segment_ids
        lion.groupby('physical_id').apply(
            lambda g: set(g['segment_id'])
        ).rename('segment_ids'),

        # The length in feet of the physical_id
        lion.drop_duplicates([
            'segment_id', 'physical_id'
        ]).groupby('physical_id')['len'].sum(),

        # The maximum number of travel lanes
        lion.groupby('physical_id')['number_travel_lanes'].max(),

        # The traffic direction
        traffic_directions.rename('traffic_direction'),

        # Whether any of the segments are connector segments
        lion.groupby('physical_id')['connector_segment'].any(),

        # The segment type
        lion.drop_duplicates(
            subset=['physical_id', 'segment_type']
        ).groupby('physical_id').apply(get_segment_type).rename('segment_type')
    ], axis=1)
    df['physical_id'] = df.index

    return df

def _get_spatial_index(df, base_crs, index_crs, version, name, object_id):
    """
    creates an rtree spatial index on nodes to aid near searches

    Params:
        nodes_df: nodes data frame
        base_crs: crs of geocoder
        index_crs: crs of index if different from base, else None
        cache_key: cache_key created for _load_lion_data

    """
    path = str(MICHI_HOME / version / 'geocoder' / name)
    index = rtree_index.Index(path)

    if len(index.leaves()) <= 1:
        print('Building {}. Base_CRS: {}, Index_CRS: {}'.format(
            name, base_crs, index_crs
        ))

        if index_crs:
            transformer = pyproj.Transformer.from_proj(
                pyproj.Proj(init=base_crs), pyproj.Proj(init=index_crs)
            )

        def get_bounding_box(geom, index_crs):
            if index_crs:
                geom = transform(transformer.transform, geom)
            X, Y = geom.xy
            X, Y = sorted(X), sorted(Y)
            xmin, xmax, ymin, ymax = X[0], X[-1], Y[0], Y[-1]
            return xmin, ymin, xmax, ymax

        for i, row in df.iterrows():
            index.insert(
                i, (get_bounding_box(row['geometry'], index_crs)),
                obj=row[object_id]
            )

        index.close()
        index = rtree_index.Index(path)

    return index

def _get_multi_borough_street_codes(nodes_df):
    def get_multi_borough_streets(group):
        borough_counts = group.groupby('node')['borough_code'].nunique()
        nodes = borough_counts[borough_counts > 1].index

        return group[
            group['node'].isin(nodes)
        ]['street_code'].unique() if len(nodes) else None

    multi_borough_streets = nodes_df.sort_values(
        'borough_code'
    ).drop_duplicates(
        subset=['segment_id', 'node']
    ).groupby('street').apply(get_multi_borough_streets).dropna()

    records = []
    for street_codes in multi_borough_streets:
        for street_code in street_codes:
            records.append([street_code, '+'.join(sorted(street_codes))])

    street_code_df = pd.DataFrame.from_records(
        records, columns=['street_code', 'new_street_code']
    )

    return street_code_df

def _merge_generic_street_codes(lion_df, new_df):
    lion_df = lion_df.merge(new_df, on='street_code', how='left')
    lion_df['new_street_code'] = lion_df['new_street_code'].fillna(
        lion_df['street_code']
    )
    lion_df = lion_df.drop(columns=['street_code']).rename(
        columns={'new_street_code': 'street_code'}
    )
    return lion_df

def _east_west_generic_street_code(lion_df):
    connected_streets = []
    for borough_code in ['1', '2']:
        streets = lion_df[
            (lion_df['borough_code'] == borough_code) &
            lion_df['street'].str.contains('^(?:EAST|WEST){0,1} \d+ STREET') &
            ~lion_df['street'].str.contains('(?:PEDESTRIAN|FOOTBRIDGE|BIKE)')
        ].copy()
        streets['number'] = streets['street'].str.extract(
            '^(EAST|WEST){0,1} (\d+) STREET'
        )[1].str.zfill(3)
        streets_dict = _get_streets_dict(streets)

        def connected(group):
            street_codes = group['street_code'].unique()
            if len(street_codes) < 2:
                return (None, None)

            a, b = street_codes
            if streets_dict[a]['nodes'].intersection(
                streets_dict[b]['nodes']
            ):
                return (a, b), group['number'].iloc[0]

            return None, None

        for street_codes, number in streets.groupby('number').apply(connected):
            if number:
                for street_code in street_codes:
                    connected_streets.append((
                        street_code, '%s_%s_street' % (borough_code, number)
                    ))

    return _merge_generic_street_codes(
        lion_df,
        pd.DataFrame.from_records(
            connected_streets, columns=['street_code', 'new_street_code']
        )
    )

def _create_generic_street_codes(lion_df):
    """
    Handle places where the street continues, sometimes under a slightly
    different name, and always under a different street_code.

    For example, streets going accross the brooklyn queens border change
    street codes, but should be connected.

    Or enable "Park Avenue" to also refer to "Park Avenue South"
    """
    lion_df['original_street_code'] = lion_df['street_code']

    # MN AND BX East/West streets that connect
    lion_df = _east_west_generic_street_code(lion_df)

    # Manhattan Park Ave
    lion_df.loc[
        (lion_df['borough_code'] == '1') &
        lion_df['street'].str.contains('^PARK AVENUE'),
        'street_code'
    ] = 'mn_park_ave'

    # Manhattan 7th Ave
    lion_df.loc[
        (lion_df['borough_code'] == '1') &
        lion_df['street'].str.contains('^(?:7 AVENUE|ADAM CLAYTON POWELL)'),
        'street_code'
    ] = 'mn_7_ave'

    # Flatbush Ave
    lion_df.loc[
        (lion_df['borough_code'] == '3') &
        lion_df['street'].isin(['FLATBUSH AVENUE', 'FLATBUSH AVENUE EXTENSION']),
        'street_code'
    ] = 'bk_flatbush_ave'

    # HIGHWAYS! TOO HARD! GUH...

    # Henry Hudson
    henry_hudson_street_codes = lion_df[
        lion_df['street'].str.contains('HENRY HUDSON') &
        (lion_df['non_pedestrian'] == 'V')
    ]['street_code'].unique()

    lion_df.loc[
        lion_df['street_code'].isin(henry_hudson_street_codes),
        'street_code'
    ] = 'west_side_highway'

    # Major Deegan
    lion_df.loc[
        lion_df['street'].str.contains('^(?:MAJOR DEEGAN|MDE)'),
        'street_code'
    ] = 'major_deegan'

    # Merge streets that go across borough boundaries
    nodes_df = _get_nodes_df(lion_df)
    lion_df = _merge_generic_street_codes(
        lion_df, _get_multi_borough_street_codes(nodes_df)
    )

    return lion_df

def _get_dead_end_nodes(nodes_df):
    nodes_df = nodes_df[
        ~nodes_df['node'].str.contains('\*') &
        (nodes_df['street'] != 'DRIVEWAY')
    ].drop_duplicates(['segment_id', 'node'])
    dead_ends = nodes_df.groupby('node').filter(lambda g: len(g) == 1)

    return set(dead_ends['node'])

def _get_streets_dict(lion_df):
    streets = {}
    for street_code, group in lion_df.groupby('street_code'):
        streets[street_code] = {
            'df': group,
            'nodes': set(pd.concat([group['node_to'], group['node_from']])),
        }

    return streets


def _get_nodes_dict(nodes_df):
    """
    The same as `geocoder._get_nodes_dict` but also return whether the node is
    internal.
    """
    return dict(nodes_df.groupby('node').apply(
        lambda g: {
            'segments': set(g['segment_id']),
            'geometry': g['geometry'].iloc[0],
            'dead_end': g['dead_end'].iloc[0],
            'internal': g['internal'].iloc[0]
        }
    ))

def _get_segments_dict(lion_df, columns=None):
    if columns is None:
        columns = [
            'segment_id', 'physical_id', 'node_from', 'node_to', 'geometry',
            'right_blockface_id', 'left_blockface_id',
            'street', 'street_code',
            'street_width_min', 'street_width_max', 'len',
            'number_travel_lanes', 'traffic_direction',
            'connector_segment', 'segment_type'
        ]

        columns = [col for col in columns if col in lion_df]

    def create_dict(group):
        d = dict([(col, group[col].iloc[0]) for col in columns])
        d['street_code'] = set(group['street_code'])
        d['street'] = set(group['street'])
        d['street_street_code_pair'] = set(
            [tuple(i) for i in group[['street', 'street_code']].values]
        )
        return d

    return dict(lion_df.groupby('segment_id').apply(create_dict))

# Call super left to right
# Geocoder -> SearchMixin -> NetworkMixin
#class Geocoder(SearchMixin, NetworkMixin):
class Geocoder():
    """
    A class that provides functionality for getting street stretches, etc.

    Exposes the following functions:

        normalize_street_name
            Returns a street in the format the LION and Geosupport use.
        get_street_code
            Given a street name, return the street code.
        street_code_exists
            Return whether the given street code exists in LION's streets
        autocomplete
            Given a text search, return possible street name matches.
        address
            Find the street segment and side of street of an address.
    """
    def __init__(self, lion_version, crs='epsg:2263',
                 lion_loader=load_lion_gdf, force_rebuild=False,
                 network_type='cscl',
                 include_spatial_index=False, spatial_index_crs=None, **kwargs):

        self.crs = crs
        self.lion_version = lion_version.lower()
        self.lion_loader = lion_loader
        self.force_rebuild = force_rebuild
        self.cache_path = MICHI_HOME / self.lion_version / 'geocoder'
        if not self.cache_path.exists():
            self.cache_path.mkdir(parents=True)

        # ensure CRS parameters are parsable before instantiation
        pyproj.crs.CRS(crs)
        if spatial_index_crs:
            pyproj.crs.CRS(spatial_index_crs)

        # Instantiate Geosupport
        try:
            if geosupport_version is None:
                geosupport_version = self.lion_version
            self.geosupport = Geosupport(geosupport_version=version)
        except:
            # Use default geosupport
            self.geosupport = Geosupport()
            warn(
                "Using default Geosupport. "
                "May not match LION %s" % self.lion_version
            )

        (
            self.lion_df, self.nodes_df,
            self.lion_segments, self.streets,
            self.street_code_map
        ) = self._load_lion_data()

        self.street_names = self._build_search_data()

        self.network_type = network_type
        if network_type == 'cscl':
            self.segment_column = 'physical_id'
            self.segment_dict = 'cscl_segments'
        elif network_type == 'lion':
            self.segment_column = 'segment_id'
            self.segment_dict = 'lion_segments'

        # A regex to parse geometry strings, to be used by `self.parse_geometry`
        self.geometry_regex = re.compile(r'([a-z_]*):*(\d+)([A-Z]{0,1})')

        (
            self.cscl_segments_df, self.cscl_segments,
            self.node_network, self.segment_network,
            self.nodes_df, self.nodes
        ) = self._build_networks(self.network_type)

        self.segments = getattr(self, self.segment_dict)

        # SearchMixin
        # TODO Don't skip this, testing!
        #super().__init__(clear_cache=clear_cache, **kwargs)

        # get cache key to uniquely recognize spatial indexes
        #cache_key = get_cache_key(('_load_lion_data',) + (self,) + ({},))

        # load spatial indexes
        self.include_spatial_index = include_spatial_index
        self.spatial_index_crs = spatial_index_crs

        if self.include_spatial_index:
            self.lion_index = _get_spatial_index(
                self.lion_df, self.crs, self.spatial_index_crs, lion_version,
                'segment_index', 'segment_id'
            )
            self.node_index = _get_spatial_index(
                self.nodes_df, self.crs, self.spatial_index_crs, lion_version,
                'node_index', 'node'
            )

    @method_file_cache('lion.pkl')
    def _load_lion_data(self):
        print("Downloading and Building Geocoder Data...")
        #lion_df = _load_lion_df(
        #    self.lion_version, self.crs
        #)
        lion_df = self.lion_loader(self.lion_version)

        '''return (
            lion_df,# nodes_df,
            None, None, None, None
            #segments, streets,
            #street_code_map
        )'''

        lion_df = _clean_lion_df(lion_df, self.crs)

        # Create nodes_df
        nodes_df = _get_nodes_df(lion_df)

        # Get dead ends
        dead_end_nodes = _get_dead_end_nodes(nodes_df)
        nodes_df['dead_end'] = nodes_df['node'].isin(dead_end_nodes)

        # Convert to GeoDataFrame
        nodes_df = gp.GeoDataFrame(
            nodes_df, geometry='geometry', crs={'init': self.crs}
        )

        # Cache stuff for quick access
        streets = _get_streets_dict(lion_df)
        segments = _get_segments_dict(lion_df)

        # Add dead end nodes to `streets`
        dead_end_street_codes = self._get_dead_end_street_codes()
        streets['dead_end'] = {'nodes': dead_end_nodes, 'df': None}

        # Create a mapping of street codes to internal street codes
        street_code_map = dict(lion_df[[
            'original_street_code', 'street_code'
        ]].drop_duplicates().values)

        # Add dead ends to street_code_map
        street_code_map.update(dict([
            (i, 'dead_end') for i in dead_end_street_codes
        ]))

        # Add mapping from new values to themselves
        for key, value in list(street_code_map.items()):
            if value not in street_code_map:
                street_code_map[value] = value

        return (
            lion_df, nodes_df,
            segments, streets,
            street_code_map
        )

    @method_file_cache('search.pkl')
    def _build_search_data(self):
        print("Building Search Data...")

        # Create dataframe with street names for autocomplete
        street_names = self.lion_df[
            self.lion_df['feature_type'].isin(['0', '6', '8', 'A', 'W'])
        ].groupby([
            'street', 'street_code', 'borough_code'
        ]).count()['segment_id'].reset_index()

        # Add options for a dead end for each borough to the dataframe.
        dead_end_street_codes = self._get_dead_end_street_codes()
        street_names = street_names.append([{
            'street': 'DEAD END',
            'street_code': 'dead_end',
            'borough_code': street_code[0],
            'segment_id': 100
        } for street_code in dead_end_street_codes])

        # A function to get the geometry of each "street" used for the
        # centroid distance functionality of autocomplete.
        def get_geometry(street_code):
            if street_code == 'dead_end':
                return self.nodes_df[
                    self.nodes_df['node'].isin(self.streets['dead_end']['nodes'])
                ]['geometry'].unary_union

            return self.streets[street_code]['df']['geometry'].unary_union

        street_names['geometry'] = street_names['street_code'].apply(get_geometry)

        street_names = gp.GeoDataFrame(street_names.sort_values(
            'segment_id', ascending=False
        ).reset_index(drop=True))

        return street_names

    def _get_street_code(self, borough, street):
        return self.geosupport.get_street_code({
            'street_name': street, 'borough_code': borough
        })['B10SC - First Borough and Street Code']

    def _get_dead_end_street_codes(self):
        return [self._get_street_code(b, 'DEAD END') for b in BOROUGHS]

    def _get_intersection(self, street_code_1, street_code_2):
        street_code_1 = self.normalize_street_code(street_code_1)
        street_code_2 = self.normalize_street_code(street_code_2)

        return self.streets[street_code_1]['nodes'].intersection(
            self.streets[street_code_2]['nodes']
        )

    def get_street_code(self, borough, street):
        """
        Given a borough and street name, return the street code.

        Parameters
        ----------
        street : str
        borough : str
            The borough code, abbreviation or name.

        Returns
        -------
        str
            The street code

        Raises
        ------
        NotAStreetError
            If the "street" is a valid identifier in Geosupport, but isn't a
            drivable street in LION.
        StreetNameNotFoundError
            If the street name isn't recognized raise this error. The error
            has an attribute `options` with up to 10 alternate street names
            that are similar to the given one.
        """
        try:
            street_code = self.normalize_street_code(
                self._get_street_code(borough, street)
            )

            if self.street_code_exists(street_code):
                return street_code
            else:
                raise NotAStreetError(street, street_code)

        except GeosupportError as error:
            # If geosupport raised an error, include it's suggested street
            # names in the raised StreetNameNotFoundError.
            options = []
            for s in error.result['List of Street Names']:
                street_code = self.normalize_street_code(
                    self._get_street_code(borough, s)
                )
                # Only return valid streets
                if self.street_code_exists(street_code):
                    # Append the name and the code.
                    options.append((s, street_code))

            raise StreetNameNotFoundError(street, options)

    def normalize_street_code(self, street_code):
        if street_code in self.street_code_map:
            return self.street_code_map[street_code]
        street_code = str(street_code)[:6].zfill(6)
        return self.street_code_map.get(street_code, street_code)

    def street_code_exists(self, street_code):
        return self.normalize_street_code(street_code) in self.streets

    def normalize_street_name(self, street):
        """
        Return the street name normalized into the format used by LION and
        Geosupport.

        Parameters
        ----------
        street : str
            The raw street name.

        Returns
        -------
        str
        """
        return re.sub(
            r'\s+', ' ',
            self.geosupport.normalize_street_name(
                street=street
            )['First Street Name Normalized']
        )

    def autocomplete(self, text, borough=None, cross_street_code=None,
                     return_top_n=10, centroid=None):
        """
        Given a string and optional filter parameters, return the most likely
        streets.

        Parameters
        ----------
        text : str
            The street name or partial street name.
        borough : str, optional
            Optionally constrain the search to a single borough which can be
            provided as a borough code, abbreviation or full name.
        cross_street_code : str, optional
            Only return streets that intersect with the given street.
        return_top_n : int, optional
            The number of results to return. (Default 10)
        centroid : shapely.geometry.Point, optional
            A point in the same crs as geocoder.crs. If given, sort the results
            by distance from the centroid.
        """
        text = text.strip().upper()
        text_normalized = self.normalize_street_name(text)

        df = self.street_names.copy()

        if borough:
            df = df[df['borough_code'] == str(BOROUGH_CODES[str(borough)])]

        if cross_street_code:
            cross_street_code = self.normalize_street_code(cross_street_code)

            # Get all the nodes that are on the given street code.
            nodes = self.nodes_df[self.nodes_df['node'].isin(
                    self.streets[cross_street_code]['nodes']
            )]

            # Get a list of street codes that intersect with those nodes.
            street_codes = nodes['street_code'].unique().tolist()

            # If any of the nodes are a dead end, add that street.
            if nodes['dead_end'].any():
                street_codes.append('dead_end')

            boroughs = nodes['borough_code'].unique()

            df = df[
                df['street_code'].isin(street_codes) &
                df['borough_code'].isin(boroughs) # Handles the extra dead ends
            ]

        query = (
            # Starts with the text
            df['street'].str.startswith(text) |
            df['street'].str.startswith(text_normalized) |

            # Contains the full word
            df['street'].str.contains(r'\b%s\b' % text) |
            df['street'].str.contains(r'\b%s\b' % text_normalized)
        )

        # If the string is 5 or more characters, search anywhere in the string
        if len(text) >= 5:
            query = (
                query | df['street'].str.contains(text) |
                df['street'].str.contains(text_normalized)
            )

        df = df[query].copy()

        if centroid:
            # If a cross street is given, find the distance to the place
            # where the two streets intersect, and not just to anywhere along
            # the street.
            if cross_street_code:
                func = lambda s: centroid.distance(
                    self.nodes[
                        self._get_intersection(s, cross_street_code).pop()
                    ]['geometry']
                )
                df['distance'] = df['street_code'].apply(func)
            else:
                df['distance'] = df['geometry'].distance(centroid)

            # By default, the options are sorted by "segment_id", which is the
            # number of segments with that street name in the city.
            # i.e, show common streets like "Broadway" before "Broad Street"
            # When sorting by distance, we want to take into account both
            # the distance and how common the street is.
            # The log of the count will map the count into a much smaller but
            # still increasing number. "+ e - 1" ensures that the devisor starts
            # at 1 for streets that only have 1 segment.
            # After division, if two streets are the same distance from the
            # centroid, the one with a higher count will have a lower
            # "distance." But the count won't overpower the actual distance.
            # After, the options are sorted by distance in ascending order.
            df['distance'] = df['distance'] / np.log(df['segment_id'] + np.e - 1)
            df = df.sort_values('distance')

        df = df.head(return_top_n)

        # Convert the resulting dataframe into a list of dictionaries
        results = []
        for i,row in df.iterrows():
            results.append({
                'street': row['street'],
                'street_code': row['street_code'],
                'borough_code': row['borough_code'],
                'node': list(self._get_intersection(
                    cross_street_code, row['street_code']
                )) if cross_street_code else None
            })

        return results

    def address(self, house_number, street, borough, drivable=True):
        """
        Given an address as house number, street and borough, return the
        segment id, physical id, blockface id  and side of street of that
        address.

        Parameters
        ----------
        house_number : str or int
            The house number, including hyphens for Queens addresses.
        street : str
        borough : str
            The borough code, abbreviation or name.
        drivable : bool, optional
            Whether to only return drivable segments. (Default True)

        Returns
        -------
        dict
            dict of segment id, physical id, blockface id  and side of street
        """
        street_code = self._get_street_code(borough, street)
        street_normalized = self.normalize_street_name(street)

        # Create a list of possible geographic identifiers for this address.
        # Sometimes the physical location of a building is not reflected
        # in its address, so we'll use GeoSupport to identify other options.
        # The first one is simply the house number and street code.
        lgis = [(house_number, street_code)]

        # Then pass the address to GeoSupport and iterate through its
        # list of geographic identifiers and add all of them.
        for lgi in self.geosupport.address(
            borough=borough, street=str(house_number) + ' ' + street_normalized
        )['LIST OF GEOGRAPHIC IDENTIFIERS']:
            lgis.append((
                lgi['High House Number'],
                lgi['Borough Code'] + lgi['5-Digit Street Code']
            ))

        # Eliminate generic segments and, if drivable is True, only include
        # drivable segments.
        df = self.lion_df[
            (
                self.lion_df['traffic_direction'].isin(['W', 'A', 'T'])
                if drivable else True
            ) & (~self.lion_df['segment_type'].isin(['G']))
        ]

        # A function to determine if an address is on the given side of the
        # street.
        def same_parity(a, b):
            return (a % 2) == (b % 2)

        for house, street_code in lgis:
            # First, normalize the house number.
            if type(house) == str:
                if '-' in house:
                    if street_code[0] == '4':
                        # If the street is in Queens and the number contains
                        # a hyphen, then convert it into the format tha LION
                        # uses by multiplying the first part by 1000 and adding
                        # the second.
                        a,b = house.split('-')
                        house = 1000*int(a) + int(b)
                    else:
                        # If not Queens, treat it as a range and use the first.
                        house = int(house.split('-')[0])
                else:
                    house = int(house)

            # Get a dataframe of the segments that could match the given
            # street code and house number.
            segments = df[
                (df['original_street_code'] == street_code[:6]) &
                (
                    (
                        (df['from_left'] <= house) & (df['to_left'] >= house) &
                        same_parity(df['from_left'], house)
                    ) | (
                        (df['from_right'] <= house) & (df['to_right'] >= house) &
                        same_parity(df['from_right'], house)
                    )
                )
            ]

            # If there are matches, check whether the address matches the
            # left or right side.
            for i,row in segments.iterrows():
                if (row['from_left'] != 0) and (
                    (row['from_left'] % 2) == (house % 2)
                ):
                    return {
                        'segment_id': row['segment_id'],
                        'physical_id': row['physical_id'],
                        'blockface_id': row['left_blockface_id'],
                        'side': 'L'
                    }
                if (row['from_right'] != 0) and (
                    (row['from_right'] % 2) == (house % 2)
                ):
                    return {
                        'segment_id': row['segment_id'],
                        'physical_id': row['physical_id'],
                        'blockface_id': row['right_blockface_id'],
                        'side': 'R'
                    }

    def _get_terminators(self, nodes):
        """
        Terminator segments are segments which exist at the point when a
        multi-roadbed street becomes a single roadbed.

        Geocoder uses these to prevent u-turns from one roadbed onto another
        at these nodes where the terminator segments meet.

        This function returns a dictionary which has a key for each
        physical_id that is a terminator where the value is a set of all
        terminator physical_ids which connect to that segment.

        Parameters
        ----------
        nodes : dict
            The nodes dictionary from `_get_nodes_dict`

        Returns
        -------
        dict of physical_id -> set of physical_ids
        """
        terminators = {}

        for node in nodes:
            # Get all the physical_ids that are terminator segments that
            # connect to the given nodes.
            pids = [
                self.lion_segments[sid]['physical_id']
                for sid in nodes[node]['segments']
                if self.lion_segments[sid]['segment_type'] == 'T'
            ]

            # Create a set of the physical_ids connected to this node and then
            # update that group with any other existing groups that overlap.
            group = set(pids)
            for p in pids:
                if p in terminators:
                    group.update(terminators[p])

            # TODO: I think this is supposed to be
            #   `for p in group:`
            # instead of in pids.
            for p in pids: # Add the group to the dictionary of terminators.
                terminators[p] = group

        return terminators

    @method_file_cache('network.pkl')
    def _build_networks(self, network_type):
        print("Building Routing Networks...")

        # Create a basic network for both LION and CSCL
        # This is needed to create the physical_id geometry, at least.
        lion_network = build_monodirectional_network(self.lion_df, 'segment_id')
        cscl_network = build_monodirectional_network(self.lion_df, 'physical_id')
        drop_internal_nodes(cscl_network)

        # Update nodes with whether it's an internal node or not.
        cscl_nodes = [n.split(':')[1] for n in cscl_network.nodes if 'node' in n]
        self.nodes_df['internal'] = ~self.nodes_df['node'].isin(cscl_nodes)
        nodes = _get_nodes_dict(self.nodes_df)

        def merge_geometry(group):
            """
            Merge the segments into a physical ID to get attributes that depend
            on the order of the segments.

            Parameters
            ----------
            group : pandas.DataFrame
                A subset of lion_df with all the segments of a single physical_id

            Returns
            -------
            shapely.LineString
            """
            # Get all the segments and nodes that are part of the given physical_id
            nodes = set(
                ['segment_id:%s' % i for i in group['segment_id']] +
                ['node:%s' % i for i in group['node_from']] +
                ['node:%s' % i for i in group['node_to']]
            )

            # Get a subgraph of lion_network with all the nodes and segments
            # and all the edges between them.
            subnetwork = lion_network.subgraph(nodes)

            # Next, order all of the nodes in subnetwork so that the geometry
            # will be in the right order to be merged.
            try:
                # Most physical_ids can be sorted via topological_sort
                stretch = nx.topological_sort(subnetwork)
                segments = [s.split(':')[1] for s in stretch if 'segment_id' in s]
            except:
                # But a few physical ids have cicrles/cycles, so in that case,
                # get an order by trying all combinations of start/end point
                # and use the first one that contains all the nodes.
                for a, b in itertools.combinations(nodes, 2):
                    try:
                        stretch = nx.shortest_path(subnetwork, a, b)
                        if len(stretch) == len(nodes):
                            break
                    except:
                        pass
                segments = [s.split(':')[1] for s in stretch if 'segment_id' in s]

            # Create a LineString from the individual coordinates in the
            # ordered list of segments.
            coords = [
                c for s in segments
                for c in self.lion_segments[s]['geometry'].coords
            ]
            coords = drop_consecutive_duplicates(coords)
            return LineString(coords)

        geometry = self.lion_df.groupby('physical_id').apply(
            merge_geometry
        ).rename('geometry')

        terminators = self._get_terminators(nodes)
        terminators = self.lion_df.groupby('physical_id').apply(
            lambda g: terminators.get(g.iloc[0]['physical_id'])
        ).rename('terminator_group')

        # Create the DataFrame and dict for physical_ids
        cscl_segments_df = _get_cscl_segments_df(self.lion_df)
        cscl_segments_df = cscl_segments_df.join(
            geometry, how='left'
        ).join(terminators, how='left')
        cscl_segments = cscl_segments_df.to_dict('index')

        if self.network_type == 'cscl':
            segments = cscl_segments
            network = cscl_network
        else:
            segments = self.lion_segments
            network = lion_network

        # Create a directional network of the given type
        # This network still has nodes
        node_network = build_directional_network(network, segments)

        # Create a network where segments connect directly to segments.
        segment_network = build_segment_network(
            node_network, default_cost_function(
                segments, nodes, turn_cost=100000, intersection_cost=0
            )
        )

        return (
            cscl_segments_df, cscl_segments, node_network, segment_network,
            self.nodes_df, nodes
        )

    def get_segment(self, segment):
        """
        Given a segment_id string, return a dictionary from `self.segments`.

        Parameters
        ----------
        segment : str
            A segment_id in a format accepted by `parse_geometry`.

        Returns
        -------
        dict
        """
        type_, id_, side = self.geometry_regex.match(segment).groups()
        return self.segments.get(id_, None)

    def normalize_segment_id(self, id_):
        if self.segment_column == 'segment_id':
            return id_.zfill(7)
        else:
            return str(int(id_))

    def parse_geometry(self, geometry):
        """
        Parse a "geometry" string and return a standardize geometry string,
        the type, id, and side of street.

        A geometry string is the type of geometry, followed by the id and
        optionally a side of street.

        For example:

            node:0055555M
            segment_id:0005555L
            physical_id:555R

        Parameters
        ----------
        geometry : str

        Returns
        -------
        geometry : str
            The geometry in a standardized format
        type : str
            node, segment_id or physical_id
        id : str
            The geometry ID
        side_of_street : str
            A letter for the side of street. One of: '', 'R', 'L', 'E', 'B'
        """
        try:
            type_, id_, letter = self.geometry_regex.match(geometry).groups()
            if not type_:
                if (id_.zfill(7) + letter) in self.nodes:
                    type_ = 'node'
                elif self.normalize_segment_id(id_) in self.segments:
                    type_ = self.segment_column

            assert type_ in ['node', self.segment_column]
            if type_ == 'node':
                id_ = id_.zfill(7) + letter
                letter = ''
                assert id_ in self.nodes
            else:
                assert letter in ['', 'R', 'L', 'E', 'B']
                id_ = self.normalize_segment_id(id_)
                assert id_ in self.segments


            return '%s:%s%s' % (type_, id_, letter), type_, id_, letter
        except:
            raise ValueError("Unrecognized geometry: %s" % geometry)


    def get_street_stretch_by_geometry(self, geometry_1, geometry_2,
                                       on_street_code=None):
        """
        Given two endpoint geometries, return a shortest path street stretch.
        Geometries can either be nodes or segments or a combination of the two.

        For nodes, if an optional on_street_code is provided, only start and
        end on streets on the given street code.

        This function works by adding temporary nodes to the Geocoder's
        segment_network called START and END. START connects to
        geometry_1 and END connects to geometry_2.

        Then find a shortest path from START to END and return it as a
        StreetStretch.

        This function will always return a result if there is a possible path
        from geometry_1 to geometry_2 even if it is not actually a "stretch"
        (not all along one on street). You can use the StreetStretch object's
        attributes to determine if the stretch is valid for your use case.

        Parameters
        ----------
        geometry_1, geometry_2 : str
            Start and endpoints for the stretch
        on_street_code : str, optional
            An on street code to start and end on.

        Returns
        -------
        StreetStretch
        """
        geometry_1, type_1, id_1, side_1 = self.parse_geometry(geometry_1)
        geometry_2, type_2, id_2, side_2 = self.parse_geometry(geometry_2)

        # Add start node (START -> geometry_1)
        if type_1 == 'node':
            # Since we will be routing on the segment network, for nodes connect
            # START to all segments that the node connects to.
            for segment in self.node_network[geometry_1]:
                # If on_street_code is provided, only connect to segments
                # on the given street.
                if (
                    (on_street_code is None) or
                    (on_street_code in self.get_segment(segment)['street_code'])
                ):
                    self.segment_network.add_edge('START', segment, weight=1)
        elif type_1 == self.segment_column:
            # For segments, connect the segment itself.
            # If side isn't given, allow both sides.
            if not side_1:
                for side in ['L', 'R']:
                    self.segment_network.add_edge('START', geometry_1 + side, weight=1)
            else:
                self.segment_network.add_edge('START', geometry_1, weight=1)

        # Add End Node in the same fashion, but connecting geometry_2 -> END
        if type_2 == 'node':
            for segment in self.node_network.predecessors(geometry_2):
                if (
                    (on_street_code is None) or
                    (on_street_code in self.get_segment(segment)['street_code'])
                ):
                    self.segment_network.add_edge(segment, 'END', weight=1)
        elif type_2 == self.segment_column:
            if not side_2:
                for side in ['L', 'R']:
                    self.segment_network.add_edge(geometry_2 + side, 'END', weight=1)
            else:
                self.segment_network.add_edge(geometry_2, 'END', weight=1)

        try:
            path = nx.bidirectional_dijkstra(
                self.segment_network, 'START', 'END', weight='weight'
            )[1]
            return StreetStretch(self, path[1:-1])
        finally:
            # Even if there's an error, make sure to remove START and END from
            # the network.
            for n in ['START', 'END']:
                self.segment_network.remove_node(n)

    def get_street_stretch_by_code(self, on_street_code, from_street_code,
                                   to_street_code):
        """
        Given an on street code, from street code and to street code,
        return a list of possible stretches.

        There can be more than one because sometime streets intersect multiple
        times.

        Parameters
        ----------
        on_street_code, from_street_code, to_street_code : str
            The street codes of the on, from and to streets

        Returns
        -------
        list of StreetStretch
            A list of stretches for all combinations of from and to
            intersections sorted from shortest to longest.
        """
        on_street_code = self.normalize_street_code(on_street_code)
        from_street_code = self.normalize_street_code(from_street_code)
        to_street_code = self.normalize_street_code(to_street_code)

        nodes_from = self._get_intersection(on_street_code, from_street_code)
        nodes_to = self._get_intersection(on_street_code, to_street_code)

        stretches = []

        for node_from in nodes_from:
            for node_to in nodes_to:
                if node_from != node_to:
                    try:
                        stretches.append(self.get_street_stretch_by_geometry(
                            'node:' + node_from, 'node:' + node_to,
                            on_street_code=on_street_code
                        ))
                    except:
                        pass

        return sorted(stretches, key=len)

    def __str__(self):
        return "Geocoder (lion_version=%s, crs=%s, index_crs=%s)" % (
            self.lion_version, self.crs, self.spatial_index_crs
        )
