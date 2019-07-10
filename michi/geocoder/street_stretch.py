from collections import OrderedDict
import itertools

import pandas as pd
from shapely.geometry import LineString
from shapely.ops import linemerge

from ..utils.utils import drop_consecutive_duplicates

class StreetStretch:
    """
    An object that represents a "stretch" - usually a list of segments along
    a single street.

    It is created from a list of segments and provides functions to get
    the on/from/to streets, geometry and length of the stretch.
    """
    def __init__(self, geocoder, segments, side=None, ignore_traffic_direction=False):
        """
        Parameters
        ----------
        geocoder : Geocoder
            A reference to a Geocoder, usally the one that created this stretch
            through get_street_stretch_by_geometry or get_street_stretch_by_code.
            But a StreetStretch can be created manually from a list of segments.
            The given segments must exist in the given Geocoder.
        segments : list of str
            A list of segments in the normalized format (physical_id:<id>)
        side : str, optional
            A side of street to drive on - either 'L' or 'R' (Default None)
        """
        # Reference to the geocoder that created this stretch
        self.geocoder = geocoder
        self._segments = segments
        self.side = side

        self._on_from_to = None

        self.ignore_traffic_direction = ignore_traffic_direction
        self.n_network = self.geocoder.non_directional_node_network \
            if ignore_traffic_direction \
            else self.geocoder.node_network
        self._segments_against_traffic = None

    def get_segments(self, include_side_of_street=True):
        """
        Return the list of segment IDs.

        Parameters
        ----------
        include_side_of_street : bool, optional
            Whether to include the side of street character (L or R) with the
            segment IDs. (Default True)

        Returns
        -------
        list of str
        """
        if include_side_of_street:
            return [s.split(':')[1] for s in self._segments]
        else:
            return [s.split(':')[1][:-1] for s in self._segments]

    @property
    def segments(self):
        return self.get_segments(False)

    def __len__(self):
        """
        Returns
        -------
        int
            The number of segments on the stretch.
        """
        return len(self._segments)

    @property
    def length(self):
        """
        Returns
        -------
        The length of the stretch in feet.
        """
        return sum([self.geocoder.segments[i]['len'] for i in self.segments])

    def _get_on_streets(self, segments):
        """
        Given a list of segments, return a list of sets of street codes that the
        segments are on. Sets of street codes are returned because sometimes
        multiple street codes refer to the same physical street.

        If a street transitions into another street, consider it the same
        street. For example, Hogan Place turns into Leonard Street over three
        segments. The first segment is just Hogan Place, then one segment is
        both Hogan and Leonard, and the final one is just Leonard. Since the
        streets overlapped, it will be considered one street.

        Parameters
        ----------
        segments : list of str

        Returns
        -------
        list of sets of str
        """
        streets = []
        for segment_id in segments:
            # Get the set of street codes for each segment
            street_codes = self.geocoder.segments[segment_id]['street_code']

            # Check if this segment's street codes overlap with any of the
            # already processed segments, if so, add this segment's street
            # codes to the existing set.
            match = False
            for i in range(len(streets)):
                if streets[i].intersection(street_codes):
                    streets[i] = streets[i].union(street_codes)
                    match = True

            if not match:
                streets.append(street_codes)

        return streets

    @property
    def number_of_on_streets(self):
        return len(self._get_on_streets(self.segments))

    @property
    def start_and_end_on_same_street(self):
        """
        Returns
        -------
        bool
            Whether or not the street that the stretch starts on is the same
            as the one that it ends on.
        """
        # Get the on streets using _get_on_streets to handle street codes
        # that change even though the street physically stays the same.
        segments = self.segments
        on_streets = self._get_on_streets(segments)

        # Get the on street codes specifically for the endpoints.
        endpoints = self._get_on_streets([segments[0], segments[-1]])

        # If there is only one street code set for the endpoints, then they
        # must start and end on the same street.
        if len(endpoints) == 1:
            return True

        # Otherwise, check if each of the endpoints intersects with any of the
        # streets the strtech goes on. Since `_get_on_from_to` handles
        # transitioning street codes, this ensures that even if the endpoints
        # themselves have different street codes, if the street codes overlap
        # during the stretch, then it will be counted as starting and stopping
        # on the same street.
        for street in on_streets:
            if endpoints[0].intersection(street):
                if endpoints[1].intersection(street):
                    return True

        return False

    @property
    def number_of_turns(self):
        """
        Return the number of "turns" on as stretch, which is the number of times
        that a segment's street codes don't match the next segment's street
        codes at all.

        Returns
        -------
        int
        """
        turns = 0
        previous_street = None
        for segment_id in self.segments:
            street = self.geocoder.segments[segment_id]['street_code']
            if previous_street and not previous_street.intersection(street):
                turns += 1
            previous_street = street
        return turns

    def get_geometry(self, merge=True):
        """
        Return the geometry of the stretch, either as a list of geometries for
        each segment, or as one single geometry.

        Parameters
        ----------
        merge : bool, optional
            Whether to merge the segment geometries into a single geometry.
            (Default True)

        Returns
        -------
        shapely.LineString or list of shapely.LineString
        """
        geometries = []

        for segment in self.get_segments():
            segment = self.geocoder.segment_column + ':' + segment
            segment_id, side_of_street = self.geocoder.parse_geometry(segment)[2:]
            geometry = self.geocoder.segments[segment_id]['geometry']
            traffic_direction = self.geocoder.segments[segment_id]['traffic_direction']

            # Flip the geometry if direction of travel is reverse of
            # the drawn direction
            if (
                    (((traffic_direction == 'A') or
                      ((traffic_direction == 'T') and (side_of_street == 'L')))
                     and not self.ignore_traffic_direction)
                    or
                    (self.ignore_traffic_direction and side_of_street == 'L')

            ):
                # Create a new LineString from the coordinates reversed
                geometries.append(LineString(geometry.coords[::-1]))
            else:
                geometries.append(geometry)

        if merge:
            # Manually Merge the geometries by getting all of the coordinates
            # from each segment in order
            coords = [c for g in geometries for c in g.coords]

            # Drop consecutive points - necessary?
            coords = drop_consecutive_duplicates(coords)

            return LineString(coords)
        else:
            return geometries

    @property
    def endpoint_nodes(self):
        """
        Return a tuple of node IDs with the start and end nodes of the stretch.

        Returns
        -------
        (str, str)
            A tuple (start_node, end_node)
        """
        return (
            # Get the node that comes before the first segment
            self.geocoder.parse_geometry(
                list(self.n_network.predecessors(
                    self._segments[0]
                ))[0]
            )[2],

            # And the node that comes after the last
            self.geocoder.parse_geometry(
                list(self.n_network[self._segments[-1]])[0]
            )[2]
        )

    @property
    def on_from_to(self):
        if not self._on_from_to:
            self._on_from_to = self.get_on_from_to()
        return self._on_from_to

    @property
    def on_streets(self):
        if not self._on_from_to:
            self._on_from_to = self.get_on_from_to()
        return self._on_streets

    @property
    def from_streets(self):
        if not self._on_from_to:
            self._on_from_to = self.get_on_from_to()
        return self._from_streets

    @property
    def to_streets(self):
        if not self._on_from_to:
            self._on_from_to = self.get_on_from_to()
        return self._to_streets

    @property
    def against_traffic(self):
        segments_against_traffic = []

        if self._segments_against_traffic:
            return True

        if self.ignore_traffic_direction:
            for segment in self.get_segments():
                segment = self.geocoder.segment_column + ':' + segment
                segment_id, side_of_street = self.geocoder.parse_geometry(segment)[2:]
                traffic_direction = self.geocoder.segments[segment_id]['traffic_direction']
                # add segment if it is against traffic
                if (
                        (traffic_direction == 'A' and side_of_street == 'R') or
                        (traffic_direction == 'W' and side_of_street == 'L')
                ):
                    segments_against_traffic.append(segment_id)

            if len(segments_against_traffic) > 0:
                self._segments_against_traffic = segments_against_traffic
                return True

        return False

    def get_segments_against_traffic(self):
        if self.against_traffic:
            return self._segments_against_traffic

    def get_on_from_to(self):
        """
        Return a list of dictionaries of On/From/To street options. Each
        dictionary has `on_street`, `on_street_code`, `from_street`,
        `from_street_code`, `to_street` and `to_street_code`.

        If the on/from/to is unambiguous, then it will return a list of length
        one. When ambiguous, return more than one option, with the "most likely"
        option first. Likelihood is determined by the number of times that
        street appears along the stretch, how often it appears globally in NYC,
        and whether it is at the start or end of the stretch.

        Returns
        -------
        list of dict
        """

        def get_streets_from_segments(segments, sort_cols, segments_dict):
            """
            Return a list of (street, street_code) tuples for the given segments
            sorted in "likelihood" order.

            Parameters
            ----------
            segments: list
                A list of segment IDs
            sort_cols: list
                A subset of ['count', 'start', 'end', 'global_count'] used to
                sort the streets into likelihood order. For the on street, use
                all. For from/to, use count and global_count.
            """
            # Iterate through all the segments` street/street_code pairs and
            # Add them to the streets dictionary.
            streets = {}
            for i, segment in enumerate(segments):
                pairs = segments_dict[segment]['street_street_code_pair']
                for street, street_code in pairs:
                    pair = (street, street_code)
                    if pair not in streets:
                        streets[pair] = {
                            'street': street,
                            'street_code': street_code,
                            # Keep track of occurances of this pair.
                            'global_count': len(self.geocoder.streets[street_code]['df']),
                            'count': 0, 'start': 0, 'end': 0,
                        }

                    # If the street appears at the start or end, favor it.
                    if i == 0:
                        streets[pair]['start'] += 1
                    if i == (len(segments) - 1):
                        streets[pair]['end'] += 1

                    # Count the number of occurances of that pair.
                    streets[pair]['count'] += 1

            # Return (street, street_code) tuples sorted by likelihood
            return [
                (street['street'], street['street_code']) for street in
                sorted(streets.values(), key=lambda street: tuple(
                    -street[col] for col in sort_cols
                ))
            ]

        # Get the unique on segment IDs and use them to get on street options.
        on_segments = self.segments
        on_streets = get_streets_from_segments(
            on_segments, ['count', 'start', 'end', 'global_count'],
            self.geocoder.segments
        )

        def drop_overlapping_streets(a, b):
            """
            Return the streets in a that are not in b unless a and b are the
            same.
            """
            a_codes = [s[1] for s in a]
            b_codes = [s[1] for s in b]
            if set(a_codes).difference(b_codes):
                return [s for s in a if s[1] not in b_codes]

            return a

        def get_node_streets(node):
            """A function to get street options for the nodes."""
            # If the node is a dead end, just return DEAD END.
            if self.geocoder.nodes[node]['dead_end']:
                return [(
                    'DEAD END', 'dead_end'
                )]

            # Get the segments at the node, not inculding the on segments.
            segments = set([
                s for s in self.geocoder.nodes[node]['segments']
                #if self.geocoder.lion_segments[s]['physical_id'] not in on_segments
            ])
            segments2 = set([
                s for s in segments if
                self.geocoder.lion_segments[s]['physical_id'] not in on_segments
            ])
            if segments2:
                segments = segments2

            streets = get_streets_from_segments(
                segments, ['count', 'global_count'], self.geocoder.lion_segments
            )

            return drop_overlapping_streets(streets, on_streets)

        # Get from node, to node and the respective street options
        from_node, to_node = self.endpoint_nodes
        from_streets = get_node_streets(from_node)
        to_streets = get_node_streets(to_node)
        on_streets = drop_overlapping_streets(on_streets, from_streets + to_streets)

        # Cache the results on the object for future lookup
        self._on_streets = on_streets
        self._from_streets = from_streets
        self._to_streets = to_streets

        # Return a list of dictionaries of the combinations of on/from/to
        return [
            {
                'on_street': os, 'from_street': fs, 'to_street': ts,
                'on_street_code': osc, 'from_street_code': fsc,
                'to_street_code': tsc
            }
            for (os, osc), (fs, fsc), (ts, tsc)
            in itertools.product(on_streets, from_streets, to_streets)
        ]
