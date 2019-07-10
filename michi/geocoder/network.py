import networkx as nx

def build_monodirectional_network(lion_df, column='physical_id'):
    """
    Builds a network from a lion DataFrame that simply connects
    from_node->segment->to_node for each row of the DataFrame.
    Does not take traffic direction into account.

    Note: Drops pedestrian only segments from the network before it's
    constructed.

    Parameters
    ----------
    lion_df: pandas.DataFrame
        A dataframe with rows for each segment in lion. Must contain columns
        `node_from`, `node_to` and `<id_column>`
    column: str
        One of 'physical_id' or 'segment_id'

    Returns
    -------
    nx.DiGraph
    """
    network = nx.DiGraph()

    for i, row in lion_df[lion_df['traffic_direction'] != 'P'].iterrows():
        segment_id = column + ':' + str(row[column])

        node_from = 'node:' + str(row['node_from'])
        node_to = 'node:' + str(row['node_to'])

        network.add_edge(node_from, segment_id)
        network.add_edge(segment_id, node_to)

    return network

def drop_internal_nodes(network):
    """
    Given a network from `build_monodirectional_network`, drop nodes that are
    "internal" to a segment. Nodes are internal if their edges only connect it
    with a single segment. Doesn't drop nodes that are dead_ends.

    Drops the nodes in place and returns a list of the nodes that were dropped.

    Parameters
    ----------
    network: nx.DiGraph
        A graph created by `build_monodirectional_network`.

    Returns
    -------
    list of nodes that were dropped from the network.
    """
    to_remove = []

    for obj in network:
        if 'node' in obj:
            if (
                # Not a dead end
                (len(network[obj]) > 0) and
                (len(list(network.predecessors(obj))) > 0) and

                # Comes and goes to the same physical ID
                (len(set(network[obj]).union(network.predecessors(obj))) == 1)
            ):
                to_remove.append(obj)

    for obj in to_remove:
        network.remove_node(obj)

    return to_remove

def build_directional_network(network, segments):
    """
    Given a segment->node network that doesn't have directionality
    (such as from `build_monodirectional_network`, build a network that
    adds directionality and side of street.

    Parameters
    ----------
    network: nx.DiGraph
        A graph from `build_monodirectional_network`
    segments: dict
        A dictionary of segments where each value is a dictionary that has
        a "traffic_direction" attribute.

        For example:

            {'123456': {'traffic_direction': 'T', ...}}

    Returns
    -------
    nx.DiGraph
    """
    network2 = nx.DiGraph()

    for obj in network:
        if 'node' in obj:
            continue

        nodes_from = list(network.predecessors(obj))
        nodes_to = list(network[obj])

        if len(nodes_from) == 0:
            nodes_from = nodes_to
        if len(nodes_to) == 0:
            nodes_to = nodes_from

        traffic_direction = segments[obj.split(':')[1]]['traffic_direction']

        right_blockface = obj + 'R'
        left_blockface =  obj + 'L'

        for node_from in nodes_from:
            for node_to in nodes_to:
                if traffic_direction == 'W':
                    network2.add_edge(node_from, left_blockface)
                    network2.add_edge(left_blockface, node_to)

                    network2.add_edge(node_from, right_blockface)
                    network2.add_edge(right_blockface, node_to)

                elif traffic_direction == 'A':
                    network2.add_edge(node_to, left_blockface)
                    network2.add_edge(left_blockface, node_from)

                    network2.add_edge(node_to, right_blockface)
                    network2.add_edge(right_blockface, node_from)

                elif traffic_direction == 'T':
                    network2.add_edge(node_from, right_blockface)
                    network2.add_edge(right_blockface, node_to)

                    network2.add_edge(node_to, left_blockface)
                    network2.add_edge(left_blockface, node_from)

    return network2

def default_cost_function(segments, nodes,
                          turn_cost=3000, intersection_cost=500,
                          min_travel_lanes_for_deadend_uturn=1,
                          deadend_uturn_cost=0):
    """
    Return the default cost function for building the segment->segment nework.

    Doesn't allow vehicles to go across a street from one blockface to another.

    Parameters
    ----------
    turn_cost : int
        The cost pentalty for making a turn (Default 3000)
    intersection_cost : int
        The cost pentalty for crossing an intersection (Default 500)
    min_travel_lanes_for_deadend_uturn : int
        The number of lanes required for the vehicle to make a u-turn on a
        dead end.  For trucks it should be >= 2. (Default 1)
    deadend_uturn_cost : int
        The additional cost penalty for making a uturn. Used to disincentivize
        making a u-turn on a dead end, but not disallow it. (Default 0)

    Returns
    -------
    function(geocoder, segment_id_1, segment_id_2, node_id)
    """

    def cost_function(segment_id_1, segment_id_2, node_id):
        """
        A function that returns the cost of going from segment_id_1 to
        segment_id_2.

        If the transition isn't possible, return None.

        Parameters
        ----------
        geocoder : gomi.geocoder.Geocoder
            A reference to a Geocoder is passed so that the function can
            look up attributes of the segments and node.
        segment_id_1, segment_id_2 : str
            The segments
        node_id : str
            The node connecting the two segments
        """
        segment_id_1 = segment_id_1[:-1]
        segment_id_2 = segment_id_2[:-1]

        travel_lanes = segments[segment_id_1]['number_travel_lanes']

        if (
            # No transitions between the same segment
            (segment_id_1 != segment_id_2) or
            # Unless it's a dead end with enough travel lanes
            (
                (nodes[node_id]['dead_end']) and
                (travel_lanes >= min_travel_lanes_for_deadend_uturn)
            )
        ):
            cost = 0

            # If crossing physical IDs, it's an intersection
            if (segments[segment_id_1]['physical_id'] !=
                segments[segment_id_2]['physical_id']):
                cost = intersection_cost

            # If the street codes don't overlap it's a turn
            if len(segments[segment_id_1]['street_code'].intersection(
                segments[segment_id_2]['street_code']
            )) == 0:
                cost = turn_cost

            # Include half the length of each segment
            cost += segments[segment_id_1]['len']/2
            cost += segments[segment_id_2]['len']/2

            # If it's a dead end, add u-turn penalty
            if segment_id_1 == segment_id_2:
                cost += deadend_uturn_cost

            return cost

    return cost_function

def build_segment_network(network, cost_function):
    """
    Generate a segment->segment network.

    cost_function is a user defined function that calculates the
    cost to travel from one segment to the next, for example, taking into
    account a turn.

    The function signature is:

        def cost_function(segment_id_1, segment_id_2, node_id)

    Return an integer in "feet" of the cost of transitioning from segment 1
    to segment 2.  If transition is impossible return None.
    (for example, because it would require a u-turn and the vehicle you're
    routing for is too big).

    See `default_cost_function` for an example.

    Parameters
    ----------
    network: nx.DiGraph
        A segment->node network such as returned by `build_directional_network`
    cost_function: function

    Returns
    -------
    nx.DiGraph
    """
    network2 = nx.DiGraph()

    for segment_1 in network:
        # Skip nodes
        if 'node' in segment_1:
            continue

        segment_id_1 = segment_1.split(':')[1]

        for node in network[segment_1]:
            node_id = node.split(':')[1]

            for segment_2 in network[node]:
                segment_id_2 = segment_2.split(':')[1]

                cost = cost_function(
                    segment_id_1, segment_id_2, node_id
                )

                if cost is not None:
                    network2.add_edge(segment_1, segment_2, weight=cost)

    return network2
