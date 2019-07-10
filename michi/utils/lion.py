import io
import zipfile

import geopandas as gp
import requests

from ..config import LION_URL, MICHI_HOME

LION_COLUMNS = {
    'Street': 'street',
    'SAFStreetName': 'special_address_street_name',
    'FeatureTyp': 'feature_type',
    'SegmentTyp': 'segment_type',
    'IncExFlag': 'include_exclude_flag',
    'RB_Layer': 'rb_layer',
    'NonPed': 'non_pedestrian',
    'TrafDir': 'traffic_direction',
    'SpecAddr': 'special_address_type',
    'FaceCode': 'face_code',
    'SeqNum': 'sequence_number',
    'StreetCode': 'street_code',
    'SAFStreetCode': 'special_address_street_code',
    'SegmentID': 'segment_id',
    'LocStatus': 'location_status',
    'LZip': 'left_zip',
    'RZip': 'right_zip',
    'LBoro': 'left_borough',
    'RBoro': 'right_borough',
    'L_CD': 'left_community_district',
    'R_CD': 'right_community_district',
    'LSubSect': 'left_sanitation_subsection',
    'RSubSect': 'right_sanitation_subsection',
    'SanDistInd': 'sanitation_district_indicator',
    'BoroBndry': 'borough_boundary',
    'XFrom': 'x_from',
    'YFrom': 'y_from',
    'XTo': 'x_to',
    'YTo': 'y_to',
    'ArcCenterX': 'arc_center_x',
    'ArcCenterY': 'arc_center_y',
    'CurveFlag': 'curve_flag',
    'Radius': 'radius',
    'NodeIDFrom': 'node_id_from',
    'NodeIDTo': 'node_id_to',
    'NodeLevelF': 'node_level_from',
    'NodeLevelT': 'node_level_to',
    'RW_TYPE': 'roadawy_type',
    'PhysicalID': 'physical_id',
    'GenericID': 'generic_id',
    'LBlockFaceID': 'left_blockface_id',
    'RBlockFaceID': 'right_blockface_id',
    'Status': 'status',
    'StreetWidth_Min': 'street_width_min',
    'StreetWidth_Max': 'street_width_max',
    'POSTED_SPEED': 'posted_speed',
    'Snow_Priority': 'snow_priority',
    'Number_Travel_Lanes': 'number_travel_lanes',
    'Number_Park_Lanes': 'number_park_lanes',
    'Number_Total_Lanes': 'number_total_lanes',
    'LLo_Hyphen': 'left_low_hyphen',
    'LHi_Hyphen': 'left_high_hyphen',
    'RLo_Hyphen': 'right_low_hyphen',
    'RHi_Hyphen': 'right_high_hyphen',
    'FromLeft': 'from_left',
    'ToLeft': 'to_left',
    'FromRight': 'from_right',
    'ToRight': 'to_right',
    'Join_ID': 'join_id',
}

def download_lion(version):
    version = version.lower()
    url = LION_URL % version
    response = requests.get(url)
    zip = zipfile.ZipFile(io.BytesIO(response.content))
    path = MICHI_HOME / version / 'download'
    zip.extractall(path)

    return path

def rename_lion_columns(df):
    return df.rename(columns=LION_COLUMNS)

def load_lion_gdf(version):
    download_path = download_lion(version)
    df = gp.read_file(download_path / 'lion' / 'lion.gdb', layer='lion')

    # Rename to standardized names
    df = rename_lion_columns(df)

    # Geometries get loaded as MultiLineString, convert to LineString
    df['geometry'] = df['geometry'].apply(lambda g: g.geoms[0])

    return df[list(LION_COLUMNS.values()) + ['geometry']]
