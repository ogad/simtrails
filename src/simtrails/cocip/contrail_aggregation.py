import geopandas as gpd
import numpy as np

# This is somewhat deprecated.
# You should instead use the contrails as individual segments


def contrail_shape(df_contrail):
    from shapely import LineString, Point

    if len(df_contrail) == 1:
        return Point(df_contrail.longitude.iloc[0], df_contrail.latitude.iloc[0])

    lats = df_contrail.latitude
    lons = df_contrail.longitude

    # check for 360 degree jumps in longitude
    if (np.abs(lons.diff()) > 180).any():
        lons %= 360

    shape = LineString(gpd.points_from_xy(lons, lats))
    return shape


def flight_contrails(df_flight):
    # split dataframe after "False" values of df_flight.continuous
    split_indices = np.where(df_flight.continuous == False)[0] + 1
    split_indices = [0] + list(split_indices) + [len(df_flight)]
    dfs = [
        df_flight.iloc[start_index:end_index]
        for start_index, end_index in zip(split_indices[:-1], split_indices[1:])
    ]
    dfs = [df for df in dfs if not df.empty]
    contrails = []
    for df_contrail in dfs:
        id = (
            df_contrail.flight_id.iloc[0]
            + "-"
            + df_contrail.formation_time.iloc[0].strftime("%H%M")
            + df_contrail.formation_time.iloc[-1].strftime("%H%M")
        )
        length = df_contrail.segment_length.sum()
        width = df_contrail.width.median()

        iwp = df_contrail.iwp.median()
        iwc_gm3 = df_contrail.iwc_gm3.median()
        eff_radius = df_contrail.eff_radius.median()

        rf_lw = (df_contrail.rf_lw * df_contrail.segment_length).sum()
        rf_sw = (df_contrail.rf_sw * df_contrail.segment_length).sum()
        rf_net = (df_contrail.rf_net * df_contrail.segment_length).sum()

        age = df_contrail.age.max()

        contrail = {
            "id": id,
            "length": length,
            "width": width,
            "geometry": contrail_shape(df_contrail),
            "iwp": iwp,
            "iwc_gm3": iwc_gm3,
            "eff_radius": eff_radius,
            "rf_lw": rf_lw,
            "rf_sw": rf_sw,
            "rf_net": rf_net,
            "age": age,
        }
        contrails.append(contrail)

    contrails = gpd.GeoDataFrame(contrails)
    contrails.set_geometry("geometry", crs="EPSG:4326", inplace=True)

    return contrails


def aggregate_contrails(df):
    from tqdm import tqdm

    tqdm.pandas()
    return (
        df.groupby("time")
        .progress_apply(lambda df: df.groupby("flight_id").apply(flight_contrails))
        .set_index("id")
        .reset_index()
    )
    return df.groupby("flight_id").apply(flight_contrails).set_index("id").reset_index()


def flight_from_trail(contrail_id):
    return contrail_id[:-9]


def first_formation_time(contrail_id):
    return contrail_id[-8:-4]


def last_formation_time(contrail_id):
    return contrail_id[-4:]


def contrails_match(id_first, id_second):
    if flight_from_trail(id_first) != flight_from_trail(id_second):
        return False
    if (
        int(first_formation_time(id_second)) - int(last_formation_time(id_first)) > 0
    ):  # second contrail starts after first ends
        return False
    if (
        int(last_formation_time(id_second)) - int(first_formation_time(id_first)) < 0
    ):  # second contrail ends before first starts
        return False
    return True


def old_id_getter(prev_contrails):
    def old_id(id2):
        val = prev_contrails.id[
            prev_contrails.id.apply(lambda id1: contrails_match(id1, id2))
        ].values
        if len(val) == 0:
            return None
        if len(val) == 1:
            return val[0]
        raise ValueError("Multiple matches found")

    return old_id
