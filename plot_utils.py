import sys, os
import pandas as pd
import geopandas as gpd
import xarray as xr
from pylab import *
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
import plotly.express as px
## colors
import branca.colormap as cm
## under RUNNING ACTIVITIES, cal_heatmap function
import calplot ## https://github.com/tomkwok/calplot.git (pip install calplot)
import vector_utils as vu



################################
## WEB MAPS
################################

def webmap_of_grid(grid_file):
    """creates leaflet webmap with lat-lon coordinates in bottom-right corner"""
    grid = gpd.read_file(grid_file)
    if grid_file.endswith("LUCinLA_grid_8858.gpkg"):
        grid = grid[grid.CEL_projec.str.contains(str("PARAGUAY"))]
        grid = grid.set_crs(8858)
        grid = grid.to_crs({'init': 'epsg:4326'}) 
    elif grid_file.endswith("AI4B_grid_UTM31N.gpkg"):
        grid = grid.set_crs(32631)
        grid = grid.to_crs({'init': 'epsg:4326'})
    elif grid_file.endswith("cape_grid_utm32S.gpkg"):
        grid = grid.set_crs(32734)
        grid = grid.to_crs({'init': 'epsg:4326'})
    ## create folium map
    m = folium.Map(location=(grid.iloc[0].geometry.centroid.y, grid.iloc[0].geometry.centroid.x), zoom_start=5, width="%100", height="%100", epsg="EPSG4326")
    # add basemap tiles 
    tile = folium.TileLayer(tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
                            attr = 'ESRI', name = 'ESRI Satellite', overlay = False, control = True, show = True).add_to(m)
    ## add lucinLA paraguay grids
    fg = folium.map.FeatureGroup(name='grid', show=False).add_to(m)
    for k, v in grid.iterrows():
        poly=folium.GeoJson(v.geometry, style_function=lambda x: { 'color': 'black' , 'fillOpacity':0} )
        popup=folium.Popup("UNQ: "+str(v.UNQ)) 
        poly.add_child(popup)
        fg.add_child(poly) 
    folium.LatLngPopup().add_to(m)
    m.add_child(folium.ClickForMarker(popup="user added point- highlight coords in bottom right corner, then hover over point and CTRL+C / CTRL+V into for next function"))
    folium.LayerControl(collapsed=False, autoZIndex=False).add_to(m)
    return m

def click_to_coords(raw_coords, class_name):
    split_coords = raw_coords.split(":")
    new_coords = float(split_coords[0]), float(split_coords[1])
    return new_coords, class_name

def xy_marker(feature, marker_dict):
    return marker_dict.get(feature.marker)

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))




################################
## COLORS
################################

def random_color_feats(features):
    feat_colors=[]
    for i in range(len(features)):
        hex_color = '#%02x%02x%02x' % tuple(np.random.choice(range(256), size=3))
        feat_colors.append(hex_color)
    FeatColor_dict = dict(zip(features, feat_colors))
    return FeatColor_dict


def linear_cmap(df, col_name, color_list):
    vmax = df[col_name].max()
    vmin = df[col_name].min()

    linear = cm.LinearColormap(color_list, vmin=vmin, vmax=vmax)
    
    colormap = colormap_choice.scale(vmin, vmax)
    colormap.caption = col_name
    return colormap


################################
## FIGURES
################################

def histo_treat_control(gdf, field):
    C_vils = gdf[gdf['treated'] == 0]
    T_vils = gdf[gdf['treated'] == 1]
    x = C_vils[field]
    y = T_vils[field]
    plt.hist(y, 10, alpha=0.5, label='Treatment overlap %')
    plt.hist(x, 10, alpha=0.5,  label='Control overlap %')
    plt.legend(loc='upper right')
    plt.show()
    
def plot_time_series(grid_file, web_coord_list, VI):
    grid = gpd.read_file(grid_file)
    img_dir="/home/sandbox-cel/capeTown/monthly/landsat/vrts/"     
    crs="EPSG:32734"        
    # matplotlib figure parameters
    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes()    
    # color by each coordinate pair 
    color_list = ["goldenrod", "green", "purple", "gray", "pink"]
    # initialize list to append smoothed TS max's 
    y_maxs=[]
    y_mins=[]     
    item_num = 0    
    # transform web mercator XY coordinates to coordinates in projection rasters are in
    rast_list = sorted([os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith(".vrt") and VI+"_20" in img])
    date_ts=[i[:-4].split("_")[-1] for i in rast_list]   
    for i in web_coord_list:
        TS_list=[]
        xy_coords= vu.transform_point_coords(inepsg="EPSG:4326", outepsg=crs, XYcoords=i[0])
        for rast in rast_list:
            with xr.open_dataset(rast ) as xrimg:
                point = xrimg.sel(x=xy_coords[0], y=xy_coords[1], method="nearest")
                TS_list.append(point.band_data.values[0])
            y_maxs.append(max(TS_list))
            y_mins.append(min(TS_list))      
        ax.plot(date_ts, TS_list, color=color_list[item_num], label='_nolegend_')   ##
        item_num+=1
    y_ax_max=max(y_maxs)
    y_ax_min=min(y_mins)   
    ax.set_ylim([y_ax_min, y_ax_max])   
    ax.set_xticklabels(date_ts, rotation=45, ha='right', fontsize= 6 )
    ax.set_xlabel(date_ts[0] + " to " + date_ts[-1], fontsize = 10)   
    ax.set_ylabel(VI, fontsize = 10)      
    ax.legend(loc='upper right')
    return fig

def plot_climate(out_dir, temp, Tstat, Pstat, climate_csv='/home/l_sharwood/code/demilunes/village_climate.csv'):
    df = pd.read_csv(climate_csv)
    df=df.set_index('villID')
    Edf = df[[i for i in df.columns.to_list() if(i.startswith(temp)+"_" and i.endswith("_"+Tstat)) ]]
    print(Edf.columns.to_list())
    Edf.columns = [i.replace(temp+"_", "").replace("_"+Tstat, "") for i in Edf.columns.to_list()]
    Edf.insert(0, "version", ["CHIRTSe"]*180, True)
    df_long_T = pd.melt(Edf.reset_index(), id_vars=['villID', 'version'], value_vars=Edf.reset_index().columns.to_list())
    df_long_T.columns = ['villID', 'version', 'time', temp+"_"+Tstat]
    df_long_T[temp+"_"+Tstat] = [i-df_long_T[temp+"_"+Tstat].min() for i in df_long_T[temp+"_"+Tstat]]
    Pdf = df[[i for i in df.columns.to_list() if (i.startswith("precip") and i.endswith(Pstat)) ]]
    Pdf.columns = [i.replace("precip_", "").replace("_"+Pstat, "").replace("_UTM32", "") for i in Pdf.columns.to_list()]
    Pdf.insert(0, "version", ["CHIRPS"]*180, True)
    df_long_P = pd.melt(Pdf.reset_index(), id_vars=['villID', 'version'], value_vars=Pdf.reset_index().columns.to_list())
    df_long_P.columns = ['villID', 'version', 'time', 'precip_'+Pstat]
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = sorted(list(set(df_long_P['time'])))
    precip_avgs = df_long_P[["time", "precip"+"_"+Pstat]].groupby(['time']).mean()[['precip'+"_"+Pstat]]
    temp_avgs =  df_long_T[["time", temp+"_"+Tstat]].groupby(['time']).mean()[[temp+"_"+Tstat]]
    print(temp_avgs)
    fig.add_trace( go.Bar(x=dates, y=precip_avgs['precip'+"_"+Pstat], name='precip'+"_"+Pstat),  secondary_y=False )
    fig.add_trace( go.Scatter(x=dates, y=temp_avgs[temp+"_"+Tstat], name=temp+"_"+Tstat),  secondary_y=True )
    fig.update_layout(  title_text="village climate: "+'precip'+" "+Pstat+" and "+temp+" "+Tstat )
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="monthly "+Pstat+" "+" precip", secondary_y=False)
    fig.update_yaxes(title_text="monthly "+Tstat+" "+temp, secondary_y=True)
    fig.show()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.write_html(os.path.join(out_dir, "vill_clim_"+'precip'+"_"+Pstat+"_"+temp+"_"+Tstat+".html"))

def plot_climate_avgs(out_dir, temp, Tstat, Pstat, climate_csv='/home/l_sharwood/code/demilunes/village_climate.csv'):
    df = pd.read_csv(climate_csv)
    Edf = df[[i for i in df.columns.to_list() if(i.startswith(temp) and i.endswith(Tstat)) ]]
    Edf.columns = [i.replace(temp+"_", "").replace("_"+Tstat, "") for i in Edf.columns.to_list()]
    Pdf = df[[i for i in df.columns.to_list() if (i.startswith("precip") and i.endswith(Pstat)) ]]
    #Pdf.columns = [i.replace("precip_", "").replace("_"+Pstat, "").replace("_UTM32", "") for i in Pdf.columns.to_list()]
    print(Pdf)
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    dates = sorted(list(set([i.replace("precip_", "").replace("_"+Pstat, "") for i in Pdf.columns.to_list()])))
    fig.add_trace( go.Bar(x=dates, y=Pdf.iloc[0].to_list(), name='precip'+"_"+Pstat),  secondary_y=False )
    fig.add_trace( go.Scatter(x=dates, y=Edf.iloc[0].to_list(), name=temp+"_"+Tstat),  secondary_y=True )
    fig.update_layout(  title_text="village climate: "+'precip'+" "+Pstat+" and "+temp+" "+Tstat )
    fig.update_xaxes(title_text="month")
    fig.update_yaxes(title_text="monthly "+Pstat+" "+" precip", secondary_y=False)
    fig.update_yaxes(title_text="monthly "+Tstat+" "+temp, secondary_y=True)
    fig.show()
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    fig.write_html(os.path.join(out_dir, "AVGvil_"+Pstat+'P_'+Tstat+temp.replace("Tmin", "TLow").replace("Tmax", "THigh")+'.html'))



################################
## RUNNING PLOTS
################################

def plot_heatmap(df, out_name):
    # df = df.set_index("time").sort_index()
    df = df.iloc[::1, :]  ## subset every 1 points to keep file size down
    # for routes polylines
    all_routes_xy = []
    for k, v in df.groupby(['file']):
        route_lats = v.lat.to_list()
        route_lons = v.lon.to_list()
        all_routes_xy.append(list(zip(route_lats, route_lons)))
    # df = df[start:end].dropna()
    heatmap = folium.Map(location=[np.mean(df.lat), np.mean(df.lon)], control_scale=False, zoom_start=12)
    tile = folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='ESRI', name='ESRI Satellite', overlay=True, control=True, show=True).add_to(heatmap)
    cluster = plugins.HeatMap(data=[[la, lo] for la, lo in zip(df.lat, df.lon)], name="heatmap", min_opacity=0.15,
                              max_zoom=10, radius=9, blur=8)
    heatmap.add_child(cluster)
    fg = folium.FeatureGroup("routes")
    folium.PolyLine(locations=all_routes_xy, weight=1.0, opacity=1.0, color='red', control=True, show=True).add_to(fg)
    fg.add_to(heatmap)
    folium.LayerControl().add_to(heatmap)
    heatmap.save(out_name)
    return heatmap


def plot_3d(df, out_fi):
    route_3d = px.scatter_3d(df, x='lon', y='lat', z='ele', color='ele')
    route_3d.update_traces(marker=dict(size=2.0), selector=dict(mode='markers'))
    route_3d.write_html(out_fi)
    return route_3d


def cal_heatmap(df, col_name, mpl_cmap, out_name):
    df['ascent'] = df.ascent.astype(float)
    df = df[df.ascent < 20000]
    df['ascent_ft'] = [float(i) * float(3.28084) for i in df.ascent.to_list()]
    df['ascent_m'] = np.where(df['ascent'] >= 2000, 2000, df['ascent'])
    # Create a column containing the month
    df['month'] = pd.to_datetime(df['PST']).dt.to_period('M')
    df['week'] = pd.to_datetime(df['PST']).dt.to_period('W')
    df = df.reindex(sorted(df.columns), axis=1)
    df['Year'] = [int(str(i).split("-")[0]) for i in df.PST.astype(str)]
    df = df.sort_values('PST')
    events = pd.Series([i for i in df[col_name]],
                       index=[i for i in df['PST'].astype('datetime64[ns]')])
    cal_fig = calplot.calplot(events, suptitle="running " + col_name + " per day", cmap=mpl_cmap, colorbar=True, yearlabel_kws={'fontname': 'sans-serif'})
    plt.savefig(out_name)
    return cal_fig[0]

