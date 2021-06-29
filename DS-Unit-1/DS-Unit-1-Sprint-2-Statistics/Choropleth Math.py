# # sample below from https://plotly.com/python/choropleth-maps/
#
# from urllib.request import urlopen
# import json
# with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
#     counties = json.load(response)
#
# import pandas as pd
# df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
#                    dtype={"fips": str})
#
# print (df)
# print (counties)
#
# import plotly.express as px
# px.choropleth()
# fig = px.choropleth(df, geojson=counties, locations='fips', color='unemp',
#                            color_continuous_scale="Viridis",
#                            range_color=(0, 12),
#                            scope="usa",
#                            labels={'unemp':'unemployment rate'}
#                           )
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()
#


# # MAKING USA ONLY PLOTS
# import plotly.graph_objects as go
#
# import pandas as pd
# df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv')
#
# fig = go.Figure(data=go.Choropleth(
#     locations=df['code'], # Spatial coordinates
#     z = df['total exports'].astype(float), # Data to be color-coded
#     locationmode = 'USA-states', # set of locations match entries in `locations`
#     colorscale = 'Reds',
#     colorbar_title = "Millions USD",
# ))
#
# fig.update_layout(
#     title_text = '2011 US Agriculture Exports by State',
#     geo_scope='usa', # limite map scope to USA
# )
#
# fig.show()
# end of sample https://plotly.com/python/choropleth-maps/


# USING THE PLOTLY ON GUNS DATA:-

import plotly.graph_objects as go

import pandas as pd
url = 'https://raw.githubusercontent.com/BuzzFeedNews/nics-firearm-background-checks/master/data/nics-firearm-background-checks.csv'
df = pd.read_csv(url)
url2 = 'https://raw.githubusercontent.com/plotly/datasets/master/2011_us_ag_exports.csv'
df_2 = pd.read_csv(url2)


#selecting a particular year
df_2020_10 = df[df['month'] == '2020-10']
# print (df_2020_10)
# print (df_2)
# print (df_2020_10['state'][0])
print (df_2020_10['state'].isin(df_2['state']))
# df_2020_10.drop(index = [8,11,21,41,49], inplace= True)
# print (df_2020_10.shape)
final = pd.merge(df_2020_10, df_2, on = 'state')
# print (final)

#plotting
fig = go.Figure(data=go.Choropleth(
    locations=final['code'], # Spatial coordinates
    z = final['totals'].astype(float), # Data to be color-coded
    locationmode = 'USA-states', # set of locations match entries in `locations`
    colorscale = 'Reds',
    colorbar_title = "Guns Sold"
))

fig.update_layout(
    title_text = '2020-10 US Gun Sales by State',
    geo_scope='usa', # limite map scope to USA
)

fig.show()

# https://plotly.com/javascript/map-animations/

