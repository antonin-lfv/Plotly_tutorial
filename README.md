# Plotly tutorial - Data analysis and Machine learning 
<br/>
<a href="https://www.python.org" class="fancybox" ><img align="right" width="110" height="110" src="https://user-images.githubusercontent.com/63207451/97306728-26fce600-185f-11eb-9784-14151a6b2c43.png"><a/>
	
## Introduction

Ce projet à pour objectif de présenter le module __Plotly__ qui est l'un des modules les plus utilisés pour faire de la visualisation de données avec Python. Plotly étant le plus compliqué mais également le plus interactif. Dans ce __README__ toutes les fonctions seront accompagnées du résultat. Le code complet pour ce repository est dans les fichiers sous le nom __code.py__ . Plotly utilise comme structure de données de base les dataframe.
<br/>

## Index
- [Analyse de données](#analyse-de-données)
	- [Importations](#importations)
	- [Première approche](#première-approche)
		- [Premier exemple](#premier-exemple)
		- [Deuxième exemple](#deuxième-exemple)
	- [Fonctions principales plotly.express](#fonctions-principales-plotlyexpress)
		- [Scatter plot](#scatter-plot)
		- [Courbe de tendance et densité](#courbe-de-tendance-et-densité)
		- [Error bars](#error-bars)
		- [Bar charts](#bar-charts)
		- [Graphiques de corrélations](#graphiques-de-corrélations)
		- [Scatter plot avec échelle des tailles des points](#scatter-plot-avec-échelle-des-tailles-des-points)
		- [Plot avec animation](#plot-avec-animation)
		- [Line Charts](#line-charts)
		- [Area charts](#area-charts)
		- [Pie charts](#pie-charts)
		- [Pie charts avec partie en dehors](#pie-charts-avec-partie-en-dehors)
		- [Donut charts](#donut-charts)
		- [Sunburst charts](#sunburst-charts)
		- [Treemaps](#treemaps)
		- [Histograms](#histograms)
		- [Boxplots](#boxplots)
		- [Violon plots](#violon-plots)
		- [Density contours](#density-contours)
		- [Heatmap](#heatmap)
		- [Point sur une carte](#point-sur-une-carte)
		- [Surface sur une carte](#surface-sur-une-carte)
		- [Polar plots](#polar-plots)
		- [Polar bar charts](#polar-bar-charts)
		- [Radar charts](#radar-charts)
		- [Coordonnées en 3D](#coordonnées-en-3d)
		- [Ternary charts](#ternary-charts)
	- [Graphiques multiples - Subplots](#graphiques-multiples---subplots)
		- [Pie subplots](#pie-subplots)
		- [Graphe subplots](#graphe-subplots)
		- [Les types de subplot](#les-types-de-subplot)
	- [Graphiques en 3D](#graphiques-en-3d)
		- [Surface](#surface)
		- [Nuage de points](#nuage-de-points)
	- [Slide bar](#slide-bar)
		- [Interactive plots](#interactive-plots)
		- [Sliders](#sliders)
		- [Sliders et sélecteur d'intervalles](#sliders-et-sélecteur-dintervalles)
		
		<br/>
- [Machine Learning](#machine-learning)
	- [Regression linéaire](#regression-linéaire)
	- [Regression surfacique en 3D](#regression-surfacique-en-3d)

<br/>

## Installation
<br/>
Installation : <br/>
<br/>


```py
pip install plotly
```
<br/>

Documentation [Plotly](https://plotly.com/python/) .
<br/>

# Analyse de données

## Importations

<br/>

```py
from plotly.offline import plot  # pour travailler en offline!
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
```
<br/>

## Première approche

<br/>

### Premier exemple

```py
wide_df = px.data.medals_wide()
fig = px.bar(wide_df, x="nation", y=["gold", "silver", "bronze"],
             title="Wide-Form Input, relabelled", # le titre
             labels={"value": "count", "variable": "medal"}, # le nom des axes
             color_discrete_map={"gold": "gold", "silver": "silver", "bronze": "#c96"}, # la couleur par classe
             template="simple_white") # couleur du fond
fig.update_layout(font_family="Rockwell", # police du texte
                  showlegend=False)
fig.add_annotation(text="over target!", x="South Korea", # ajouter un texte avec une flèche
                   y=49, arrowhead=1, showarrow=True)
fig.add_shape(type="line", line_color="salmon", line_width=3, opacity=1, line_dash="dot", #najouter une ligne horizontale
              x0=0, x1=1, xref="paper", y0=40, y1=40, yref="y")
plot(fig)
```

<br/>
<p align="center">
<img width="1131" alt="Capture d’écran 2020-11-23 à 20 50 08" src="https://user-images.githubusercontent.com/63207451/100008283-83571500-2dcd-11eb-9011-a36d86335e10.png">
<p/>

### Deuxième exemple

```py
fig = go.Figure(go.Pie(
    name = "",
    title = "languages populaire",
    values = [2, 5, 3, 2.5],
    labels = ["R", "Python", "Java Script", "Matlab"],
    text = ["textA", "TextB", "TextC", "TextD"],
    hovertemplate = "%{label}: <br>Popularity: %{percent} </br> %{text}"  # ce qu'on voit en passant la souris dessus
))
plot(fig)
```
<br/>
<p align="center">
<img width="901" alt="Capture d’écran 2020-11-23 à 20 52 48" src="https://user-images.githubusercontent.com/63207451/100008564-e5177f00-2dcd-11eb-8a5a-740cc6177d08.png">
<p/>

<br/>

## Fonctions principales plotly.express

<br/>

### Scatter plot

```py
df = px.data.iris() # pandas dataframe
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", title='Scatter')
plot(fig)
```
<br/>
<p align="center">
<img width="1184" alt="Capture d’écran 2020-11-23 à 21 41 58" src="https://user-images.githubusercontent.com/63207451/100013856-cddc8f80-2dd5-11eb-8dac-85110bdda97e.png">	
<p/>
<br/>

### Courbe de tendance et densité

```py
df = px.data.iris()
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",marginal_y="violin",
                 marginal_x="box", trendline="ols", template="simple_white")
# trendline = ols pour lineaire et lowess pour non linéaire
plot(fig)
```
<br/>
<p align="center">
<img width="1184" alt="Capture d’écran 2020-11-23 à 21 42 12" src="https://user-images.githubusercontent.com/63207451/100014025-1300c180-2dd6-11eb-9a02-214a1174b9c8.png">
<p/>
<br/>

### Error bars

```py
df = px.data.iris()
df["e"] = df["sepal_width"]/100 
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species", error_x="e", error_y="e")
plot(fig)
```
<br/>
<p align="center">
<img width="1184" alt="Capture d’écran 2020-11-23 à 21 42 24" src="https://user-images.githubusercontent.com/63207451/100014061-23b13780-2dd6-11eb-9996-866420a6b799.png">	
<p/>
<br/>

### Bar charts

```py
df = px.data.tips()
fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group")
# barmode="group" pour séparer les bars par color
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 42 41" src="https://user-images.githubusercontent.com/63207451/100014104-3461ad80-2dd6-11eb-8f06-02a555611f21.png">	
<p/>
<br/>

### Graphiques de corrélations

```py
df = px.data.iris()
fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 43 03" src="https://user-images.githubusercontent.com/63207451/100014133-404d6f80-2dd6-11eb-9f4e-f0bca9ee9b96.png">
<p/>
<br/>

### Scatter plot avec échelle des tailles des points

```py
df = px.data.gapminder()
fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp", size="pop", color="continent",
           hover_name="country", log_x=True, size_max=60)
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 43 15" src="https://user-images.githubusercontent.com/63207451/100014179-53f8d600-2dd6-11eb-810e-d1f9c64dad45.png">	
<p/>
<br/>

### Plot avec animation

```py
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country", facet_col="continent",
           log_x=True, size_max=45, range_x=[100,100000], range_y=[25,90])
# facet_col pour couper les données en plusieurs colonnes
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 43 34" src="https://user-images.githubusercontent.com/63207451/100014215-5fe49800-2dd6-11eb-9806-cd62c521106d.png">	
<p/>
<br/>

### Line charts

```py
df = px.data.gapminder()
fig = px.line(df, x="year", y="lifeExp", color="continent", line_group="country", hover_name="country",
        line_shape="spline", render_mode="svg")
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 43 48" src="https://user-images.githubusercontent.com/63207451/100014261-6d018700-2dd6-11eb-9d87-7945753a2a19.png">	
<p/>
<br/>

### Area charts

```py
df = px.data.gapminder()
fig = px.area(df, x="year", y="pop", color="continent", line_group="country")
plot(fig)
```
<br/>
<p align="center">
<img width="1187" alt="Capture d’écran 2020-11-23 à 21 43 58" src="https://user-images.githubusercontent.com/63207451/100014395-991d0800-2dd6-11eb-933a-1390d94c4e6b.png">	
<p/>
<br/>

### Pie charts

```py
df = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")
df.loc[df['pop'] < 2.e6, 'country'] = 'Other countries' # Represent only large countries
fig = px.pie(df, values='pop', names='country', title='Population of European continent')
fig.update_traces(textposition='inside', textinfo='percent+label')
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 44 17" src="https://user-images.githubusercontent.com/63207451/100014448-ab974180-2dd6-11eb-8256-220db37a275e.png">	
<p/>
<br/>

### Pie charts avec partie en dehors

```py
labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500, 2500, 1053, 500]

# pull is given as a fraction of the pie radius
fig = go.Figure(data=[go.Pie(labels=labels, values=values, pull=[0, 0, 0.2, 0])])
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 44 27" src="https://user-images.githubusercontent.com/63207451/100014497-ba7df400-2dd6-11eb-8931-121589445a3f.png">	
<p/>
<br/>

### Donut charts

```py
labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500, 2500, 1053, 500]
# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 44 36" src="https://user-images.githubusercontent.com/63207451/100014528-c5d11f80-2dd6-11eb-9467-2a775d101bd8.png">	
<p/>
<br/>

### Sunburst charts

```py
df = px.data.gapminder().query("year == 2007")
fig = px.sunburst(df, path=['continent', 'country'], values='pop',
                  color='lifeExp', hover_data=['iso_alpha'])
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 44 47" src="https://user-images.githubusercontent.com/63207451/100014559-d2557800-2dd6-11eb-928f-2f080b549af9.png">	
<p/>
<br/>

### Treemaps

```py
df = px.data.gapminder().query("year == 2007")
fig = px.treemap(df, path=[px.Constant('world'), 'continent', 'country'], values='pop',
                  color='lifeExp', hover_data=['iso_alpha'])
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 44 58" src="https://user-images.githubusercontent.com/63207451/100014601-e26d5780-2dd6-11eb-92bf-97314f41bb47.png">	
<p/>
<br/>

### Histograms

```py
df = px.data.tips()
fig = px.histogram(df, x="total_bill", y="tip", color="sex", hover_data=df.columns)
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 45 11" src="https://user-images.githubusercontent.com/63207451/100014649-f5802780-2dd6-11eb-80b4-3af4f9555091.png">	
<p/>
<br/>

### Boxplots

```py
df = px.data.tips()
fig = px.box(df, x="day", y="total_bill", color="smoker", notched=True)
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 45 20" src="https://user-images.githubusercontent.com/63207451/100014681-0335ad00-2dd7-11eb-9830-34997f3030ff.png">	
<p/>
<br/>

### Violon plots

```py
df = px.data.tips()
fig = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all", hover_data=df.columns)
plot(fig)
```
<br/>
<p align="center">
<img width="1163" alt="Capture d’écran 2020-11-23 à 21 45 28" src="https://user-images.githubusercontent.com/63207451/100014702-0d57ab80-2dd7-11eb-8173-80050ccde0a5.png">	
<p/>
<br/>

### Density contours

```py
df = px.data.iris()
fig = px.density_contour(df, x="sepal_width", y="sepal_length")
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 45 43" src="https://user-images.githubusercontent.com/63207451/100014731-1a749a80-2dd7-11eb-82df-de6468a28d7d.png">	
<p/>
<br/>

### Heatmap

```py
df = px.data.iris()
fig = px.density_heatmap(df, x="sepal_width", y="sepal_length", marginal_y="histogram")
plot(fig)

fig = px.imshow([[1, 20, 30],
                 [20, 1, 60],
                 [30, 60, 1]])
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 45 59" src="https://user-images.githubusercontent.com/63207451/100014759-252f2f80-2dd7-11eb-9d56-630b029358a2.png">	
<p/>
<br/>

### Point sur une carte

```py
df = px.data.carshare()
fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon", color="peak_hour", size="car_hours",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                  mapbox_style="carto-positron")
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 46 10" src="https://user-images.githubusercontent.com/63207451/100014788-311af180-2dd7-11eb-951b-a803e3e458ed.png">	
<p/>
<br/>

### Surface sur une carte

```py
df = px.data.election()
geojson = px.data.election_geojson()

fig = px.choropleth_mapbox(df, geojson=geojson, color="Bergeron",
                           locations="district", featureidkey="properties.district",
                           center={"lat": 45.5517, "lon": -73.7073},
                           mapbox_style="carto-positron", zoom=9)
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 46 25" src="https://user-images.githubusercontent.com/63207451/100014823-43952b00-2dd7-11eb-9696-d81e18e1667a.png">	
<p/>
<br/>

### Polar plots

```py
df = px.data.wind()
fig = px.scatter_polar(df, r="frequency", theta="direction", color="strength", symbol="strength",
            color_discrete_sequence=px.colors.sequential.Plasma_r)
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 46 40" src="https://user-images.githubusercontent.com/63207451/100014889-59a2eb80-2dd7-11eb-8e5f-ce79cc9a9100.png">	
<p/>
<br/>

### Polar bar charts

```py
df = px.data.wind()
fig = px.bar_polar(df, r="frequency", theta="direction", color="strength", template="plotly_dark",
            color_discrete_sequence= px.colors.sequential.Plasma_r)
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 46 55" src="https://user-images.githubusercontent.com/63207451/100014925-6aebf800-2dd7-11eb-9316-e808a0563909.png">	
<p/>
<br/>

### Radar charts

```py
df = px.data.wind()
fig = px.line_polar(df, r="frequency", theta="direction", color="strength", line_close=True,
            color_discrete_sequence=px.colors.sequential.Plasma_r)
plot(fig)
```
<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 47 08" src="https://user-images.githubusercontent.com/63207451/100014956-763f2380-2dd7-11eb-93fa-c342ade13527.png">	
<p/>
<br/>

### Coordonnées en 3D

```py
df = px.data.election()
fig = px.scatter_3d(df, x="Joly", y="Coderre", z="Bergeron", color="winner", size="total", hover_name="district",
                  symbol="result", color_discrete_map = {"Joly": "blue", "Bergeron": "green", "Coderre":"red"})
plot(fig)
```
<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 33 48" src="https://user-images.githubusercontent.com/63207451/100017987-02534a00-2ddc-11eb-8d5e-08b43e4f9516.png">	
<p/>
<br/>

### Ternary charts

```py
df = px.data.election()
fig = px.scatter_ternary(df, a="Joly", b="Coderre", c="Bergeron", color="winner", size="total", hover_name="district",
                   size_max=15, color_discrete_map = {"Joly": "blue", "Bergeron": "green", "Coderre":"red"} )
plot(fig)
```

<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 47 50" src="https://user-images.githubusercontent.com/63207451/100015002-8d7e1100-2dd7-11eb-8f25-14d3424f0daa.png">	
<p/>

<br/>

## Graphiques multiples - Subplots

<br/>

### Pie subplots

```py
labels = ["US", "China", "European Union", "Russian Federation", "Brazil", "India","Rest of World"]
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]]) # 'domain' for pie subplots
fig.add_trace(go.Pie(labels=labels, values=[16, 15, 12, 6, 5, 4, 42], name="GHG Emissions"),1, 1)
fig.add_trace(go.Pie(labels=labels, values=[27, 11, 25, 8, 1, 3, 25], name="CO2 Emissions"),1, 2)
fig.update_traces(hole=.4, hoverinfo="label+percent+name")
fig.update_layout(
    title_text="Global Emissions 1990-2011",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='GHG', x=0.18, y=0.5, font_size=20, showarrow=False),
                 dict(text='CO2', x=0.82, y=0.5, font_size=20, showarrow=False)])
plot(fig)
```

<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 22 11 39" src="https://user-images.githubusercontent.com/63207451/100016645-08e0c200-2dda-11eb-8965-e8b24587f971.png">
<p/>
<br/>

### Graphe subplots

```py
df = px.data.iris() # pandas dataframe
fig = make_subplots(rows=1, cols=2,subplot_titles=("Plot 1", "Plot 2")) #titre de chaque subplot
fig.add_trace(go.Scatter(x=df["sepal_width"], y=df["sepal_length"]),1,1)
fig.add_trace(go.Scatter(x=df["sepal_width"], y=df["sepal_length"]),1,2)
fig.update_layout(title_text="subplot")
# pour changer les axes de chaque subplot :
fig.update_xaxes(title_text="xaxis 1 title", showgrid=False, row=1, col=1) # sans grid x
fig.update_xaxes(title_text="xaxis 2 title", range=[0, 10], row=1, col=2)
fig.update_yaxes(title_text="yaxis 1 title", showgrid=False,row=1, col=1) # sans grid y
fig.update_yaxes(title_text="yaxis 2 title", range=[0, 10], row=1, col=2)
plot(fig)
```

<br/>

```py
# pour avoir l'axe X en commun :
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02)

# pour avoir l'axe Y en commun
fig = make_subplots(rows=2, cols=2, shared_yaxes=True)
```

<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-23 à 22 15 19" src="https://user-images.githubusercontent.com/63207451/100016992-84db0a00-2dda-11eb-81ee-03184e77c9f4.png">
<p/>
<br/>

### Les types de subplot

```
xy: 2D Cartesian subplot type for scatter, bar, etc. This is the default if no type is specified.

scene: 3D Cartesian subplot for scatter3d, cone, etc.

polar: Polar subplot for scatterpolar, barpolar, etc.

ternary: Ternary subplot for scatterternary.

mapbox: Mapbox subplot for scattermapbox.

domain: Subplot type for traces that are individually positioned. pie, parcoords, parcats, etc.

trace type: A trace type name (e.g. bar, scattergeo, carpet, mesh, etc.) 
which will be used to determine the appropriate subplot type for that trace.
```

<br/>

## Graphiques en 3D

<br/>

### Surface

```py
z_data = df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")
fig = go.Figure(data=[go.Surface(z=z_data, colorscale='IceFire')]) # Z1 liste de liste
fig.update_layout(title='Mountain')
plot(fig)
```

<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 30 54" src="https://user-images.githubusercontent.com/63207451/100017741-ad173880-2ddb-11eb-8643-78795b0e3e57.png">
<p/>
<br/>

### Nuage de points

```py
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species', size='petal_length', size_max=18,symbol='species', opacity=0.7)
plot(fig)
```
<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 31 34" src="https://user-images.githubusercontent.com/63207451/100017787-bbfdeb00-2ddb-11eb-9ca5-603cca91a999.png">
<p/>

<br/>

## Slide bar

<br/>

### Interactive plots

```py
np.random.seed(1)

x0 = np.random.normal(2, 0.4, 400)
y0 = np.random.normal(2, 0.4, 400)
x1 = np.random.normal(3, 0.6, 600)
y1 = np.random.normal(6, 0.4, 400)
x2 = np.random.normal(4, 0.2, 200)
y2 = np.random.normal(4, 0.4, 200)

# Create figure
fig = go.Figure()

# Add traces
fig.add_trace(go.Scatter(x=x0,y=y0,mode="markers",marker=dict(color="DarkOrange")))
fig.add_trace(go.Scatter(x=x1,y=y1,mode="markers",marker=dict(color="Crimson")))
fig.add_trace(go.Scatter(x=x2,y=y2,mode="markers",marker=dict(color="RebeccaPurple")))

# Add buttons that add shapes
cluster0 = [dict(type="circle",xref="x", yref="y",x0=min(x0), y0=min(y0),x1=max(x0), y1=max(y0),line=dict(color="DarkOrange"))]
cluster1 = [dict(type="circle",xref="x", yref="y",x0=min(x1), y0=min(y1),x1=max(x1), y1=max(y1),line=dict(color="Crimson"))]
cluster2 = [dict(type="circle",xref="x", yref="y",x0=min(x2), y0=min(y2),x1=max(x2), y1=max(y2),line=dict(color="RebeccaPurple"))]

fig.update_layout(updatemenus=[dict(type="buttons",buttons=[
                dict(label="None",
                     method="relayout",
                     args=["shapes", []]),
                dict(label="Cluster 0",
                     method="relayout",
                     args=["shapes", cluster0]),
                dict(label="Cluster 1",
                     method="relayout",
                     args=["shapes", cluster1]),
                dict(label="Cluster 2",
                     method="relayout",
                     args=["shapes", cluster2]),
                dict(label="All",
                     method="relayout",
                     args=["shapes", cluster0 + cluster1 + cluster2])]
		     ,)])
fig.update_layout(title_text="Highlight Clusters",showlegend=False,)
plot(fig)
```

<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 38 56" src="https://user-images.githubusercontent.com/63207451/100019492-6e36b200-2dde-11eb-941e-373380479b9f.png">
<p/>

<br/>

### Sliders

```py
df = px.data.gapminder()
fig = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year", animation_group="country",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

fig["layout"].pop("updatemenus") # optional, drop animation buttons
plot(fig)
```

<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 46 18" src="https://user-images.githubusercontent.com/63207451/100019524-7bec3780-2dde-11eb-88f6-e82ed1f54875.png">
<p/>

<br/>

### Sliders et sélecteur d'intervalles

```py
# Load data
df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv")
df.columns = [col.replace("AAPL.", "") for col in df.columns]

# Create figure
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(df.Date), y=list(df.High)))

# Set title
fig.update_layout(title_text="Time series with range slider and selectors")

# Add range slider
fig.update_layout(xaxis=dict(rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")])
            ),rangeslider=dict(visible=True),type="date"))
plot(fig)
```

<br/>
<p align="center">
<img width="1267" alt="Capture d’écran 2020-11-23 à 22 47 10" src="https://user-images.githubusercontent.com/63207451/100019545-860e3600-2dde-11eb-8bd2-01372384f569.png">
<p/>

# Machine Learning

## Regression linéaire

```py
from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
fig.show()
```

<br/>
<p align="center">
<img width="1166" alt="Capture d’écran 2020-11-24 à 10 45 21" src="https://user-images.githubusercontent.com/63207451/100077394-9821c000-2e42-11eb-9700-8f103b2000c5.png">
<p/>

## Regression surfacique en 3D

Le contenu de df_final est disponible dans les fichiers du github.

```py
from sklearn.svm import SVR

mesh_size = .02
margin = 0

df = df_final

X = df[['x', 'y']]
y = df['hauteurs']

# Condition the model on sepal width and length, predict the petal width
model = SVR(C=1.)
model.fit(X, y)

# Create a mesh grid on which we will run our model
x_min, x_max = X.x.min() - margin, X.x.max() + margin
y_min, y_max = X.y.min() - margin, X.y.max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# Run model
pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

# Generate the plot
fig = px.scatter_3d(df, x='x', y='y', z='hauteurs')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
plot(fig)
```

<br/>
<p align="center">
<img width="1093" alt="Capture d’écran 2020-11-24 à 11 25 44" src="https://user-images.githubusercontent.com/63207451/100082229-411ee980-2e48-11eb-89c2-4a480a5eb613.png">
<p/>


<br/>
<p align="center">
<a href="#plotly-tutorial---data-analysis-and-machine-learning"> haut de la page </a>

<p/>
<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
