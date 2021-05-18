<p align="center">
<a href="https://www.python.org" class="fancybox" ><img width="70" height="70" src="https://user-images.githubusercontent.com/63207451/97306728-26fce600-185f-11eb-9784-14151a6b2c43.png"><a/>
<a href="https://www.python.org" class="fancybox" ><img width="200" height="70" src="https://user-images.githubusercontent.com/63207451/118670736-29bd2980-b7f7-11eb-8aa4-ad41fa393ed1.png"><a/>
	<p/>
<br/>

<h1 align="center"><b>Plotly tutorial - Data analysis and Machine learning</b></h1>
<br/>
	
<p align="center">
	Ce repo à pour objectif de présenter le module <b>Plotly</b> qui est l'un des modules les plus utilisés pour faire de la visualisation de données avec Python. Plotly étant le plus compliqué mais également le plus interactif. Il est construit sur 2 composantes principales, à savoir <b>plotly.express</b> qui est basé sur l'utilisation des dataframes pandas, et qui est simple et rapide, et <b>plotly.graph_objects</b> qui est beaucoup plus puissant, et beaucoup plus personnalisable, il est basé sur la POO. Dans ce <b>README</b> toutes les fonctions seront accompagnées du résultat. Le code complet pour ce repository est dans les fichiers sous le nom <b>code.py</b> .
<br/>
	<p/>

<br/>

> Pour comprendre plus en détails comment plotly fonctionne, et pour personnaliser au maximum vos graphiques, je vous invite à consulter mon [article](https://github.com/antonin-lfv/Plotly_tutorial/blob/main/README.md) sur plotly. ( bientôt disponible )
<br/>

# Index

- [Plotly.Express](#plotlyexpress) 
	- [Scatter plot](#scatter-plot)
		- [Exemple simple](#Exemple-simple)
		- [Superposition de figures](#Superposition-de-figures)
		- [Subplots](#subplots)
		- [Animations](#animations)
		- [Range Slider](#range-slider)
		- [Rectangles et lignes](#rectangles-et-lignes)
		- [Marges statistiques](#marges-statistiques)
		- [Curseurs](#curseurs)
		- [Plot 3D](#plot-3d)
	- [Bar chart](#bar-chart)
		- [Premier exemple](#premier-exemple)
		- [Indicateurs marginaux](#indicateurs-marginaux)
	- [Pie chart](#pie-chart)
		- [Exemple basique](#exemple-basique)
	- [Polar bar charts](#polar-bar-charts)
	- [Points sur une carte](#Points-sur-une-carte)
	- [Machine Learning](#Machine-learning)
		- [Regression linéaire](#regression-linéaire)
	  	- [UMAP](#UMAP)
		- [t-SNE](#t-SNE)
	- [Graphique de corrélation](#graphique-de-corrélation)

<br/>

- [Plotly.Graph_Objects](#plotlygraph_objects)
	- [Subplots](#subplots-Go)
	- [Scatter](#scatter)
		- [Scatter basique](#scatter-basique)
		- [Annotations](#annotations)
		- [Droite et plage de valeurs](#droite-et-plage-de-valeurs)
	- [Pie chart](#Pie-chart-Go)
	- [Violin chart](#Violin-chart)
	- [Histogramme/Bar](#Histogrammebar)
	- [Graphiques en 3D](#graphiques-en-3d)
		- [Surface](#surface)
		- [Nuage de points](#nuage-de-points) 
	- [Réseau de neurones](#réseau-de-neurones)
	- [Regression surfacique en 3D](#regression-surfacique-en-3d)

# Installation
<br/>
Installation : <br/>
<br/>


```py
pip install plotly
```
<br/>

Documentation [Plotly](https://plotly.com/python/) .
<br/>

# Importations

<br/>

```py
from plotly.offline import plot  # pour travailler en offline!
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import load_digits
from umap import UMAP # pip install umap-learn
from sklearn.manifold import TSNE
import networkx as nx
```
<br/>

# Plotly.Express

Plotly express, importée en tant que __px__, est la partie de plotly permettant de créer rapidement et simplement n'importe quel graphique en 1 ligne de code. Son interet est notamment basé sur le fait qu'elle marche parfaitement bien avec les __DataFrames__ de Pandas, mais on peut également travailler avec des listes ou tout autre type de données. Son utilisation est limitée notamment dans la conception de subplots. Vous verrez dans cette section quelles sont les fonctions les plus utiles de __plotly express__, et comment les personnaliser au maximum.
<br/>

Voici la syntaxe globale d'un code utilisant plotly express :
```py
fig = px.chart_type(df, parameters)
fig.update_layout("layout_parameters or add annotations")
fig.update_traces("further graph parameters")
fig.update_xaxis() # ou update_yaxis
fig.add_trace() # ajouter une figure avec graph_objects
plot(fig)
```

<br/>

## Scatter plot

### Exemple simple

On personnalise les figures de plotly express suivant ce modèle : 

```py
df = px.data.iris()

fig = px.scatter(df, x="sepal_width", # nom de la colonne du dataframe
                 y="sepal_length", # nom de la colonne du dataframe
                 color="species", # nom de la colonne du dataframe
                 )

fig.update_layout(title="Scatter avec px et ajout d'une figure avec Go",
                  font_family="Rockwell")

fig.update_xaxes(title_text='largeur sepales')
fig.update_yaxes(title_text='longueur sepales')

fig.add_trace(
    go.Scatter(
        x=[2, 4],
        y=[4, 8],
        mode="lines",
        marker=dict(color="gray"),
        name="droite",
        showlegend=True) # True par défaut
)

plot(fig)
```
<br/>

<img width="1419" alt="Capture d’écran 2021-05-18 à 09 39 56" src="https://user-images.githubusercontent.com/63207451/118611343-05dcf200-b7bd-11eb-98ba-cde2bcecf80e.png">

<br/>

### Superposition de figures

Si on veut superposer des courbes avec les données du même dataset on écrit : 

```py
df = px.data.iris()
fig = px.line(df, y=["sepal_width", "sepal_length", "petal_width"],
              #text="species_id" # pour ajouter l'id de l'espèce au dessus de chaque point
              color_discrete_map={"sepal_width":"blue", 
	      			  "sepal_length":"black",
				  "petal_width":"green" }, # couleur de chaque ligne
            )
plot(fig)
```

<br/>
<img width="1419" alt="Capture d’écran 2021-05-18 à 09 42 40" src="https://user-images.githubusercontent.com/63207451/118611691-6a984c80-b7bd-11eb-92cb-bf433f8fa8b7.png">
<br/>

### Subplots

On peut aussi séparer les figures en plusieurs, c'est le seul moyen de faire des subplots avec plotly express ! Il y a 2 types de subplots, soit en lignes, soit en colonnes, spécifié par le paramètre facet_col ou facet_row. Partons du code suivant : 

```py
df = px.data.iris()
fig = px.line(df, y=["sepal_width", "sepal_length", "petal_width"],
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"sepal_width":"blue", 
	      			  "sepal_length":"black",
				  "petal_width":"green" }, # couleur de chaque ligne
            )
plot(fig)
```

Alors on obtient ces résultats en fonction du paramètre de séparation :

<br/>

| facet_col="species" | facet_row="species" |
|---------------------|---------------------|
|<img width="600" alt="Capture d’écran 2021-05-18 à 09 44 19" src="https://user-images.githubusercontent.com/63207451/118611980-a59a8000-b7bd-11eb-8ae9-2b8a673d7e5d.png">|<img width="600" alt="Capture d’écran 2021-05-18 à 09 45 25" src="https://user-images.githubusercontent.com/63207451/118612164-cb278980-b7bd-11eb-9586-abeb9e3a7fb9.png">|

<br/>

### Animations

On fait bouger les points qui suivent l'évolution des données au fil des années :

```py
df = px.data.gapminder()
df_fr=df[df['country']=='France']
df_us=df[df['country']=='United States']
df = pd.concat([df_fr, df_us])

fig = px.scatter(df,
        y="gdpPercap",
        x="year",
        color="country",
        title="évolution pib france et USA",
        range_x=[1952,2007],
        range_y=[df['gdpPercap'].min(), df['gdpPercap'].max()],
        animation_frame="year")
plot(fig)
```

<br/>
<img width="1403" alt="Capture d’écran 2021-05-18 à 09 49 41" src="https://user-images.githubusercontent.com/63207451/118612783-67519080-b7be-11eb-8bc7-9bc9e9c1c55d.png">
<br/>

### Range Slider

Personnalisation des axes avec un range slider visible :

```py
df = px.data.carshare()
fig = px.line(df, y="car_hours",
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"car_hours":"black"}, # couleur de chaque ligne
            )
fig.update_xaxes(rangeslider_visible=True)
plot(fig)
```

<br/>
<img width="1403" alt="Capture d’écran 2021-05-18 à 09 51 49" src="https://user-images.githubusercontent.com/63207451/118613064-ac75c280-b7be-11eb-8648-05aa6843c10b.png">
<br/>

### Rectangles et lignes

On peut colorier une plage de valeurs, ou ajouter une ligne :

```py
df = px.data.stocks(indexed=True)
fig = px.line(df, facet_col="company",
              facet_col_wrap=2 # nombre de figure par ligne
              )
fig.add_hline( # ou vline pour verticale avec x=...
              y=1, line_dash="dot",
              annotation_text="1er janvier 2018",
              annotation_position="bottom right")

fig.add_vrect( # ou hrect pour horizontal
              x0="2018-09-24", x1="2018-12-18",
              col=2, # numéro de la colonne (les figures de droites)
              annotation_text="24/09 au 18/12 2018",
              annotation_position="top left",
              fillcolor="red", opacity=0.2, line_width=0.1)

fig.add_hrect( # ou hrect pour horizontal
              y0=1.1, y1=1.7,
              col=1, # numéro de la colonne (les figures de droites)
              annotation_text="1.1 à 1.7",
              annotation_position="top right",
              fillcolor="blue", opacity=0.15, line_width=0.4)

plot(fig)
```

<br/>
<img width="1403" alt="Capture d’écran 2021-05-18 à 09 52 53" src="https://user-images.githubusercontent.com/63207451/118613217-d4652600-b7be-11eb-8b8f-bdebc1c54626.png">
<br/>

### Marges statistiques

On ajoute un indicateur statistique sur chacune des variables du scatter :

```py
df = px.data.iris()
fig = px.scatter(df, x="sepal_length", # données
                 color="species", # couleur par expèce
                 marginal_x='box', # marge en boxplot
                 marginal_y='violin', # marge en violon
                 trendline="ols" # courbe de tendances
                 )
plot(fig)
```
<br/>
<img width="1403" alt="Capture d’écran 2021-05-18 à 09 53 30" src="https://user-images.githubusercontent.com/63207451/118613295-e941b980-b7be-11eb-9930-e1c1a496e8f6.png">
<br/>

### Curseurs

Curseurs qui apparaissent avec survol de la souris sur un point du graphique :

```py
df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color="country",
              title="curseurs")
fig.update_traces(mode="markers+lines") # courbe avec ligne et points apparent
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
plot(fig)
```

<br/>
<img width="1403" alt="Capture d’écran 2021-05-18 à 09 54 47" src="https://user-images.githubusercontent.com/63207451/118613484-168e6780-b7bf-11eb-9be4-204fb57f00ab.png">
<br/>

### Plot 3D

```py
df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species', size='petal_length', size_max=18,symbol='species', opacity=1)
plot(fig)
```

<br/>
<img width="1411" alt="Capture d’écran 2021-05-18 à 10 42 21" src="https://user-images.githubusercontent.com/63207451/118620412-c666d380-b7c5-11eb-87e6-346920499cca.png">
<br/>

## Bar chart

### Premier exemple

Simple Barchart, on colorie sur une colonne et on sépare par couleur :

```py
df = px.data.tips()
fig=px.bar(df,
           x="sex",
           y="total_bill",
           color="smoker",
           barmode="group")
fig.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
plot(fig)
```

<br/>
<img width="1413" alt="Capture d’écran 2021-05-18 à 16 12 29" src="https://user-images.githubusercontent.com/63207451/118666930-da292e80-b7f3-11eb-98c8-860f7dada52b.png">
<br/>

### Indicateurs marginaux

On va tester 2 types de marges, violon et boxplot, qu'on ajoute avec le paramètre marginal :

```py
df = px.data.iris()
fig = px.histogram(df, x="sepal_length",
                   nbins=50, # on choisi le nombre de barres
		   marginal=''
                   )
plot(fig)
```

|marginal='violin'|marginal='box'|
|-----------------|--------------|
|<img width="600" alt="Capture d’écran 2021-05-18 à 10 06 10" src="https://user-images.githubusercontent.com/63207451/118615088-ae408580-b7c0-11eb-8306-bc30dfbd6c46.png">|<img width="600" alt="Capture d’écran 2021-05-18 à 10 07 45" src="https://user-images.githubusercontent.com/63207451/118615324-e5af3200-b7c0-11eb-91e7-becb37f542fe.png">|

<br/>

## Pie chart

### Exemple basique

```py
df = px.data.tips()
fig = px.pie(df, values='tip', # ce qu'on compte
             names='day', # sur quoi on tri
             color='day',
             hole=.3, # donut chart
             color_discrete_map={'Thur':'lightblue', # couleur spécifique par valeur
                                 'Fri':'lightred',
                                 'Sat':'gold',
                                 'Sun':'green'})
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 16 04 01" src="https://user-images.githubusercontent.com/63207451/118665541-ab5e8880-b7f2-11eb-903d-5a6ed3e70a11.png">
<br/>

## Polar bar charts

On va représenter ici la force des vents, sur un diagramme polaire, avec la direction comme angle, et la force en taille :

```py
df = px.data.wind()
fig = px.bar_polar(df, r="frequency", theta="direction", color="strength", 
		template="seaborn", # couleur de fond
                color_discrete_sequence= px.colors.sequential.Plasma_r)
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 16 07 18" src="https://user-images.githubusercontent.com/63207451/118666061-2627a380-b7f3-11eb-942d-25e2d9491bac.png">	
<br/>

## Points sur une carte

```py
df = px.data.carshare()
fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon", color="peak_hour", size="car_hours",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                  mapbox_style="carto-positron")
plot(fig)
```
<br/>
<img width="1166" alt="Capture d’écran 2020-11-23 à 21 46 10" src="https://user-images.githubusercontent.com/63207451/100014788-311af180-2dd7-11eb-951b-a803e3e458ed.png">	
<br/>

## Machine Learning

### Regression linéaire

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

### UMAP

```py
digits = load_digits()
umap_2d = UMAP(random_state=0)
umap_2d.fit(digits.data)

projections = umap_2d.transform(digits.data)
fig = px.scatter(
    projections, x=0, y=1,
    color=digits.target.astype(str), labels={'color': 'digit'}
)
plot(fig)
```

<br/>
<p align="center">
<img width="1399" alt="Capture d’écran 2021-04-30 à 13 02 14" src="https://user-images.githubusercontent.com/63207451/116686674-6e9e2f00-a9b4-11eb-97e0-a65286464dc0.png">
<p/>


### t-SNE

```py
df = px.data.iris()
features = df.loc[:, :'petal_width']
tsne = TSNE(n_components=3, random_state=0)
projections = tsne.fit_transform(features, )

fig = px.scatter_3d(
    projections, x=0, y=1, z=2,
    color=df.species, labels={'color': 'species'}
)
fig.update_traces(marker_size=8)
plot(fig)
```

<br/>
<p align="center">
<img width="1328" alt="Capture d’écran 2021-04-30 à 13 03 58" src="https://user-images.githubusercontent.com/63207451/116686752-8fff1b00-a9b4-11eb-8733-a3491e45f36e.png">
<p/>

<br/>

## Graphique de corrélation

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

<br/>

# Plotly.Graph_Objects

Plotly __graph_objects__, importée en tant que __go__ est la partie de Plotly utilisant la __POO__, pour créer des graphiques très complets. On va y retrouver la 
plupart des fonctions de plotly express. 
Vous verrez dans cette section quelles sont les fonctions les plus utiles de plotly __graph_objetcs__ et comment les personnaliser
au maximum.
<br/>

Voici la syntaxe globale d'un code utilisant plotly graph_objects :
```py
fig = go.Figure() # création de la figure, ou alors make_subplots

fig.add_TypeTrace("parameters") # on ajoute des figure avec TypeTrace qui est pie, scatter, surface, etc..
...
fig.add_TypeTrace("parameters")

fig.update_traces("parameters")
fig.update_layout("parameters")
plot(fig)
```

## Subplots Go

Pour un subplots on utilise un autre moyen pour initialiser la figure :

```py
fig = make_subplots(rows=2, cols=2,
                    #column_widths=[0.6, 0.4],
                    #row_heights=[0.3, 0.3, 0.3],
                    subplot_titles=["", "", ""],
                    specs=[[{'type': 'xy'}, {'type': 'domain'}], # 1er ligne
                           [{'type': 'xy', 'colspan': 2}, None]], # 2e ligne, dont la 1er colonne s'etend sur celle de droite
                    # si on s'etend sur une colonne on utilise rowspan
                    # il existe plusieurs type de specs : xy, domain, scene, polar, ternary, mapbox
                    #horizontal_spacing=0.2,
                    #vertical_spacing=0.5
                    )

"Ensuite on ajoute les figure normalement, en indiquant juste l'emplacement du graphique avec les paramètres row et col"
x=np.linspace(-4,4)
# 1er figure
fig.add_scatter(x=np.linspace(-4,4), y=np.tanh(x),
                marker=dict(color='green'),
                row=1, col=1, name="tangente hyperbolique"
                )
# 2e figure
fig.add_pie(labels=['oui', 'non'],
            values=[201902,192981], row=1, col=2)
# 3e figure
fig.add_traces( # je vais ajouter 2 courbes en même temps sur la dernière ligne
    [
        go.Scatter(x=x, y=np.square(x), mode='markers+lines', name='x²'),
        go.Scatter(x=x,y=-np.square(x), name='-x²')
    ],
    rows=2, cols=1
)

plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 19 28" src="https://user-images.githubusercontent.com/63207451/118658192-72bbb080-b7ec-11eb-995c-5a6ce06180f3.png">
<br/>

## Scatter

### Scatter basique

On pourra utiliser scattergl pour des gros datasets.

```py
x = np.linspace(-2, 2, 100)
y = 1/(1+np.exp(-x))

fig = go.Figure()
# 1ere solution, on ajoute toutes les figures avec le même appel (pour appliquer une seule fois certains parametres)
fig.add_traces([go.Scatter(x=x, y=y, mode='markers', name='sigmoid'),
               go.Scatter(x=x, y=-y, mode='lines', name='negative sigmoid')] )
# 2e solution, on ajoute les figures indépendemment
fig.add_scatter(x=x, y=1+y, mode='markers', name='sigmoid+1')
fig.add_scatter(x=x, y=1-y, mode='lines', name='1-sigmoid')

fig.update_layout(title="sigmoid")
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 21 12" src="https://user-images.githubusercontent.com/63207451/118658467-b0b8d480-b7ec-11eb-9218-95555c2d7e3b.png">
<br/>

### Annotations

```py
x = np.linspace(-5, 5, 100)
y = 1/(1+np.exp(-x))

fig = go.Figure()
fig.add_scatter(x=x,y=y,mode='lines',
                name='sigmoid',
                marker=dict(color='green'))
fig.add_annotation(x=0, y=0.5,
                   text='point en x=0',
                   showarrow=True,
                   arrowhead=1,
                   arrowsize=1,
                   arrowwidth=2,
                   arrowcolor='black',
                   bgcolor="orange",
                   borderwidth=1,
                   yshift=10)
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 22 17" src="https://user-images.githubusercontent.com/63207451/118658641-df36af80-b7ec-11eb-896d-696748e90290.png">
<br/>

### Droite et plage de valeurs

```py
x = np.linspace(-5, 5, 100)
y = -1/(1+np.exp(-x))

fig=go.Figure()
fig.add_scatter(x=x, y=y)
fig.add_hline(y=-0.5, line_dash="dot", # vline pour verticale
              annotation_text="0.5",
              annotation_position="bottom right"
              )
fig.add_hrect(y0=-0.85, y1=-0.15, # vrect pour verticale
              annotation_text="-0.15 à -0.85",
              annotation_position="top right",
              fillcolor="blue", opacity=0.15, line_width=0.4)

plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 23 22" src="https://user-images.githubusercontent.com/63207451/118658767-fd9cab00-b7ec-11eb-8826-36abcfc5b512.png">
<br/>

## Pie chart Go

```py
labels = ['Apple','Samsung','Nokia','Wiko']
values = [4500, 3000, 1053, 500]
fig = go.Figure()
fig.add_pie(labels=labels, # les valeurs sur lesquelles on compte
            values=values, # ce qui sert à faire les pourcentages
            pull=[0, 0, 0, 0.2], # represente une fraction de pi, ici on décale le 4e label
            textposition='inside',
            )
fig.update_traces(hoverinfo='label+percent', # ce qu'on voit avec la souris
                  textinfo='value', # Ce qu'on lit dans le pie
                  textfont_size=20, # taille du texte du pie
                  marker=dict(colors=['gold', 'mediumturquoise', 'darkorange', 'lightgreen']) # couleur des secteurs
                  )
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 24 13" src="https://user-images.githubusercontent.com/63207451/118658898-1dcc6a00-b7ed-11eb-924e-83338a8cb4a5.png">
<br/>

## Violin chart

```py
df = px.data.iris()
fig=go.Figure()

fig.add_violin(y=df['sepal_length'],
               points='all', # pour tous les afficher
               box_visible=True,
               meanline_visible=True,
               fillcolor='lightblue',
               opacity=0.7,
	       name="longueur sepale")
fig.update_layout(yaxis_zeroline=False)
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-18 à 15 26 23" src="https://user-images.githubusercontent.com/63207451/118659221-697f1380-b7ed-11eb-883f-362f7ce4331a.png">
<br/>

## Histogramme/Bar

A partir du code suivant, on a 2 types de diagrammes bâtons, qu'on choisi avec fig.add_histogram ou fig.add_bar :

```py
df = px.data.iris()
fig=go.Figure()

fig.add_XXXXXX(x=df['sepal_width'])
plot(fig)
```

|add_histogram|add_bar|
|-------------|-------|
|<img width="1386" alt="Capture d’écran 2021-05-18 à 15 29 05" src="https://user-images.githubusercontent.com/63207451/118659679-d4304f00-b7ed-11eb-889d-3df60e99a635.png">|<img width="1386" alt="Capture d’écran 2021-05-18 à 15 30 26" src="https://user-images.githubusercontent.com/63207451/118659902-004bd000-b7ee-11eb-89d1-11ba6f8da369.png">|

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

## réseau de neurones

```py
# liens
edge_x, edge_y = [1.5,3,None,1.5,3,None,1.5,3,None,1.5,3,None,1.5,3,None,1.5,3,None,3,4.5,None,3,4.5,None,3,4.5,None,3,4.5,None,3,4.5,None,3,4.5,None,3,4.5,None,],\
                 [1,0,None,1,2,None,1,4,None,3,0,None,3,2,None,3,4,None,0,1,None,0,1,None,0,3,None,2,1,None,2,3,None,4,1,None,4,3,None,]
# None pour couper la ligne
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=2, color='#000000'),
    hoverinfo='none',
    mode='lines')

# traçage des noeuds
node_x, node_y = [1.5,1.5,3,3,3,4.5,4.5],\
                 [1,3,0,2,4,1,3]
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers',
    hoverinfo='text',
    marker=dict(
        color='green',
        size=20,
        line_width=1.5))

# affichage
fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(
                titlefont_size=16,
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis_range=[-6,10],
                xaxis_range=[-5,11]
             )
             )
plot(fig)
```

<br/>
<p align="center">
<img width="1014" alt="Capture d’écran 2021-04-30 à 13 24 07" src="https://user-images.githubusercontent.com/63207451/116688699-5aa7fc80-a9b7-11eb-8862-1a9307be0412.png">
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

# Modele
model = SVR(C=1.)
model.fit(X, y)

# mesh grid 
x_min, x_max = X.x.min() - margin, X.x.max() + margin
y_min, y_max = X.y.min() - margin, X.y.max() + margin
xrange = np.arange(x_min, x_max, mesh_size)
yrange = np.arange(y_min, y_max, mesh_size)
xx, yy = np.meshgrid(xrange, yrange)

# On run le modele
pred = model.predict(np.c_[xx.ravel(), yy.ravel()])
pred = pred.reshape(xx.shape)

# plot
fig = px.scatter_3d(df, x='x', y='y', z='hauteurs')
fig.update_traces(marker=dict(size=5))
fig.add_traces(go.Surface(x=xrange, y=yrange, z=pred, name='pred_surface'))
plot(fig)
```

<br/>
<p align="center">
<img width="1284" alt="Capture d’écran 2021-05-18 à 10 56 25" src="https://user-images.githubusercontent.com/63207451/118622535-b2bc6c80-b7c7-11eb-8107-0b53b7437e11.png">
<p/>









<br/>
<p align="center"><a href="#Plotly-tutorial---Data-analysis-and-Machine-learning"><img src="http://randojs.com/images/backToTopButton.png" alt="Haut de la page" height="29"/></a></p>

<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>


-----------------------------
