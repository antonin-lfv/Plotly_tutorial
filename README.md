<p align="center">
<a href="https://www.python.org" class="fancybox" ><img width="70" height="70" src="https://user-images.githubusercontent.com/63207451/97306728-26fce600-185f-11eb-9784-14151a6b2c43.png"><a/>
<a href="https://plotly.com" class="fancybox" ><img width="210" height="70" src="https://user-images.githubusercontent.com/63207451/118670736-29bd2980-b7f7-11eb-8aa4-ad41fa393ed1.png"><a/>
	<p/>
<br/>

<h1 align="center"><b>Plotly tutorial - Data analysis and Machine learning</b></h1>
<br/>
	
<p align="center">
	Ce repo à pour objectif de présenter le module <b>Plotly</b> qui est l'un des modules les plus utilisés pour faire de la visualisation de données avec Python. Plotly étant le plus compliqué mais également le plus interactif. Il est construit sur 3 composantes principales, à savoir <b>plotly.express</b> qui est basé sur l'utilisation des dataframes pandas, et qui est simple et rapide, et <b>plotly.graph_objects</b> qui est beaucoup plus puissant, et beaucoup plus personnalisable, il est basé sur la POO et <b>plotly.figure_factory</b> qui est dédiée à la création de figure spécifiques qui seraient trop compliqué à developper avec les 2 autres composantes. Dans ce <b>README</b> toutes les fonctions seront accompagnées du résultat. Le code complet pour ce repository est dans les fichiers sous le nom <b>plotly_ex.py</b> .
<br/>
	<p/>

<br/>

> Pour comprendre plus en détails comment plotly fonctionne, et pour personnaliser au maximum vos graphiques, je vous invite à consulter mon [**article**](https://github.com/antonin-lfv/Plotly_tutorial/blob/main/README.md) sur plotly. ( bientôt disponible )
<br/>
	
> Pour mettre en ligne un dashboard avec une page plotly, j'ai crée un [**repository**](https://github.com/antonin-lfv/app_stock_prices) à ce sujet.

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
		- [Interpolation](#interpolation)
	- [Financial Chart](#Financial-Chart)
	- [Pie chart](#Pie-chart-Go)
	- [Violin chart](#Violin-chart)
	- [Histogramme/Bar](#Histogrammebar)
	- [Graphiques en 3D](#graphiques-en-3d)
		- [Surface](#surface)
		- [Nuage de points](#nuage-de-points) 
	- [Réseau de neurones](#réseau-de-neurones)
	- [Regression surfacique en 3D](#regression-surfacique-en-3d)
	- [Maps](#Maps)
		- [Ligne entre Miami et Chicago](#Ligne-entre-Miami-et-Chicago)
		- [Air colorée sur une carte, triangle des bermudes](#Air-colorée-sur-une-carte-triangle-des-bermudes)
		- [Scatter sur une map](#Scatter-sur-une-map)
		- [Scatter avec ensemble de points](#Scatter-avec-ensemble-de-points)

<br/>

- [Plotly.figure_factory](#plotlyfigure_factory)
    - [Distplot](#Distplot)
    - [Heatmap avec annotations](#Heatmap-avec-annotations)
    - [Dendrogrames](#Dendrogrames)
    - [Champ vectoriel](#Champ-vectoriel)
    - [Lignes de flux](#Lignes-de-flux)
    - [Création d'un tableau](#Création-dun-tableau)
        - [À la main avec LaTex](#À-la-main-avec-LaTex) 
        - [À partir d'un dataframe pandas](#À-partir-dun-dataframe-pandas)

<br/>

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

### Interpolation

```py
temps_exec = [0.40251994132995605, 0.014573812484741211, 0.23034405708312988, 0.4499189853668213, 0.8043158054351807, 0.21565508842468262, 0.10195517539978027, 0.35376596450805664, 0.5991549491882324, 0.08605694770812988, 1.1117901802062988, 0.9011919498443604, 0.3789708614349365, 0.8676671981811523, 1.3645083904266357, 0.8764557838439941, 0.13943982124328613, 0.05235695838928223, 0.1372683048248291, 0.29303503036499023]
l = [3.4549849033355713, 5.4536731243133545, 1.2118861675262451, 0.7063937187194824, 4.295026779174805, 11.98727297782898, 1.0320260524749756, 5.288934707641602, 9.74186897277832, 1.484644889831543, 6.555363893508911, 0.8726191520690918, 2.6839470863342285, 9.980525970458984, 0.665977954864502, 4.907128095626831, 2.7749810218811035, 5.096926927566528, 11.398299217224121, 3.3110921382904053]

fig=make_subplots(rows=2, cols=1,
                  subplot_titles=["Courbe bleue", "Courbe rouge"],
                  specs=[[{'type': 'xy'}], [{'type': 'xy'}]],
                  shared_xaxes=True,
                  )

## figures principales en haut ##
# Interpolation linéaire
fig.add_scatter(y=l, mode='lines', opacity=0.4, line=dict(color='royalblue', width=4), line_dash='dot', name='interpolation linéaire',row=1,col=1)
# Interpolation par spline
fig.add_scatter(y=l, mode='lines', line=dict(color='royalblue', width=4), line_shape='spline', name='interpolation par spline',
                          hovertemplate = "<br>%{y:.0f}</br>", row=1,col=1)
fig.update_xaxes(title="x", row=1, col=1)
fig.update_yaxes(title="y", row=1, col=1)

## Titre + axes labels ##
fig.update_layout(title="Interpolation")
## ligne moyenne ##
fig.add_shape(type="line", line_color="firebrick", line_width=2, opacity=1, line_dash="dot",
              x0=0, x1=len(l)-1, y0=np.mean(l), y1=np.mean(l), row=1, col=1)
## fleche moyenne ##
fig.add_annotation(text="Moyenne : {}".format(int(np.mean(l)), grouping=True, monetary=True),
                   x=int(len(l)/5)*4, # arrows' head
                   y=np.mean(l)*1.2,  # arrows' head
                   arrowhead=2, showarrow=True, row=1, col=1)
## layout custom ##
fig.update_xaxes(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
        ), row=1,col=1)
fig.update_yaxes(
        showgrid=False,
        zeroline=True,
        showline=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',
    ), row=1, col=1)


## figures principales en bas ##
# Interpolation linéaire
fig.add_scatter(y=temps_exec, mode='lines', opacity=0.4, line=dict(color='firebrick', width=4), line_dash='dot', name="Interpolation linéaire",
                          row=2,col=1)
# Interpolation par spline
fig.add_scatter(y=temps_exec, mode='lines', line=dict(color='firebrick', width=4), line_shape='spline', name="Interpolation par spline",
                          hovertemplate = "<br>%{y:.3f}</br>", row=2,col=1)
## ligne moyenne ##
fig.add_shape(type="line", line_color="royalblue", line_width=2, opacity=1, line_dash="dot",
              x0=0, x1=len(l)-1, y0=np.mean(temps_exec), y1=np.mean(temps_exec), row=2, col=1)
## fleche moyenne ##
fig.add_annotation(text="Moyenne : {} sec".format(round(np.mean(temps_exec),2), grouping=True, monetary=True),
                   x=int(len(temps_exec)/5)*4, # arrows' head
                   y=np.mean(temps_exec)*1.2,  # arrows' head
                   arrowhead=2, showarrow=True, row=2, col=1)
fig.update_xaxes(title="x", row=2, col=1)
fig.update_yaxes(title="y", row=2, col=1)
## layout custom ##
fig.update_xaxes(
    showline=True,
    showgrid=False,
    showticklabels=True,
    linecolor='rgb(204, 204, 204)',
    linewidth=2,
    ticks='outside',
    tickfont=dict(
        family='Arial',
        size=12,
        color='rgb(82, 82, 82)',
    ), row=2, col=1)
fig.update_yaxes(
    showgrid=False,
    zeroline=True,
    showline=True,
    linecolor='rgb(204, 204, 204)',
    linewidth=2,
    ticks='outside',
    tickfont=dict(
        family='Arial',
        size=12,
        color='rgb(82, 82, 82)',
    ), row=2, col=1)

fig.update_layout(plot_bgcolor='white',
                  hoverlabel_align='right')
plot(fig)
```

<br/>
<img width="1424" alt="Capture d’écran 2021-05-19 à 14 58 16" src="https://user-images.githubusercontent.com/63207451/118816667-b2e56680-b8b2-11eb-8937-7d631592cd1f.png">
<br/>
	
## Financial Chart
	
Le dataset et disponible dans les fichiers du repo : "EURUSD_5y.csv"	
```py
vert = '#599673'
rouge = '#e95142'
noir = '#000'
df = pd.read_csv('path/EURUSD_5y.csv')

fig = go.Figure()

fig.add_trace(go.Scatter(
    y = df['Close'],
    x = df['Date'],
    line=dict(color=noir, width=1),
    name="",
    hovertemplate=
    "Date: %{x}<br>" +
    "Close: %{y}<br>"
))

fig.add_hline(y=df['Close'].iloc[0],
              line_dash="dot",
              annotation_text="25 mai 2016",
              annotation_position="bottom right",
              line_width=1.5, line=dict(color='black'))

# montée 1

fig.add_vrect(x0='2017-01-01',x1='2018-02-20',
              fillcolor=vert, opacity=0.2, line_width=0.4,
              annotation_text='01-01-2017 au 02-20-2018',
              annotation_position="top left",
              annotation=dict(font=dict(size=8))
              )

fig.add_traces(go.Indicator(
    mode = "number+delta",
    value = 1.239864,
    number={'prefix': "$", 'font_size' : 40},
    delta = {"reference": 1.052698, "valueformat": ".6f", "position" : "bottom"},
    title = {"text": "Eur/USD"},
    domain = {'y': [0, 0.5], 'x': [0.15, 0.4]}))

# descente 1

fig.add_traces(go.Indicator(
    mode = "number+delta",
    value = 1.077702,
    number={'prefix': "$", 'font_size' : 40},
    delta = {"reference": 1.237317, "valueformat": ".6f", "position" : "bottom"},
    title = {"text": "Eur/USD"},
    domain = {'y': [0.5, 0.7], 'x': [0.55, 0.75]}))

fig.add_vrect(x0='2018-04-18',x1='2020-04-24',
              fillcolor=rouge, opacity=0.2, line_width=0.4,
              annotation_text='18-04-2018 au 24-04-2020',
              annotation_position="top right",
              annotation=dict(font=dict(size=8))
              )

fig.update_layout(
    template='simple_white',
    yaxis_title="Euro/USD close",
    title_text="Euro/USD Close"
)

plot(fig)	
```
	
<br/>
<img width="1405" alt="Capture d’écran 2021-05-25 à 11 31 28" src="https://user-images.githubusercontent.com/63207451/119475385-6637da80-bd4d-11eb-824b-58dc00a7e73a.png">
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

<br>

## Maps

### Ligne entre Miami et Chicago

```py
fig = go.Figure()

fig.add_scattermapbox(
    # on relie Miami (lat = 25.7616798, long = -80.1917902) et Chicago (lat = 41.8119, long = -87.6873)
    mode = "markers+lines",
    lon = [-80.1917902, -87.6873],
    lat = [25.7616798, 41.8119],
    marker = {'size': 10,
              'color': 'firebrick',
              })

fig.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0}, # marge left, top, bottom, right
    mapbox = {
        'center': {'lon': -80, 'lat': 40},
        'style': "stamen-terrain",
        'zoom': 3})

plot(fig)
```

<br/>
<img width="1413" alt="Capture d’écran 2021-05-19 à 10 51 17" src="https://user-images.githubusercontent.com/63207451/118784252-24f88400-b890-11eb-9ade-6912e2479c95.png">
<br/>


### Air colorée sur une carte, triangle des bermudes

```py
fig = go.Figure()

# les 3 points :
# Bermudes : lat = 32.320236, long = -64.7740215
# Miami : lat = 25.7616798, long = -80.1917902
# San Juan : lat = 18.2232855, long = -66.5927315

fig.add_scattermapbox(
    fill = "toself",
    lon = [-64.7740215, -80.1917902, -66.5927315], lat = [32.320236, 25.7616798, 18.2232855],
    marker = { 'size': 2, 'color': "red" })

fig.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'style': "stamen-terrain",
        'center': {'lon': -80, 'lat': 25 },
        'zoom': 3},
    showlegend = False)

plot(fig)
```
<br/>
<img width="1413" alt="Capture d’écran 2021-05-19 à 10 51 55" src="https://user-images.githubusercontent.com/63207451/118784355-3b064480-b890-11eb-89a5-c2a27cdd1370.png">
<br/>

### Scatter sur une map

```py
df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(df, locations="iso_alpha", # on situe le pays avec son raccourci international
                     color="continent", # on colorie par continent
                     hover_name="country", # ce qu'on voit avec la souris
                     size="gdpPercap", # la taille des points dépend du pib du pays
                     projection="natural earth" # type de carte
                     )
plot(fig)
```
<br/>
<img width="1413" alt="Capture d’écran 2021-05-19 à 10 53 23" src="https://user-images.githubusercontent.com/63207451/118784620-756fe180-b890-11eb-8f0f-c2f14f6c1d70.png">
<br/>
	
### Scatter avec ensemble de points

```py
token = 'your token from https://studio.mapbox.com'
fig = go.Figure()
	
fig.add_scattermapbox(
    mode = "markers",
    name="",
    lon = list(df['long'].apply(lambda x : float(x))),
    lat = list(df['lat'].apply(lambda x : float(x))),
    marker = dict(size= 5,
              color= df['richter'],
              showscale = True,
              colorscale="jet"
    ),
    hovertemplate=
    "longitude: %{lon}<br>" +
    "latitude: %{lat}<br>"+
    "intensité: %{marker.color}"  ,
)
fig.update_layout(
    margin ={'l':0,'t':0,'b':0,'r':0},
    mapbox = {
        'accesstoken': token,
        'style': 'light',
        'center': {'lon': -80, 'lat': 25 },},
)
plot(fig)
```
<br/>
<img width="1426" alt="Capture d’écran 2021-05-22 à 23 16 19" src="https://user-images.githubusercontent.com/63207451/119240977-c105e200-bb53-11eb-885f-7aaf7dc59c4e.png">


<br/>

# Plotly.figure_factory

__Plotly.figure_factory__ est la partie de plotly qui intervient quand l'utilisation de Go et Px devient impossible.

## Distplot

```py
x = [np.random.randn(150)]

label = ['Groupe 1']
color = ['#B36CD2']
fig = ff.create_distplot(x, label, colors=color,
                         bin_size=.2, show_rug=False)

fig.update_layout(title_text='Distplot')
plot(fig)
```

<br/>
<img alt="Distplot" src="https://user-images.githubusercontent.com/63207451/141830733-cf252fc1-d152-4ffa-b137-97586a9257fc.png">
<br/>

## Heatmap avec annotations

```py
z = [[1, 1, 3],
     [3, 1, 3],
     [3, 1, 1]]

x = ['Équipe A', 'Équipe B', 'Équipe C']
y = ['Match 3', 'Match 2', 'Match 1']

z_text = [['Perdu', 'Perdu', 'Gagné'],
          ['Gagné', 'Perdu', 'Gagné'],
          ['Gagné', 'Perdu', 'Perdu']]

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='gnbu')
plot(fig)
```

<br/>
<img width="1413" alt="Capture d’écran 2021-05-19 à 12 14 27" src="https://user-images.githubusercontent.com/63207451/118796290-c33e1700-b89b-11eb-8705-76939c7c381b.png">
<br/>

## Dendrogrames

```py
X = np.array([[1],[2], [5], [3]])
fig = ff.create_dendrogram(X)
fig.update_layout(width=1080, height=675)
plot(fig)
```

<br/>
<img width="1133" alt="Capture d’écran 2021-05-19 à 12 15 08" src="https://user-images.githubusercontent.com/63207451/118796388-dbae3180-b89b-11eb-95be-b8b2ca1f4caf.png">
<br/>

## Champ vectoriel

```py
x,y = np.meshgrid(np.arange(0, 2, .2), np.arange(0, 2, .2))
u = -np.cos(y)*x
v = np.sin(x)*y+1

fig = ff.create_quiver(x, y, u, v)
plot(fig)
```

<br/>
<img width="1386" alt="Capture d’écran 2021-05-19 à 12 15 34" src="https://user-images.githubusercontent.com/63207451/118796485-f2ed1f00-b89b-11eb-9fd7-4904de35c834.png">
<br/>

## Lignes de flux

```py
x = np.linspace(-4, 4, 80)
y = np.linspace(-4, 4, 80)
Y, X = np.meshgrid(x, y)
u = -(1 + X )**2 + 2*Y
v = 1 - X + (Y+1)**2

fig = ff.create_streamline(x, y, u, v, arrow_scale=.2)
plot(fig)
```
<br/>
<img width="1386" alt="Capture d’écran 2021-05-19 à 12 16 29" src="https://user-images.githubusercontent.com/63207451/118796606-0dbf9380-b89c-11eb-8642-a9e7d3d86cf4.png">
<br/>

## Création d'un tableau

### À la main avec LaTex

```py
data_matrix = [['Forme factorisée', 'Forme developpée'],
               ['$(a+b)^{2}$', '$a^{2}+2ab+b^{2}$'],
               ['$(a-b)^{2}$', '$a^{2}-2ab+b^{2}$'],
               ['$(a+b)(a-b)$', '$a^{2}-b^{2}$']]

fig =  ff.create_table(data_matrix)
plot(fig, include_mathjax='cdn')
```
<br/>
<img width="1424" alt="Capture d’écran 2021-05-19 à 12 27 17" src="https://user-images.githubusercontent.com/63207451/118798097-8f63f100-b89d-11eb-8b50-e6ec0f4f66c3.png">
<br/>

### À partir d'un dataframe pandas

```py
df = px.data.iris()

fig=ff.create_table(df)
plot(fig)
```
<br/>
<img width="1424" alt="Capture d’écran 2021-05-19 à 12 27 49" src="https://user-images.githubusercontent.com/63207451/118798163-a1459400-b89d-11eb-923c-d30c47b77c69.png">
<br/>






<br/>
<p align="center"><a href="#Plotly-tutorial---Data-analysis-and-Machine-learning"><img src="http://randojs.com/images/backToTopButton.png" alt="Haut de la page" height="29"/></a></p>

<p align="center">
	  <a href="https://antonin-lfv.github.io" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/127334786-f48498e4-7aa1-4fbd-b7b4-cd78b43972b8.png" title="Web Page" width="38" height="38"></a>
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>


-----------------------------
