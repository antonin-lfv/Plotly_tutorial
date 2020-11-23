# Plotly and Matplotlib.pyplot tutorial
<br/>
<a href="https://www.python.org" class="fancybox" ><img align="right" width="110" height="110" src="https://user-images.githubusercontent.com/63207451/97306728-26fce600-185f-11eb-9784-14151a6b2c43.png"><a/>
	
## Introduction

Ce projet à pour objectif de présenter les modules __Matplotlib.pyplot__ et __Plotly__ qui sont les modules les plus utilisés pour faire de la visualisation de données avec Python. Plotly étant le plus compliqué mais également le plus interactif. Dans ce __README__ toutes les fonctions seront accompagnées du résultat. Le code complet pour ce repository est dans les fichiers sous le nom __code.py__ .

<br/>

## Index
- [Plotly](#Plotly)
	- [Importations](#importations)
	- [Première approche](#première-approche)
		- [Premier exemple](#premier-exemple)
		- [Deuxième exemple](#deuxième-exemple)
	- [Fonctions principales plotly.express](#fonctions-principales-plotlyexpress)
	- [Graphiques multiples - Subplots](#graphiques-multiples---subplots)
	- [Graphiques en 3D](#graphiques-en-3d)
	- [Slide bar](#slide-bar)
- [Matplotlib.pyplot](#Matplotlib.pyplot)

<br/>

# Plotly

<br/>
Installation :

<br/>

```py
pip install plotly
```
<br/>
Documentation [Plotly](https://plotly.com/python/) .
<br/>

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

## Fonctions principales plotly.express

<br/>

### Scatter plot

```py
df = px.data.iris() # pandas dataframe
fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",title='Scatter')
plot(fig)
```

### Courbe de tendance et densité

```py

```

### Error bars

```py

```

### Bar charts

```py

```

### Graphiques de corrélations

```py

```

### Scatter plot avec échelle des tailles des points

```py

```

### Plot avec animation

```py

```

### Line charts

```py

```

### Area charts

```py

```

### Pie charts

```py

```

### Pie charts avec partie en dehors

```py

```

### Donut charts

```py

```

### Sunburst charts

```py

```

### Treemaps

```py

```

### Histograms

```py

```

### Boxplots

```py

```

### Violon plots

```py

```

### Density contours

```py

```

### Heatmap

```py

```

### Point sur une carte

```py

```

### Surface sur une carte

```py

```

### Polar plots

```py

```

### Polar bar charts

```py

```

### Radar charts

```py

```

### Coordonnées en 3D

```py

```

### Ternary charts

```py

```

<p align="center">
<a href="#index"> retour au sommaire </a>
<p/>
















# Matplotlib.pyplot
<br/>
Installation :

<br/>

```py
pip install matplotlib
```

Documentation [Matplotlib.pyplot](https://matplotlib.org/3.3.1/api/_as_gen/matplotlib.pyplot.html) .
<br/>






<p align="center">
<a href="#plotly-and-matplotlibpyplot-tutorial"> haut de la page </a>
<p/>
<p align="center">
  <a href="https://github.com/antonin-lfv" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97302854-e484da80-1859-11eb-9374-5b319ca51197.png" title="GitHub" width="40" height="40"></a>
  <a href="https://www.linkedin.com/in/antonin-lefevre-565b8b141" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303444-b2c04380-185a-11eb-8cfc-864c33a64e4b.png" title="LinkedIn" width="40" height="40"></a>
  <a href="mailto:antoninlefevre45@icloud.com" class="fancybox" ><img src="https://user-images.githubusercontent.com/63207451/97303543-cec3e500-185a-11eb-8adc-c1364e2054a9.png" title="Mail" width="40" height="40"></a>
</p>
