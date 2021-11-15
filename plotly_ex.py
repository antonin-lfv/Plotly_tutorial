""" Plotly examples """

#### importation --------------------------------------------------

from plotly.offline import plot  # pour travailler en offline!
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.datasets import load_digits
from umap import UMAP # pip install umap-learn
from sklearn.manifold import TSNE
import networkx as nx


#### Templates

templates_plotly = ['ggplot2', 'seaborn', 'simple_white', 'plotly',
         'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
         'ygridoff', 'gridon', 'none']


#### Plotly.Express -----------------------------------------------

# Scatter plot

"On personnalise les figures de plotly express suivant ce modèle : "

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

"Si on veut superposer des courbes avec les données du même dataset on écrit : "

df = px.data.iris()
fig = px.line(df, y=["sepal_width", "sepal_length", "petal_width"],
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"sepal_width":"blue", "sepal_length":"black","petal_width":"green" }, # couleur de chaque ligne
            )
plot(fig)

"On peut aussi séparer les figures en plusieurs, c'est le seul moyen de faire des subplots avec plotly express"

df = px.data.iris()
fig = px.line(df, y=["sepal_width", "sepal_length", "petal_width"],
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"sepal_width":"blue", "sepal_length":"black","petal_width":"green" }, # couleur de chaque ligne
              facet_col="species" # ou facet_col pour séparer en colonne
            )
plot(fig)

df = px.data.iris()
fig = px.line(df, y=["sepal_width", "sepal_length", "petal_width"],
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"sepal_width":"blue", "sepal_length":"black","petal_width":"green" }, # couleur de chaque ligne
              facet_row="species" # ou facet_col pour séparer en colonne
            )
plot(fig)


"création d'animations"

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


"Personnalisation des axes avec un range slider visible"

df = px.data.carshare()
fig = px.line(df, y="car_hours",
              #text="species_id" on ajoute ici l'id de l'espèce au dessus de chaque point
              color_discrete_map={"car_hours":"black"}, # couleur de chaque ligne
            )
fig.update_xaxes(rangeslider_visible=True)
plot(fig)

"Ajout de rectangle et de ligne"

df = px.data.stocks(indexed=True)
fig = px.line(df, facet_col="company",
              facet_col_wrap=2 # nombre de figure par ligne
              )
fig.add_hline( # ou vline pour verticale
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

"Avec indicateurs marginaux"

df = px.data.iris()
fig = px.scatter(df, x="sepal_length", # données
                 color="species", # couleur par expèce
                 marginal_x='box', # marge en boxplot
                 marginal_y='violin', # marge en violon
                 trendline="ols" # courbe de tendances
                 )
plot(fig)

"Curseur sur les axes"
df = px.data.gapminder().query("continent=='Oceania'")
fig = px.line(df, x="year", y="lifeExp", color="country",
              title="curseurs")
fig.update_traces(mode="markers+lines") # courbe avec ligne et points apparent
fig.update_xaxes(showspikes=True)
fig.update_yaxes(showspikes=True)
plot(fig)

"plot 3D"

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species', size='petal_length', size_max=18,symbol='species', opacity=1)
plot(fig)

# BarChart

"simple barchart, on colorie sur une colonne et on sépare par couleur"

df = px.data.tips()
fig=px.bar(df,
           x="sex",
           y="total_bill",
           color="smoker",
           barmode="group")
fig.update_traces( marker_line_color='rgb(8,48,107)',
                  marker_line_width=1.5, opacity=0.6)
plot(fig)

"histogramme avec marge en violon"

df = px.data.iris()
fig = px.histogram(df, x="sepal_length",
                   nbins=50, # on choisi le nombre de barres
                   marginal='violin' # rug, box
                   )
plot(fig)

"histogramme avec marge en boxplot"

df = px.data.iris()
fig = px.histogram(df, x="sepal_length",
                   nbins=50, # on choisi le nombre de barres
                   marginal='box'
                   )
plot(fig)


# PieChart

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

# Polar bar charts

df = px.data.wind()
fig = px.bar_polar(df, r="frequency", theta="direction", color="strength",
                   template="seaborn",
            color_discrete_sequence= px.colors.sequential.Plasma_r)
plot(fig)


# ML

"regression linéaire"

from sklearn.linear_model import LinearRegression

df = px.data.tips()
X = df.total_bill.values.reshape(-1, 1)

model = LinearRegression()
model.fit(X, df.tip)

x_range = np.linspace(X.min(), X.max(), 100)
y_range = model.predict(x_range.reshape(-1, 1))

fig = px.scatter(df, x='total_bill', y='tip', opacity=0.65)
fig.add_traces(go.Scatter(x=x_range, y=y_range, name='Regression Fit'))
plot(fig)

"UMAP"

digits = load_digits()
umap_2d = UMAP(random_state=0)
umap_2d.fit(digits.data)

projections = umap_2d.transform(digits.data)
fig = px.scatter(
    projections, x=0, y=1,
    color=digits.target.astype(str), labels={'color': 'digit'}
)
plot(fig)

"tsne"

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

# Graphique de corrélations

df = px.data.iris()
fig = px.scatter_matrix(df, dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"], color="species")
plot(fig)



#### Plotly.graph_objects -----------------------------------------

# subplots

"pour un subplots on utilise un autre moyen pour initialiser la figure :"

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


# Scatter
# pour un grand dataset, on utilisera scattergl

"basique scatter"
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

"ajout d'annotations"
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

"ajout de plage de valeurs"
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

"Interpolation"

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

# Financial chart

vert = '#599673'
rouge = '#e95142'
noir = '#000'
df = pd.read_csv('EURUSD_5y.csv')

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


# Pie chart

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

# violin chart

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

# histogramme, bar

df = px.data.iris()
fig=go.Figure()

fig.add_histogram(x=df['sepal_width'])
plot(fig)

fig.add_bar(y=df['sepal_width'])
plot(fig)

# graphiques 3D

"surface"

z_data = df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/volcano.csv")
fig = go.Figure(data=[go.Surface(z=z_data, colorscale='IceFire')]) # Z1 liste de liste
fig.update_layout(title='Mountain')
plot(fig)

"nuage de points"

df = px.data.iris()
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
              color='species', size='petal_length', size_max=18,symbol='species', opacity=0.7)
plot(fig)

"réseau de neurones"

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

"regression surfacique"


z1=[0,0,0,1,2,2,2,3,4,5,5,5,5,5,4,4,5,4,4,4,4,3,3,2,2]
z2=[0,1,1,2,2,3,3,3,4,5,6,6,6,5,5,5,6,5,5,5,4,4,3,3,3]
z3=[1,1,2,2,3,3,3,4,4,5,6,6,6,6,6,6,6,6,6,5,5,4,4,3,3]
z4=[1,2,2,3,3,3,4,4,5,6,6,6,7,7,7,6,6,6,6,5,5,5,4,4,3]
z5=[2,2,3,3,4,4,4,4,5,6,6,7,7,7,7,7,7,7,6,6,5,5,5,4,4]
z6=[2,2,3,4,4,4,5,5,6,6,7,7,7,7,7,7,7,7,6,6,6,6,5,5,4]
z7=[2,3,4,4,4,5,5,6,6,7,7,7,8,8,8,8,7,7,7,7,7,6,5,5,4]
z8=[3,3,4,4,5,5,6,6,7,7,7,8,8,9,8,8,8,7,7,7,7,6,6,5,5]
z9=[3,4,4,4,5,6,6,7,7,8,8,9,9,9,9,8,8,8,7,7,7,6,6,6,5]
z10=[3,4,4,5,5,6,6,7,7,8,8,9,10,10,9,8,8,7,7,7,7,6,6,6,5]
z11=[3,4,4,5,6,6,7,7,8,8,9,10,10,9,9,8,7,7,7,7,6,6,6,6,5]
z12=[3,4,4,5,6,6,7,7,8,8,9,9,9,9,8,7,7,7,7,6,6,6,5,5,4]
z13=[3,4,4,5,6,6,7,7,8,8,8,9,9,9,8,7,7,7,7,6,6,5,5,4,4]
z14=[3,4,4,5,5,6,6,7,7,8,8,8,8,8,8,7,7,7,7,6,6,5,5,4,3]
z15=[3,3,4,4,5,6,6,7,7,7,7,7,8,8,8,7,7,7,7,6,6,5,5,4,4]
z16=[3,3,4,4,5,6,6,6,6,7,7,7,7,7,7,7,7,7,6,6,6,5,5,5,4]
z17=[3,3,4,4,5,6,6,6,6,6,6,7,7,7,7,7,6,6,6,6,5,5,5,4,3]
z18=[2,3,4,4,5,5,5,6,6,6,6,6,7,7,6,6,6,6,5,5,5,5,4,4,3]
z19=[2,3,4,4,4,5,5,5,5,4,5,6,6,6,6,6,5,5,5,4,4,4,4,3,3]
z20=[2,3,4,4,4,4,4,4,3,3,5,6,6,6,6,5,5,4,4,4,4,4,3,3,2]
z21=[2,3,3,3,3,3,3,3,2,4,5,6,6,6,5,4,4,4,3,3,3,3,2,2,2]
z22=[2,2,3,3,3,3,3,2,2,4,4,5,5,5,5,4,4,3,3,2,2,2,2,1,1]
z23=[1,2,2,2,2,2,2,1,2,4,4,4,5,5,4,4,3,3,2,2,1,1,1,1,1]
z24=[1,1,1,1,1,1,1,1,2,4,4,4,4,4,4,3,3,2,1,1,1,1,0,0,0]
z25=[0,0,0,0,0,0,0,0,1,2,3,3,3,3,3,2,2,1,1,1,0,0,0,0,0]

Z1 = [z1,z2,z3,z4,z5,z6,z7,z8,z9,z10,z11,z12,z13,z14,z15,z16,z17,z18,z19,z20,z21,z22,z23,z24,z25]

lis=[]
for j in Z1:
    for i in j:
        lis.append(i)
df_z = pd.DataFrame(lis)
df_z.columns=['hauteurs']

xlis = [i for i in range (25)]*25
df_xlis=pd.DataFrame(xlis)

ylis = []
for i in range (25):
    for j in range (25):
        ylis.append(i)
df_ylis=pd.DataFrame(ylis)
df_2d=pd.concat([df_xlis,df_ylis],axis=1)
df_2d.columns=['x','y']

df_final = pd.concat([df_z,df_2d], axis=1)


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


# Maps

"Ligne entre Miami et Chicago"

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

"Aire colorée sur une carte, triangle des bermudes"

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

"Scatter sur une map"

df = px.data.gapminder().query("year == 2007")
fig = px.scatter_geo(df, locations="iso_alpha", # on situe le pays avec son raccourci international
                     color="continent", # on colorie par continent
                     hover_name="country", # ce qu'on voit avec la souris
                     size="gdpPercap", # la taille des points dépend du pib du pays
                     projection="natural earth" # type de carte
                     )
plot(fig)

"Scatter avec groupe de points"

token = 'your token from https://studio.mapbox.com'
fig = go.Figure()
df = pd.read_table('quake.dat', names=['depth', 'lat','long','richter'], sep=',' ,encoding='utf-8', skiprows=8)
fig.add_scattermapbox(
    mode="markers",
    name="",
    lon=list(df['long'].apply(lambda x: float(x))),
    lat=list(df['lat'].apply(lambda x: float(x))),
    marker=dict(size=5,
                color=df['richter'],
                showscale=True,
                colorscale="jet"
                ),
    hovertemplate=
    "longitude: %{lon}<br>" +
    "latitude: %{lat}<br>" +
    "intensité: %{marker.color}",
)
fig.update_layout(
    margin={'l': 0, 't': 0, 'b': 0, 'r': 0},
    mapbox={
        'accesstoken': token,
        'style': 'light',
        'center': {'lon': -80, 'lat': 25}, },
)
plot(fig)

#### Plotly.figure_factory -----------------------------------------

"Distplot"
x = [np.random.randn(150)]

label = ['Groupe 1']
color = ['#B36CD2']
fig = ff.create_distplot(x, label, colors=color,
                         bin_size=.2, show_rug=False)

fig.update_layout(title_text='Distplot')
plot(fig)


"Heatmap avec annotations"

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

"Dendrogrames"

X = np.array([[1],[2], [5], [3]])
fig = ff.create_dendrogram(X)
fig.update_layout(width=1080, height=675)
plot(fig)


"champ vectoriel"

x,y = np.meshgrid(np.arange(0, 2, .2), np.arange(0, 2, .2))
u = -np.cos(y)*x
v = np.sin(x)*y+1

fig = ff.create_quiver(x, y, u, v)
plot(fig)

"Lignes de flux"

x = np.linspace(-4, 4, 80)
y = np.linspace(-4, 4, 80)
Y, X = np.meshgrid(x, y)
u = -(1 + X )**2 + 2*Y
v = 1 - X + (Y+1)**2

fig = ff.create_streamline(x, y, u, v, arrow_scale=.2)
plot(fig)

"Création d'un tableau"

# avec latex à la main

data_matrix = [['Forme factorisée', 'Forme developpée'],
               ['$(a+b)^{2}$',  '$a^{2}+2ab+b^{2}$'],
               ['$(a-b)^{2}$',  '$a^{2}-2ab+b^{2}$'],
               ['$(a+b)(a-b)$', '$a^{2}-b^{2}$']]

fig =  ff.create_table(data_matrix)
plot(fig, include_mathjax='cdn')

# à partir d'un dataframe pandas

df = px.data.iris()

fig=ff.create_table(df)
plot(fig)
