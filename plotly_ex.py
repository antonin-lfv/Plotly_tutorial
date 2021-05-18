""" Plotly examples """

#### importation --------------------------------------------------

from plotly.offline import plot  # pour travailler en offline!
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

# points sur une carte

df = px.data.carshare()
fig = px.scatter_mapbox(df, lat="centroid_lat", lon="centroid_lon", color="peak_hour", size="car_hours",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                  mapbox_style="carto-positron")
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
fig.show()

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
