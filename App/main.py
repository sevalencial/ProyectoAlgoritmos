from flask import Flask, render_template, send_file
from flask import jsonify
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns
from utils.kmeans_func import KMeans
from utils.scalablekmeanspp_func import ScalableKMeansPlusPlus
from utils.data_gen import get_random_data
from utils.kmeanspp_func import KMeansPlusPlus
import numpy as np


#Guardamos nuestro servidor flask en app
app = Flask(__name__)

@app.route('/')

def home():
    mkd_text = "$\sum_{i=1}^ni$"
    return render_template('index.html', mkd_text=mkd_text)


@app.route('/Implementacion')
def implementacion():

    alg = 'kmeans||'

    # Primero se deben capturar los parametros (dataset, algoritmo y valor de k)
    l = 10
    k = 20
    n = 1000
    d = 15

    # Corremos el experimento con los parametros seleccionados

    data = get_random_data(k, n, d)

    if alg == "random":
        centroids_initial = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
    
    elif alg == "kmeans++":
        centroids_initial = KMeansPlusPlus(data, 20)
    
    elif alg == "kmeans||":
        centroids_initial = ScalableKMeansPlusPlus(data, 20, l)

    output = KMeans(data, k, centroids_initial)

    cmap = plt.get_cmap('gnuplot')
    colors = [cmap(i) for i in np.linspace(0, 1, k)]

    centroids1 =output["Centroids"]
    labels1 = output["Labels"]

    for i,color in enumerate(colors,start =1):
        plt.scatter(data[labels1==i, :][:,0], data[labels1==i, :][:,1], color=color)

    for j in range(k):
        plt.scatter(centroids1[j,0],centroids1[j,1],color = 'w',marker='x')

    #plt.savefig(f"./App/static/images/result.png")

    return render_template('implementacion.html')

#Generamos la ruta de una imagen

@app.route('/image')

def plot():
    fig, ax = plt.subplots(figsize=(3,3))
    x = np.linspace(-10, 10, 1000)
    y = x**2
    #sns.set_style("dark")
    sns.lineplot(x, y, color="r")
    sns.despine()
    figura = FigureCanvas(fig)
    output = io.BytesIO()
    fig.savefig(output, transparent=True)
    output.seek(0)
    return send_file(output, mimetype='img/png')

#Le estamos diciendo que esta ruta unicamente funciona por get.
@app.route('/api/v1/users', methods=['GET'])

def get_user():
    response = {'message': "funciona!!"}
    return jsonify(response)

#Le estamos diciendo que esta ruta unicamente funciona por post (Enviar datos).
@app.route('/api/v2/users', methods=['POST'])

def post_user():
    response = {'state': "enviado!!"}
    return jsonify(response)


#Si nuestro programa es el principal, entonces se ejecuta
if __name__ == "__main__":
    app.run(debug=True) #En caso de uqe hagamos cambios se actualiza
    
    
    
    