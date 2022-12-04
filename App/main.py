from flask import Flask, render_template, send_file, request, url_for
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
from utils.mis_class_rate import MisClassRate
from utils.cluster_cost import ClusterCost
import numpy as np
import matplotlib
import pandas as pd
from sklearn.preprocessing import StandardScaler
matplotlib.use('SVG')


#Guardamos nuestro servidor flask en app
app = Flask(__name__)

@app.route('/')

def home():
    mkd_text = "$\sum_{i=1}^ni$"
    return render_template('index.html', mkd_text=mkd_text)


@app.route('/Resultados',  methods=['GET'])
def results():
    try:
        print('lista: ', values)
        l = int(values[2])
        k = int(values[1])
        d = int(values[4])
        n = int(values[3])
        data_type = int(values[0])
        print(data_type)
        #print('lista: ',l,k,d,n,data)
        # Corremos el experimento con los parametros seleccionados


        if data_type == 2:

            data = pd.read_csv("App/data/Iris.csv", index_col = 0)
            data.loc[data.Species == 'Iris-setosa','Species'] = 0
            data.loc[data.Species == 'Iris-versicolor','Species'] = 1
            data.loc[data.Species == 'Iris-virginica','Species'] = 2

            trueLabels = np.array(data['Species'])
            data = np.array(data.drop(columns = ['Species']))
            data = StandardScaler().fit_transform(data)
            print(data)
        elif data_type == 1:
            data_generator = get_random_data(k, n, d)
            data = data_generator['data']
            trueLabels = data_generator['trueLabels']
        
        # Se calculan los centroides iniciales utilizando las 3 formas:

        centroids_initial_random = data[np.random.choice(range(data.shape[0]), k, replace=False),:]
        centroids_initial_pp = KMeansPlusPlus(data, k)
        centroids_initial_scalable = ScalableKMeansPlusPlus(data, k, l)

        # Se hace el calculo utilizando las diferentes inicializaciones calculadas

        output_random = KMeans(data, k, centroids_initial_random)
        output_pp = KMeans(data, k, centroids_initial_pp)
        output_scalable = KMeans(data, k, centroids_initial_scalable)


        centroids_random =output_random["Centroids"]
        labels_random = output_random["Labels"]
        num_iterations_random = output_random["Iteration before Coverge"]

        centroids_pp =output_pp["Centroids"]
        labels_pp = output_pp["Labels"]
        num_iterations_pp = output_pp["Iteration before Coverge"]

        centroids_scalable =output_scalable["Centroids"]
        labels_scalable = output_scalable["Labels"]
        num_iterations_scalable = output_scalable["Iteration before Coverge"]

        # Se generan los diferentes plots
        cmap = plt.get_cmap('gnuplot')
        colors = [cmap(i) for i in np.linspace(0, 1, k)]
        
        plt.figure(figsize=(4,4))
        for i,color in enumerate(colors,start =1):
            plt.scatter(data[labels_random==i, :][:,2], data[labels_random==i, :][:,3], color=color)

        for j in range(k):
            plt.scatter(centroids_random[j,2],centroids_random[j,3],color = 'w',marker='X')

        plt.savefig(f"./App/static/images/result_random.png", transparent = True)
        plt.clf()

        plt.figure(figsize=(4,4))
        for i,color in enumerate(colors,start =1):
            plt.scatter(data[labels_pp==i, :][:,2], data[labels_pp==i, :][:,3], color=color)

        for j in range(k):
            plt.scatter(centroids_pp[j,2],centroids_pp[j,3],color = 'w',marker='X')
        
        plt.savefig(f"./App/static/images/result_pp.png", transparent = True)
        plt.clf()

        plt.figure(figsize=(4,4))
        for i,color in enumerate(colors,start =1):
            plt.scatter(data[labels_scalable==i, :][:,2], data[labels_scalable==i, :][:,3], color=color)

        for j in range(k):
            plt.scatter(centroids_scalable[j,2],centroids_scalable[j,3],color = 'w',marker='X')

        plt.savefig(f"./App/static/images/result_scalable.png", transparent = True)
        plt.clf

        # Se calcula el misclassification rate y los costos

        random_mis_class_rate = MisClassRate(trueLabels, output_random) # Random 
        pp_mis_class_rate = MisClassRate(trueLabels, output_pp) # KMeans++
        scalable_mis_class_rate = MisClassRate(trueLabels, output_scalable) # Scalable KMeans++

        # Costos:

        random_cost = ClusterCost(data, output_random) # Random 
        pp_cost = ClusterCost(data, output_pp) # KMeans++
        scalable_cost = ClusterCost(data, output_scalable) # Scalable KMeans++

        random_results = {'num_iterations': num_iterations_random, 'mis_class_rate': np.round(random_mis_class_rate,3),
                        'cluster_cost': np.round(random_cost,3)}

        pp_results = {'num_iterations': num_iterations_pp, 'mis_class_rate': np.round(pp_mis_class_rate,3),
                        'cluster_cost': np.round(pp_cost,3)}

        scalable_results = {'num_iterations': num_iterations_scalable, 'mis_class_rate': np.round(scalable_mis_class_rate,3),
                        'cluster_cost': np.round(scalable_cost,3)}

        return render_template('results.html', random_results = random_results,pp_results = pp_results, scalable_results = scalable_results)
    except:
        return render_template('error.html')


@app.route('/Implementacion',  methods=['GET', 'POST'])
def implementacion():
    global values
    values = [x for x in request.form.values()]
    print(values)
    # Primero se deben capturar los parametros (dataset, algoritmo y valor de k)
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
    
    
    
    