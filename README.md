# Algoritmo Scalable Kmeans

## Estudiantes:

* Rafael Mejía Zuluaga
* Sergio Valencia López

En este aplicativo web, se implementa el algoritmo de [Scalable Kmeans](https://arxiv.org/pdf/1203.6402.pdf), y se comparan sus resultados con los obtenidos al utilizar datos sintéticos generados utilizando una mixtura de gaussianas, y utilizando el dataset Iris.

## Manual de uso:

### 1. Clonar el repositorio:

```bash
$ git clone https://github.com/sevalencial/ProyectoAlgoritmos

```

### 2. Moverse al directorio del proyecto:

```bash
$ cd ProyectoAlgoritmos

```

### 3. Crear un entorno virtual y habilitarlo:

```bash
$ python -m venv venv
$ source venv/bin/activate

```

### 4. Instalar las dependencias:

```bash
$ pip install -r requirements.txt

```

### 5. Correr la aplicación:

```bash
$ python App/main.py

```

### 6. Abrir un navegador y entrar a la siguiente dirección: http://127.0.0.1:5000

### 7. Uso dentro del aplicativo:

* En la pestaña de parámetros, se seleccionan los datos de entrada.
* Una vez se seleccionan los datos y se presiona el botón Calcular, se abre la pestaña de resultados para visualizar los resultados obtenidos al utilizar los parámetros seleccionados.