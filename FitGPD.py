# Librerías importadas
import numpy as np
import pandas as pd
import configparser
import psycopg2
import os
from datetime import datetime

np.seterr(all="ignore")


# Librerías propias
from PSO import LBestPSO
from utils import GPD_Acum,A_2,Test_P_01,Datos_Indep,Convergencia_AD

# Uso las credenciales del archivo db.ini para conectarme a la base de datos
config = configparser.ConfigParser()
config.read('/opt/airflow/modules/db.ini')
#config.read('db.ini')
user = config.get('DB', 'user')
password = config.get('DB', 'password')
db = config.get('DB','db')
host = config.get('DB','host')
port = config.get('DB','port')
conexion = psycopg2.connect(host=host, database=db, user=user, password=password)
conexion.autocommit = True

# Se abre la conexión
cur = conexion.cursor()

# Tomo los datos
cur.execute("""select max(temint) as temint,fecha from ft_meteo where id_est_meteo = 'Ezeiza'  and EXTRACT(Month FROM fecha) = {}
	group by fecha
	order by fecha""".format(datetime.now().month))

# Los guardamos en una lista de listas los resultados
Datos = []
for row in cur:
    Datos.append([row[0],row[1]])
conexion.commit()
conexion.close()


# Pasamos todo a un dataframe
Datos = pd.DataFrame(Datos, columns = ['temint', 'fecha'])

# Lo ordenamos por temperatura
Datos.sort_values(by=['temint'],inplace=True)

# Nos quedamos con el tercio más alto de temperaturas
Datos = Datos.iloc[-1*len(Datos)//3:]

# Volvemos a ordenar por fecha
Datos.sort_index(inplace=True)

# Convierto la columna fecha en string
Datos = Datos.astype({'fecha':'string'})

# Convertimos los datos en independientes
Datos = Datos_Indep(Datos)

# Tomo el mínimo de todos los datos
Minimo = min([i[0]  for i in Datos])

# Hago la lista de los excedentes de ese dato mínimo (con un poco de margen)
Excedentes = [round(float(i[0])-Minimo-0.2,2) for i in Datos]

# Defino el t inicial
Threshold_Inicial = Minimo-0.2

# Busco el Threshold Final y su gamma y sigma

gamma,sigma,Threshold_Intermedio = Convergencia_AD(Excedentes,Threshold_Inicial)

# Calculo el threshold final. 

for i in range(0,200):
        if 1 - (1+(i/10)*gamma/sigma)**(-1/gamma) >= 1-10**(-1):
            print(i)
            print("========== Los datos mayores a este número son datos extremos ==============")
            treshfinal = (i/10)+Threshold_Intermedio
            print((i/10)+Threshold_Intermedio)
            break

# Guardo todo en la base de datos

conexion = psycopg2.connect(host=host, database=db, user=user, password=password)
conexion.autocommit = True
# Se abre la conexión
cur = conexion.cursor()
# Tomo los datos
cur.execute("""update Indices_evt
set Mes = {},
Anio = {},
Gamma = {},
Sigma = {},
treshold = {}
where variable = 'Temperatura Ezeiza'""".format(datetime.now().month,datetime.now().year,gamma,sigma,treshfinal))
conexion.commit()
conexion.close()