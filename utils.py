import numpy as np
from PSO import LBestPSO

# Funcion de densidad acumulada de GPD
def GPD_Acum(x,k,sigma):
    return 1 - (1-k*x/sigma)**(1/k)

# Calculo de estadístico de Anderson-Darling
def A_2(y,k,sigma):
    n = len(y)
    y.sort()
    suma = 0
    for i in range(n):
        j = i+1
        suma = suma + (2*j-1)*(np.log(GPD_Acum(y[i],k,sigma)) + np.log(1-GPD_Acum(y[n-1-i],k,sigma)))
    return -n-suma*(1/n)

# Función que te dice si el test está aprobado con p-value >0.25
# Estos valores se sacan de Choulakian, V., and M.A. Stephens. 2001. 
# Goodness-of-ft tests for the generalized Pareto distribution. 
# Technometrics 43 (4): 478–484.

def Test_P_01(k,A_2):
    if k>=-0.9 and k<-0.7 and A_2<=0.471:
        return "Se aprobó el test!"
    elif k>=-0.7 and k<-0.35 and A_2<=0.499:
        return "Se aprobó el test!"
    elif k>=-0.35 and k<-0.15 and A_2<=0.534:
        return "Se aprobó el test!"
    elif k>=-0.15 and k<-0.05 and A_2<=0.550:
        return "Se aprobó el test!"
    elif k>=-0.05 and k<0.05 and A_2<=0.569:
        return "Se aprobó el test!"
    elif k>=0.05 and k<0.15 and A_2<=0.591:
        return "Se aprobó el test!"
    elif k>=0.15 and k<0.25 and A_2<=0.617:
        return "Se aprobó el test!"
    elif k>=0.25 and k<0.35 and A_2<=0.649:
        return "Se aprobó el test!"
    elif k>=0.35 and k<0.45 and A_2<=0.688:
        return "Se aprobó el test!"
    elif k>=0.45  and A_2<=0.735:
        return "Se aprobó el test!"
    else:
        return "No se aprobó el test!"
    
# Función que recibe un dataframe de pandas con una variable y la fecha en que fue tomada 
# y devuelve una lista con datos cuya separación en dias sea mayor a n_dias.
# En caso de conflicto, se queda con el valor más grande y su fecha.
def Datos_Indep (Datos,n_dias=4):
    Años = []
    for i in Datos.values.tolist():
        Años.append(i[0][:4])
    Años = [int(i) for i in list(set(Años))]
    Años.sort()
    Datos_Independientes = []
    for Año in Años:
        Aux = Datos[Datos["fecha"].str.contains(str(Año))]
        Auxf = []
        Auxp = Datos[Datos["fecha"].str.contains(str(Año))]
        Auxp = Auxp.values.tolist()
        AuxA = []
        for i in range(len(Auxp)):
            if i ==0:
                AuxA.append(Auxp[i])
            else:
                if int(Auxp[i][0][-2:])-int(AuxA[0][0][-2:])<n_dias:
                    temp = max([Auxp[i][1],AuxA[0][1]])
                    AuxA[0][0] = Auxp[i][0]
                    AuxA[0][1] = temp
                else:
                    Auxf.append(AuxA)
                    AuxA = []
                    AuxA.append(Auxp[i])
        if len(AuxA)>0:
            Auxf.append(AuxA)
        for i in range(len(Auxf)):
            Datos_Independientes.append(Auxf[i][0][1])
    return Datos_Independientes

# Función que recibe una lista y comprueba si los datos en ella pasan el test de Anderson-Darling.
# En caso negativo, avanzan un intervalo (inter) y vuelven a comprobar hasta la convergencia.
# Devuelve la cantidad de datos en los que se llegó a la convergencia.

def Convergencia_AD(datos,T_inicial,inter=0.1,verb=False):
    Datos2  = datos
    T_Final = T_inicial
    Datos2 = [round(j,2) for j in Datos2 if round(j,2)> 0]
    if verb:
        print("El threshold inicial es de {}".format(T_inicial))
    while len(Datos2)>15:
        A = LBestPSO(Datos2,n_particulas=100,n_grupo=10)
        Ejemplo = A.fit(iter=1000)
        k = -1*Ejemplo[1]
        sigma = Ejemplo[2]
        A = A_2(Datos2,k,sigma)
        test = Test_P_01(k,A)
        if verb:
            print(A)
        if test == "No se aprobó el test!":
            Datos2 = [round(j-inter,2) for j in Datos2 if round(j-inter,2)> 0]
            T_Final = round(T_Final + inter,2)
            if verb:
                print("============= No se convergió ====================")
                print("")
                print("================ El nuevo Threshold es {} =========".format(T_Final))
            

        else:
            if verb:
                print("Convergencia en {} datos".format(len(Datos2)))
                print("Quedo gamma = {} y Sigma = {}".format(-1*k,sigma))
            
            break
    if verb:
        print("El Threshold final quedó en {}".format(T_Final))
    return -1*k,sigma,T_Final