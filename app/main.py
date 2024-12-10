from fastapi import FastAPI
import numpy as np
from multivariate_gaussian import MultivariateGaussian
import time
from pydantic import BaseModel
from typing import List
import json
import random

app = FastAPI()

# Definir el modelo para la matriz
class Matrix(BaseModel):
    matrix: List[List[float]]

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]

@app.post("/multivariate/gaussian")
async def calculo(mean: VectorF, pdf_point: VectorF,
                  covarianza: Matrix, muestras: int):
    start = time.time()

    # Definir la media y la matriz de covarianza
    #mean = [1.0, 2.0]
    #covariance = [[1.0, 0.5], [0.5, 1.0]]

    # Crear la instancia de la distribución
    gaussian = MultivariateGaussian(mean.vector, covarianza.matrix)

    # Evaluar la PDF en un punto
    #x = [1.5, 2.5]
    pdf_value = gaussian.pdf(pdf_point.vector)
    #print(f"Valor de la PDF en {x}: {pdf_value}")
    pdf_v = f"Valor de la PDF en {pdf_point.vector}: {pdf_value}"

    # Generar muestras aleatorias
    samples = gaussian.sample(muestras)
    #print(f"Muestras generadas:\n{samples}")
    samp = f"Muestras generadas:\n{samples}"


    end = time.time()

    var1 = end - start

    j1 = {
        "Time taken in seconds": var1,
        "PDF": pdf_v,
        "Sample": samp
    }
    jj = json.dumps(str(j1))

    return jj