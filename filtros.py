# Funciones utilizadas para filtrado de imágenes

import numpy as np
import matplotlib.pyplot as plt
import cv2

def imagen_dft(imagen):
    """Esta función es solo para poder visualizar las transformadas. Solo sirve con imágenes BGR,
    aunque sean a escala de grises. Usar cv2.imread(imagen, 0)"""
    dft = cv2.dft(np.float32(imagen), flags = cv2.DFT_COMPLEX_OUTPUT) # Transformada de la imagen
    dft_shift = np.fft.fftshift(dft) # Centramos la transformada
    magnitud = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) # Magnitud del espectro
    # Regresar a imagen int
    cota = 20 * np.log(magnitud)
    img_transf = 255 * cota / np.max(cota)
    img_transf = img_transf.astype(np.uint8)
    
    return img_transf

def hacer_meshgrid(M, N):
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)
    
    return V, U

def aplicar_filtro(imagen, kernel):
    transformada = np.fft.fftshift(np.fft.fft2(imagen))
    aplico_filtro = kernel * transformada
    img_filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(aplico_filtro)))
    
    return img_filtrada

def kernel_pasabajos(imagen, d0 = 15, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    M, N = imagen.shape
    V, U = hacer_meshgrid(M, N)
    # Distancias al cuadrado
    D = np.square(U - 0.5 * M) + np.square(V - 0.5 * N)
    
    if forma == 0:
        kernel = np.where(D <= d0**2, 1, 0)
    elif forma == 1:
        kernel = np.exp(-0.5 * D / d0**2)
    else:
        kernel = 1.0 / (1.0 + (D / d0**2)**n)
    
    return kernel

def kernel_pasaaltos(imagen, d0 = 15, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    kernel = 1.0 - kernel_pasabajos(imagen, d0, forma, n)
    
    return kernel

def kernel_rechazabandas(imagen, c0 = 20, w = 10, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    M, N = imagen.shape
    V, U = hacer_meshgrid(M, N)
    # Distancias al cuadrado
    D = np.square(U - 0.5 * M) + np.square(V - 0.5 * N)
    
    if forma == 0:
        provisional1 = np.where(D >= c0 - 0.5 * w, 1, 0)
        provisional2 = np.where(D >= c0 + 0.5 * w, 1, 0)
        provisional3 = provisional1 + provisional2
        kernel = np.where(provisional3 == 0, 1, 0)
    elif forma == 1:
        kernel = 1.0 - np.exp(-((D - c0**2) / (w * (D + 1e-6)))**2)
    else:
        kernel = np.where(D != c0**2, 1.0 / (1.0 + ((D * w**2) / (D + 1e-6 - c0**2))**n), 0.0)
    
    return kernel