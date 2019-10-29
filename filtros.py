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

def kernel_pasabajos(M, N, d0 = 15, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
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

def kernel_pasaaltos(M, N, d0 = 15, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    kernel = 1.0 - kernel_pasabajos(M, N, d0, forma, n)
    
    return kernel

def kernel_rechazabandas(M, N, c0 = 20, w = 10, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    V, U = hacer_meshgrid(M, N)
    # Distancias al cuadrado
    D = np.square(U - 0.5 * M) + np.square(V - 0.5 * N)
    
    if forma == 0:
        provisional1 = np.where(D <= c0 - 0.5 * w, 1, 0)
        provisional2 = np.where(D <= c0 + 0.5 * w, 1, 0)
        provisional3 = provisional2 - provisional1
        kernel = np.where(provisional3 == 0, 1, 0)
    elif forma == 1:
        kernel = 1.0 - np.exp(-((D - c0**2) / (w * (D + 1e-6)))**2)
    else:
        kernel = np.where(D != c0**2, 1.0 / (1.0 + ((D * w**2) / (D + 1e-6 - c0**2))**n), 0.0)
    
    return kernel

def kernel_pasabandas(M, N, c0 = 20, w = 10, forma = 0, n = 1):
    """forma = 0 para kernel ideal, 1 para gaussiano y cualquier otro para Butterworth."""
    kernel = 1.0 - kernel_rechazabandas(M, N, c0, w, forma, n)
    
    return kernel

 # Filtros notch
def kernel_ideal_notch(M, N, centro, d0):
    u_k = centro[0]
    v_k = centro[1]
    V, U = hacer_meshgrid(M, N)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = np.where(D_k <= d0**2, 0, 1) # Primer pasaaltos
    H_mk = np.where(D_mk <= d0**2, 0, 1) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel

def kernel_gaussiano_notch(M, N, centro, d0):
    u_k = centro[0]
    v_k = centro[1]
    V, U = hacer_meshgrid(M, N)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = 1 - np.exp(-(0.5 / d0**2) * D_k) # Primer pasaaltos
    H_mk = 1 - np.exp(-(0.5 / d0**2) * D_mk) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel

def kernel_butterworth_notch(M, N, centro, d0, n):
    u_k = centro[0]
    v_k = centro[1]
    V, U = hacer_meshgrid(M, N)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = np.divide(D_k**n, D_k**n + d0**(2*n)) # Primer pasaaltos
    H_mk = np.divide(D_mk**n, D_mk**n + d0**(2*n)) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel

def kernel_notch(M, N, d0, centro = (0, 0), forma = 0, pasa = 0, n = 1.0):
    """Filtro notch. 
    forma = 0 para ideal, 1 para gaussiano y cualquier otro valor para butterworth.
    pasa = 0 para notchreject, 1 para notchpass.
    centro y radio son los del notch. notch simétrico automático.
    Especificar n solo para butterworth"""
    
    if forma == 0:
        kernel_prov = kernel_ideal_notch(M, N, centro, d0)
    elif forma == 1:
        kernel_prov = kernel_gaussiano_notch(M, N, centro, d0)
    else:
        kernel_prov = kernel_butterworth_notch(M, N, centro, d0, n)
        
    kernel = pasa + (-1)**pasa * kernel_prov
    
    return kernel

def kernel_maestro(M, N, d0 = 15, centro = (0,0), pasa = 0, n = 1, c0 = 20, w = 10, tipo = 'pasabajos', forma = 0):
    """Me parece que lo mejor es llamar a los argumentos con todo y nombre.
    Usar solo strings permitidos para el argumento 'tipo' o no me hago responsable :3"""
    if tipo == 'pasabajos':
        kernel = kernel_pasabajos(M, N, d0, forma, n)
    elif tipo == 'pasaaltos':
        kernel = kernel_pasaaltos(M, N, d0, forma, n)
    elif tipo == 'pasabandas':
        kernel = kernel_pasabandas(M, N, c0, w, forma, n)
    elif tipo == 'rechazabandas':
        kernel = kernel_rechazabandas(M, N, c0, w, forma, n)
    else: # Asume notch
        kernel = kernel_notch(M, N, d0, centro, forma, pasa, n)
        
    return kernel

def filtrar(imagen, d0 = 15, centro = (0,0), pasa = 0, n = 1, c0 = 20, w = 10, tipo = 'pasabajos', forma = 0):
    M, N = imagen.shape
    kernel = kernel_maestro(M, N, d0, centro, pasa, n, c0, w, tipo, forma)
    
    return filtros.aplicar_filtro(imagen, kernel)