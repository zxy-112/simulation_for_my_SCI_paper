import sys
from typing import Callable
import numpy as np

def snr2value(snr: float, noisePow: float):
    """
    return the amplitude of signal whose snr is snr. assuming the 
    signal is complex signal.
    snr -- the snr of signal.
    noisePow -- the power of the noise.
    """
    ratio = np.power(10, snr / 10)
    signalPow = noisePow * ratio
    return np.sqrt(signalPow)

def hermitian(array: np.ndarray):
    """
    return the hermitian of the array.
    """
    return np.conjugate(array.T)

def calCov(array: np.ndarray):
    """
    return the covariance matrix of the array, that is, 
    array @ hermitian(array) / len(array[0]) .
    """
    return (array @ hermitian(array)) / len(array[0])

def mvdr(output: np.ndarray, expectTheta: float, aTheta: Callable, diagLoad:float=0):
    alpha0 = aTheta(expectTheta)
    covMat = calCov(output)
    covMat = covMat + np.eye(len(covMat)) * diagLoad
    invCovMat = np.linalg.pinv(covMat, hermitian=True)
    normalizeFactor = np.abs((hermitian(alpha0) @ invCovMat @ alpha0).item())
    return invCovMat @ alpha0 / normalizeFactor

def getMatrixT(size: int, width: float, theta: float = None):
    """
    the helper func for mvdrVirtualInter(). this matrix makes the 
    interference more widen.
    width: the width of the null, in degree.
    """
    if theta is not None:
        factor = np.pi ** 2 / 180 * width * np.cos(theta)
    else:
        factor = np.pi ** 2 / 180 * width
    res = np.zeros((size, size), dtype=np.complex128)
    for k in range(size):
        for kk in range(size):
            if kk == k:
                res[k][kk] = 1
            else:
                sub = k - kk
                res[k][kk] = np.sin(sub * factor) / (sub * factor)
    return res

def mvdrVirtualInter(output: np.ndarray, expectTheta: float, interThetas: tuple, virtualPows: tuple, aTheta: Callable, widenTheta: float = None, diagLoad:float=0):
    """
    the interThetas should be like (30., -40.) and 30, -40 are the theta of inter
    """
    covMat = calCov(output)
    for virtualPow, theta in zip(virtualPows, interThetas):
        if widenTheta is None:
            covMat += virtualPow * aTheta(theta) @ hermitian(aTheta(theta))
        else:
            covMat += virtualPow * aTheta(theta) @ hermitian(aTheta(theta)) * getMatrixT(len(covMat), widenTheta, theta)
    covMat += np.eye(len(covMat)) * diagLoad

    invCovMat = np.linalg.pinv(covMat)
    alpha0 = aTheta(expectTheta)
    normalizeFactor = np.abs((hermitian(alpha0) @ invCovMat @ alpha0).item())
    return invCovMat @ alpha0 / normalizeFactor

def spatialSmooth(output: np.ndarray, expectTheta: float, aTheta: Callable, subarraySize: int, diagLoad:float=0):
    """
    the spatial smooth (SS) algorithm.
    output -- the signals on the array.
    expectTheta -- the theta of expect signal.
    aTheta -- the a(theta) function
    subarraySize -- elements number of the subarray.
    """
    alpha0 = aTheta(expectTheta)
    alpha0 = alpha0[:subarraySize, :]

    cov = calCov(output)
    subCov = []
    for k in range(len(output) - subarraySize + 1):
        subCov.append(cov[k:k+subarraySize, k:k+subarraySize])
    smoothedCov = sum(subCov) / len(subCov)

    smoothedCov = smoothedCov + np.eye(subarraySize) * diagLoad
    invSmoothedCov = np.linalg.pinv(smoothedCov, hermitian=True)
    normalizeFactor = (hermitian(alpha0) @ invSmoothedCov @ alpha0).item()

    return invSmoothedCov @ alpha0 / normalizeFactor

def spatialMusic(output: np.ndarray, aTheta: Callable, subarraySize: int, signalNum: int):
    """
    """
    thetas = np.arange(-60, 60)
    
    cov = calCov(output)
    subCov = []
    for k in range(len(output) - subarraySize + 1):
        subCov.append(cov[k:k+subarraySize, k:k+subarraySize])
    smoothedCov = sum(subCov) / len(subCov)

    u, _, _ = np.linalg.svd(smoothedCov)
    MatrixU = u[:, signalNum:]
    res = []
    for theta in thetas:
        guideVec = aTheta(theta)[:subarraySize, :]
        res.append(1/ np.abs(hermitian(guideVec) @ MatrixU @ hermitian(MatrixU) @ guideVec))
    
    res = np.squeeze(np.array(res))
    res = res / np.max(res)
    res = 20 * np.log10(res)
    return thetas, np.squeeze(np.array(res))

def getMatrixFk(arraySize: int, subarraySize: int, k: int):
    """
    the helper func for spatial smooth.
    """
    matrix1 = np.zeros((subarraySize, k))
    matrix2 = np.eye(subarraySize)
    matrix3 = np.zeros((subarraySize, arraySize-k-subarraySize))
    return np.concatenate((matrix1, matrix2, matrix3), axis=1)

def ssMat(output: np.ndarray, subarraySize: int):
    """
    the helper func for calculate the covariance matrix in the spatial smooth method.
    """

    subarrayNum = len(output) - subarraySize + 1
    arraySize = len(output)

    cov = calCov(output)
    smoothedCov = np.zeros((subarraySize, subarraySize), dtype=np.complex128)
    for k in range(subarrayNum):
        matrixFk = getMatrixFk(arraySize, subarraySize, k)
        smoothedCov += matrixFk @ cov @ matrixFk.T
    smoothedCov = smoothedCov / subarrayNum
    return smoothedCov

def fss(output: np.ndarray, expectTheta: float, aTheta: Callable, subarraySize: int, diagLoad:float=0):
    """
    the forward spatial smooth method. 
    """

    smoothedCov = ssMat(output, subarraySize)
    invSmoothedCov = np.linalg.pinv(smoothedCov, hermitian=True)

    alpha0 = aTheta(expectTheta)
    alpha0 = alpha0[:subarraySize, :]

    normalizeFactor = (hermitian(alpha0) @ invSmoothedCov @ alpha0).item()
    return invSmoothedCov @ alpha0 / normalizeFactor

def fbss(output: np.ndarray, expectTheta: float, aTheta: Callable, subarraySize: int, diagLoad:float=0):
    """
    the forward backward spatial smooth method.
    """
    fMat = ssMat(output, subarraySize)
    bMat = ssMat(np.conjugate(np.flip(output, axis=0)), subarraySize)
    smoothedCov = (fMat + bMat) / 2
    invSmoothedCov = np.linalg.pinv(smoothedCov)

    alpha0 = aTheta(expectTheta)
    alpha0 = alpha0[:subarraySize, :]

    normalizeFactor = (hermitian(alpha0) @ invSmoothedCov @ alpha0).item()
    return invSmoothedCov @ alpha0 / normalizeFactor

def toeplitzLike(vec: np.ndarray, arraysize: int):
    """
    the helper func for the allFss.
    """
    colNum = arraysize - vec.size + 1
    res = np.zeros((arraysize, colNum), dtype=np.complex128)
    for k in range(colNum):
        res[k:k+vec.size, k:k+1] = vec
    
    return res

def mss(output: np.ndarray, expectTheta: float, aTheta: Callable, subarraySizes: tuple, diagLoad: float = 0):
    """
    mutiLevel spatial smooth
    """
    arraySize = len(output)
    mats = []
    for subarraySize in subarraySizes:
        weight = fss(output, expectTheta, lambda theta: aTheta(theta)[:arraySize], subarraySize)
        mats.append(toeplitzLike(weight, arraySize))
        arraySize = arraySize -subarraySize + 1
        output = hermitian(mats[-1]) @ output
    
    res = mats[0]
    for mat in mats[1:-1]:
        res = res @ mat
    
    res = res @ aTheta(expectTheta)[:res.shape[1]]
    
    return res

def wfss(output: np.ndarray, expectTheta: float, aTheta: Callable, subarraySize: int, diagLoad:float = 0):
    """
    weighted forward spatial smoothing.
    """
    pass

def getMartixA(expectTheta: float, coherentTheta: tuple, aTheta: Callable):
    """
    helper function for mcmv method.
    """
    res = []
    res.append(aTheta(expectTheta))
    for theta in coherentTheta:
        res.append(aTheta(theta))
    return np.concatenate(res, axis=1)

def mcmv(output: np.ndarray, expectTheta: float, coherentTheta: tuple, aTheta: Callable, diagLoad: float=0):
    """
    MCMV algorithm
    """
    covMat = calCov(output) + np.eye(len(output)) * diagLoad
    invCovMat = np.linalg.pinv(covMat)
    matrixA = getMartixA(expectTheta, coherentTheta, aTheta)

    vectorF = np.zeros((len(coherentTheta)+1, 1), dtype=np.complex128)
    vectorF[0, 0] = 1

    return invCovMat @ matrixA @ np.linalg.pinv(hermitian(matrixA) @ invCovMat @ matrixA) @ vectorF

def ctmv(output: np.ndarray, expectTheta: float, coherentTheta: tuple, aTheta: Callable, noisePow: float, diagLoad: float=0):
    """
    CTMV algorithm
    """
    covMat =calCov(output) + np.eye(len(output)) * diagLoad
    invCovMat = np.linalg.pinv(covMat)

    matrixA = getMartixA(expectTheta, coherentTheta, aTheta)
    matrixB = np.array(matrixA)
    matrixB [:, 0] = 0
    matrixT = np.eye(len(output), dtype=np.complex128) - \
        (matrixA - matrixB) @ np.linalg.pinv(hermitian(matrixA) @ invCovMat @ matrixA) @\
            hermitian(matrixA) @ invCovMat

    recoveredCov = matrixT @ covMat @ hermitian(matrixT) - noisePow * matrixT @ hermitian(matrixT) +\
        noisePow * np.eye(len(output))
    
    invRecoveredCov = np.linalg.pinv(recoveredCov)
    
    alpha0 = aTheta(expectTheta)
    normalizeFactor = (hermitian(alpha0) @ invRecoveredCov @ alpha0).item()
    return invRecoveredCov @ alpha0 / normalizeFactor

def duvall(output: np.ndarray, expectTheta: float, aTheta: Callable, d: float = 0.5):
    """
    duvall beamformer.
    """
    alpha0 = aTheta(expectTheta)[:-1]

    diff = output[:-1] - output[1:] * np.exp(-1j * np.pi * 2 * d * np.sin(expectTheta))
    covMat = calCov(diff)
    invCovMat = np.linalg.pinv(covMat)

    weight = invCovMat @ alpha0
    return weight / (hermitian(alpha0) @ invCovMat @ alpha0)

def duvallBasedFast(
    output: np.ndarray, 
    expectTheta: float, 
    aTheta: Callable,
    interNum: int,
    ):
    """
    Duvall-Structure-Based Fast Adaptive Beamforming for Coherent Interference Cancellation
    """
    eleNum = len(output)
    delta = output[:-1, :] - output[1:, :]
    correlation = np.sum(np.conjugate(output[-1]) * delta, axis = 1, keepdims=True) / len(output[0])

    res = []
    for k in range(interNum):
        res.append(correlation[k: k+eleNum-interNum])
    matrixRd = np.concatenate(res, axis = 1)

    alpha0 = aTheta(expectTheta)[:eleNum-interNum]

    q, _ = np.linalg.qr(matrixRd)
    projectionMat = np.eye(eleNum-interNum) - q @ hermitian(q)

    normalizeScaler = hermitian(alpha0) @ projectionMat @ alpha0
    weight = projectionMat @ alpha0 / normalizeScaler
    return weight

def getMatRp(correlationVec):
    """
    helper fuction for imporvedAdaptiveNulling to get matrix Rp.
    """
    medium = len(correlationVec + 1) // 2
    res = np.zeros((medium, medium), dtype=correlationVec.dtype)
    for row in range(medium):
        for col in range(medium):
            res[row][col] = correlationVec[medium - 1 + row - col]
    return res

def improvedAdaptiveNulling(output, expectTheta, aTheta):
    """
    Improved adaptive nulling of coherent interference without spatial smoothing
    """
    arrayNum = len(output)
    medium = (arrayNum + 1) // 2
    picked = output[medium]
    correlation = np.conjugate(picked) * output
    correlation = np.sum(correlation, axis=1, keepdims=False) / len(picked)
    matrixRp = getMatRp(correlation)
    invMatrixRp = np.linalg.pinv(matrixRp)

    alpha0 = aTheta(expectTheta)[:medium]
    normalizedFactor = hermitian(alpha0) @ invMatrixRp @ alpha0

    return invMatrixRp @ alpha0 / normalizedFactor

def response(weight: np.ndarray, thetas: np.ndarray, responseFunc: Callable, minValue=-60):
    """
    the fuction to calculate the response at every theta in thetas.
    weight -- the weight vector, which is a nx1 numpy ndarray.
    thetas -- 1 dimension numpy ndarray, contains all the thetas whose response
            need to be calculated.
    responseFunc -- the a(theta) function.
    """
    res = []
    for theta in thetas:
        res.append((hermitian(weight) @ responseFunc(theta)).item())
    res = np.array(res)
    res = np.abs(res)
    res = res / max(np.max(res), sys.float_info.min)
    res = 20 * np.log10(res)
    res[res < minValue] = minValue
    return res

def responsePhase(weight: np.ndarray, thetas: np.ndarray, responseFunc: Callable):
    """
    the fuction to calculate the response at every theta in thetas.
    weight -- the weight vector, which is a nx1 numpy ndarray.
    thetas -- 1 dimension numpy ndarray, contains all the thetas whose response
            need to be calculated.
    responseFunc -- the a(theta) function.
    """
    res = []
    for theta in thetas:
        res.append((hermitian(weight) @ responseFunc(theta)).item())
    res = np.array(res)
    res = np.angle(res)
    return res

def calPow(signal: np.ndarray):
    """
    return the power of the signal.
    """
    return sum(np.conjugate(signal) * signal) / len(signal)

def syn(output: np.ndarray, weight: np.ndarray):
    output = output[:weight.size]
    return np.squeeze(hermitian(weight) @ output)

def realCov(expectTheta: float, snr: float, interThetas: tuple, inrs: tuple, noisePow: float, aTheta: Callable):
    """
    return the covariance matrix given the thetas and snrs.
    this is used to calc the optimal weight vector.
    """
    expectPow = snr2value(snr, noisePow) ** 2
    alpha0 = aTheta(expectTheta)
    res = expectPow * alpha0 @ hermitian(alpha0)
    for theta, inr in zip(interThetas, inrs):
        alphai = aTheta(theta)
        interPow = snr2value(inr, noisePow) ** 2
        res = res + interPow * alphai @ hermitian(alphai)
    res = res + noisePow * np.eye(len(res), dtype=res.dtype)
    return res

def optWeight(expectTheta: float, snr: float, interThetas: tuple, inrs: tuple, noisePow: float, aTheta: Callable):
    """
    return the optimal weight given the thetas and snrs.
    """
    alpha0 = aTheta(expectTheta)
    covMatrix = realCov(expectTheta, snr, interThetas, inrs, noisePow, aTheta)
    invCov = np.linalg.pinv(covMatrix)
    res = invCov @ alpha0 / (hermitian(alpha0) @ invCov @ alpha0)
