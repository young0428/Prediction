#%%
import numpy as np
import ssqueezepy

def synchroextracting_transform(signal, wavelet='gmw', nv=32):
    # Continuous Wavelet Transform (CWT)
    Wx, scales = ssqueezepy.cwt(signal, wavelet=wavelet, nv=nv)
    plt.imshow(np.abs(Wx), aspect='auto', extent=[t[0], t[-1], scales[-1], scales[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Scale')
    plt.title('Synchroextracting Transform')
    plt.show()

    # Phase Transform
    dWx = np.gradient(Wx, axis=-1)
    omega = np.imag(dWx / (1j * Wx + 1e-10))

    # Synchroextracting Transform
    Tx = np.zeros_like(Wx)
    for k in range(Wx.shape[0]):
        for n in range(Wx.shape[1]):
            idx = int(np.round(omega[k, n]))
            if 0 <= idx < Tx.shape[0]:
                Tx[idx, n] += Wx[k, n]

    return Tx, scales

# 예시 신호
N = 1024
t = np.linspace(0, 1, N)
signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz 사인 파동

# Synchroextracting Transform 실행
Tx, scales = synchroextracting_transform(signal)

# 결과 확인
import matplotlib.pyplot as plt
plt.clf()
plt.imshow(np.abs(Tx), aspect='auto', extent=[t[0], t[-1], scales[-1], scales[0]])
plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Scale')
plt.title('Synchroextracting Transform')
plt.show()

#%%