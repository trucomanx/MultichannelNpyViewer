#!/usr/bin/env python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

from multichannel_npy_viewer.modules.wabout import show_about
#Multichannel NPY Viewer

def show_npy_image(npy_path):
    imagem = np.load(npy_path)

    if len(imagem.shape) <= 1:
        print(f"Problem with shape {imagem.shape} : {npy_path}")
        sys.exit(1)
        
    elif len(imagem.shape) == 2:
        plt.imshow(imagem, cmap='gray')
        plt.title(f"shape: {imagem.shape}")
        plt.axis('off')
        plt.show()
        
    elif len(imagem.shape) == 3:
        # Detectar formato
        if imagem.shape[0] < imagem.shape[2]:
            # CHW → converter para lista de canais
            channels = [imagem[c, :, :] for c in range(imagem.shape[0])]
        else:
            # HWC
            channels = [imagem[:, :, c] for c in range(imagem.shape[2])]

        num_channels = len(channels)

        # Caso RGB (ou >=3), mostrar também imagem combinada
        if num_channels >= 3:
            if imagem.shape[0] < imagem.shape[2]:
                rgb = imagem[0:3, :, :].transpose(1, 2, 0)
            else:
                rgb = imagem[:, :, 0:3]

            plt.figure()
            plt.imshow(rgb)
            plt.title(f"Channel[0:3]")
            plt.axis('off')

        # Grid quadrado
        n = math.ceil(math.sqrt(num_channels))
        fig, axes = plt.subplots(n, n)

        # Flatten axes (caso n>1)
        axes = axes.flatten() if num_channels > 1 else [axes]

        for i in range(num_channels):
            axes[i].imshow(channels[i], cmap='gray')
            axes[i].set_title(f"Channel{i}")
            axes[i].axis('off')

        # Desligar subplots extras
        for i in range(num_channels, len(axes)):
            axes[i].axis('off')
        
        plt.suptitle(f"shape: {imagem.shape}")
        plt.tight_layout()
        plt.show()
                        
    else:
        print(f"Problem with shape {imagem.shape} : {npy_path}")
        sys.exit(1)


def main():

    if len(sys.argv) < 2:
        print(f"Use: python3 {sys.argv[0]} /path/to/arquivo.npy")
        sys.exit(1)

    npy_path = sys.argv[1]
    
    if not npy_path.lower().endswith(".npy"):
        if npy_path == "--about":
            show_about()
            return
        else:
            print("Error: file must be .npy")
            sys.exit(1)
    
    show_npy_image(npy_path)


if __name__ == "__main__":
    main()
