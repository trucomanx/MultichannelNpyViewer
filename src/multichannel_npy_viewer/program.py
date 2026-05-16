#!/usr/bin/env python3

import sys
import math
import numpy as np
import matplotlib.pyplot as plt

import multichannel_npy_viewer.about as about
from multichannel_npy_viewer.modules.wabout import show_about
from multichannel_npy_viewer.desktop  import create_desktop_file 
from multichannel_npy_viewer.desktop  import create_desktop_directory
from multichannel_npy_viewer.desktop  import create_desktop_menu
from multichannel_npy_viewer.mimetype import ensure_mime_type
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
            
            MIN, MAX = rgb.min(), rgb.max()
            dtype = rgb.dtype    
            
            rgb = rgb.astype(np.float32)

            if   MIN >= 0.0 and MAX <= 1.0:
                pass
            elif MIN >= 0.0 and MAX > 1.0 and MAX < 255.0:
                rgb = rgb / 255.0
            else:
                denom=MAX-MIN
                if denom!=0:
                    rgb = (rgb-MIN)/denom
            
            plt.imshow(rgb)
            plt.title(f"Channel[0:3]\ndtype:{dtype}\nmin:{MIN:.4f}\nmax:{MAX:.4f}")
            plt.axis('off')

        # Grid quadrado
        n = math.ceil(math.sqrt(num_channels))
        fig, axes = plt.subplots(n, n)

        # Flatten axes (caso n>1)
        axes = axes.flatten() if num_channels > 1 else [axes]

        for i in range(num_channels):
            rgb = channels[i]
            
            MIN, MAX = rgb.min(), rgb.max()
            dtype = rgb.dtype   
            
            rgb = rgb.astype(np.float32)

            denom=rgb.max()-rgb.min()
            if denom!=0:
                rgb = (rgb-rgb.min())/denom
            
            axes[i].imshow(rgb, cmap='gray')
            axes[i].set_title(f"Channel{i}\ndtype:{dtype}\nmin:{MIN:.4f}\nmax:{MAX:.4f}")
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

def help():
        print("\n")
        print(f"Use: {about.__program_name__} /path/to/arquive.npy")
        print(f"Use: {about.__program_name__} --about")
        print("\n")

def main():
    
    ensure_mime_type("npy", "application/x-npy", "NumPy array file")
    
    extras="MimeType=application/x-npy;"
    #create_desktop_directory()    
    #create_desktop_menu()
    create_desktop_file('~/.local/share/applications', extras=extras)

    if len(sys.argv) < 2:
        help()
        sys.exit(1)
    
    for n in range(1,len(sys.argv)):
        if sys.argv[n] == "--applications":
            #create_desktop_directory(overwrite = True)
            #create_desktop_menu(overwrite = True)
            create_desktop_file('~/.local/share/applications', overwrite=True, extras=extras)

        elif sys.argv[n] == "--about":
            show_about()

        elif sys.argv[n].lower().endswith(".npy"):
            show_npy_image(sys.argv[n])
            return
            
        else:
            help()
            return
    

if __name__ == "__main__":
    main()
