import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

default_xkcd = 0.5
if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
    plt.xkcd(default_xkcd)
    plt.rcParams['font.family'] = 'humor sans'
plt.rcParams['image.cmap'] = 'gray'

class xkcdoff:
    def __enter__(self):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(0)
            plt.rcParams['font.family'] = 'humor sans'
        
    def __exit__(self, exc_type, exc_value, traceback):
        if "xkcd" in ([f.name for f in fm.fontManager.ttflist]):    
            plt.xkcd(default_xkcd)
            plt.rcParams['font.family'] = 'humor sans'

def linear(x, y):
    plt.xlabel('Number of hours of study')
    plt.ylabel('Test score')
    plt.title(f'{x.numel()} generated samples of students')

def logistic(x, y):
    # if values are only 1 and 0
    if y.unique().numel() == 2:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Test passed (1) or failed (0)')
        plt.title(f'{x.numel()} generated samples of students')
    else:
        plt.xlabel('Number of hours of study')
        plt.ylabel('Probability of passing the test')
        plt.title(f'{x.numel()} generated samples of students')