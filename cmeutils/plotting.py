import matplotlib.pyplot as plt

def threedplot(
    x = [],
    y = [],
    z = [],
    xlabel = "xlabel",
    ylabel = "ylabel",
    zlabel = "zlabel",
    plot_name = "plot_name"
    ):

    '''Plot a 3d heat map from 3 lists of numbers. This function is useful
    for plotting a dependent variable as a function of two independent variables.
    In the example below we use f(x,y)= -x^2 - y^2 +6 because it looks cool.

    Example

    -------

    We create two indepent variables and a dependent variable in the z axis and
    plot the result. Here z is the equation of an elliptic paraboloid.

    import random

    x = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        x.append(n)

    y = []
    for i in range(0,1000):
        n = random.uniform(-20,20)
        y.append(n)

    z = []
    for i in range(0,len(x)):
        z.append(-x[i]**2 - y[i]**2 +6)

    threedplot(x,y,z)

    Parameters

    ----------

    x,y,z : list of int

    xlabel, ylabel, zlabel : str

    plot_name : str


    '''
    x = x
    y = y
    z = z
    fig = plt.figure(figsize = (10, 10))
    ax = plt.axes(projection='3d')
    ax.set_xlabel(xlabel,fontdict=dict(weight='bold'),fontsize=12)
    ax.set_ylabel(ylabel,fontdict=dict(weight='bold'),fontsize=12)
    ax.set_zlabel(zlabel,fontdict=dict(weight='bold'),fontsize=12)
    p = ax.scatter(x, y, z, c=z, cmap='rainbow', linewidth=7);
    plt.colorbar(p, pad = .1, aspect = 2.3)
    fig.show()
    fig.savefig(plot_name, bbox_inches = "tight", facecolor = "white")
