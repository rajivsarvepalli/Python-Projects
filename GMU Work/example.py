#example to send to kostas
def bar_stack_grapher(values,bar_labels,colors,barwidth=1,legend_values=None,x_label=None,y_label=None,title=None,x_lim=None,y_lim=None,plt_show=True):#modify latetr make eay interface
    '''
    input: values in a array that follow the format that each bar is one row\n
    bar_labels label what the bars and determine the number of bars\n
    colors is the colors to use for each stack len(colors) must = len(values[0])
    barwidth is the width of the bars\n
    legned_values is what each color of the bar represents\n
    x_label,y_label are labels for x, and y axis\n
    title is title of the plot\n
    x_lim,y_lim are limits of x-axis and y-axis\n
    plt_show determines whether the plot is hown at the end
    output: a stacked bar graph plotted in matplotlib 
    '''
    
    import matplotlib.pyplot as plt
    import numpy as np
    values= np.array(values)
    try:
        t,v = np.shape(values)
    except ValueError:
        values = np.array([values])
    if len(colors)!=len(values[0]):
        raise ValueError("Length of colors must equal len of times")
    if len(values)!=len(bar_labels):
        raise ValueError("the number of value's rows must equal length of bar_labels")
    x_values_of_bars =[]
    barw =round(barwidth)
    for i in range((barw)+1,((int)(barw)+1)*(len(bar_labels))+barw+1,int(barw)+1):
        x_values_of_bars+=[i]
    f, ax1 = plt.subplots(1)
    bars=[]
    for z in range(0,len(bar_labels)):
        for i in range(0,len(values[0])):
            bars.append(ax1.bar(x_values_of_bars[z], values[z,i],width=barwidth, color = colors[i],bottom=np.sum(values[z,0:i])+0)[0])
    bars = bars[0:len(values[0])]
    plt.xticks(x_values_of_bars, bar_labels)
    if x_label !=None:
        ax1.set_xlabel(x_label)
    if y_label !=None:
        ax1.set_ylabel(y_label)
    if title != None:
        plt.title(title)
    if x_lim !=None:
        plt.xlim(x_lim)
    else:
        plt.xlim([min(x_values_of_bars)-barwidth, max(x_values_of_bars)+barwidth])
    if y_lim !=None:
        plt.ylim(y_lim)
    if legend_values ==None:
        raise ValueError("legend_values set to none while show_legend is true")
    elif len(bars)!=len(legend_values):
        raise ValueError("length of bars(columns of values) not equl to lentgh of legend_values")
    else:
        plt.legend(tuple(bars),tuple(legend_values))
    if plt_show:
        plt.show()
def memory_usage_psutil():
    '''
    Output: returns current memory usage of python in gigabytes
    '''
    import os
    import psutil
    # return the memory usage in GB
    process = psutil.Process(os.getpid())
    mem = (process.memory_info()[0] / float(2 ** 20))*0.001048576 
    return mem
if __name__ =="__main__":
    pass