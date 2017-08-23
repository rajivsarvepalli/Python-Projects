import numpy as np
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
    from stack_plotter import bar_stack_grapher as bsg
    bsg(values,bar_labels,colors,barwidth=barwidth,legend_values=legend_values,x_label=x_label,y_label=y_label,title=title,x_lim=x_lim,y_lim=y_lim,plt_show=plt_show)
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
def quick_pfp1_file_reader(directory):
    '''
    input: folder containging pfp1 files
    output: matrix of all their data
    '''
    from quick_pfp1_file_reader import alldata
    return alldata(directory)
def quick_pfp2_file_reader(directory):
    '''
    input: folder containging pfp2 files
    output: matrix of all their data (1 row per file, 1 column per feature of that file)
    '''
    from quick_pfp2_file_reader import alldata
    return alldata(directory)
def quick_txt_reader(directory):
    '''
    input: folder containging txt files
    output: matrix of all their data
    '''
    from quick_txt_reader import allData
    return allData(directory)
def extract_meta_and_data_file(directory):
    '''
    Takes a folder returns all the contents of the meta and data files as list of data_metadata objects;
    these objects contain 4 variables for each of the main namespaces, and each of these variables may 
    conatin dictionaries, or sometimes lists of dictionaries
    '''
    from extract_data import allData
    return allData(directory)
def txt_fileCreator(location,startName,size,numberOfFiles):
    '''
    creates txt file of numbers 0 through size-1, and writes them to desk\n
    Ex input: create_multiple("path to/folder location","fake14.txt",2000,100)\n
    Note the number in txt file name but before the extension,\n
    that is required
    Thsi example creates 100 files each conating 0 through 2000 all separtaed by tabs
    '''
    from txt_fileCreator import create_multiple
    from time import sleep
    print("Starting in 10 seconds")
    sleep(10)
    create_multiple(location,startName,size,numberOfFiles)
def findfile(name,path,all=False):
    '''
    input: name of file, closest knows path to file,\n
    and if all is true it will find all fiels iwth that name in that path\n
    if all is false it it will return the first match in the path
    output: path to file
    '''
    import findFile
    if all:
        return findFile.find_all(name,path)
    else:
        return findFile.find(name,path)
def confusion_matrix_plotter(y_true,y_pred,classes,normalize=False,title='Confusion matrix',plt_show=True):
    '''
    Plots the conusion matrix of one classifier
    input: the ground truth, and 
    the predicted values, \n
    a list of the classes in string format, \n
    Normalize determines whether confusion_matrix is normalized,\n
    title is title of the plot, cmap is what color is cahnged to show the outputs in graph format,
    plt_show determines whether the plot is displayed at the end or not\n
    output: a confusion matrix plot in color with a label,\n
    and displays the plain confusion matrix in printed out format as well\n
    The color is darkest where the most values, and lightest where there are the least
    '''
    from useful_classifier_graphs import confusion_matrix_plotter as cmpl
    cmpl(y_true,y_pred,classes,normalize=normalize,title=title,plt_show=plt_show)
def validation_curve(classifier,X,y,param_name,param_range,plt_show=True):
    '''
    input: instance of classifier that has predict, and fit methods, preferably sklearn classifiers\n
    the training data set x with labels y\n
    the param names that will be adjusted\n
    and the param ranges that they will adjusted to
    plt_show determines whthere the plot is shown at the end\n
    output: a graph of the validation curve (higher score is better)
    '''
    from useful_classifier_graphs import validation_curve
    validation_curve(classifier,X,y,param_name,param_range,plt_show=plt_show)
def learning_curve(classifier,X,y,cv=None,n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5),plt_show=True):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    from useful_classifier_graphs import learning_curve
    learning_curve(classifier,X,y,cv=cv,n_jobs=n_jobs, train_sizes=train_sizes,plt_show=plt_show)
    '''@inproceedings{sklearn_api,
    author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
            Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
            Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
            and Jaques Grobler and Robert Layton and Jake VanderPlas and
            Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
    title     = {{API} design for machine learning software: experiences from the scikit-learn
            project},
    booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
    year      = {2013},
    pages = {108--122},
    }'''
def simple_line_graph_with_points(x,y,color='black',annotated_values=(False,1),linestyle='--',marker='o',xlim =None,ylim=None,plt_show=True):
    '''
    simple line graph with annotated points
    input: annotated values on or off, and the number determines\n
    how the numbers are spaced\n
    ex: if 1, tehn every value is annotated\n
    if 2 then every other value
    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(x,y,linestyle=linestyle,marker=marker,c = color)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if annotated_values[0]:
        for i in range(0,len(y),annotated_values[1]):
            ax.annotate(str(y[i]),xy=(x[i],y[i]),textcoords='offset points')
    if plt_show:
        plt.show()
def plot_calibration_curve(x_train,y_train,x_test,y_test,classifier_object,classifier_name,figure_index,plt_show=False):
    '''
    must be binary classifcation problem (only 2 classes)
    classifiers must be able to predict probabilities using method called predict_proba\n
    or have a desicion function\n
    outputs a graph, and printed values
    '''
    
    from useful_classifier_graphs import plot_calibration_curve
    plot_calibration_curve(x_train,y_train,x_test,y_test,classifier_object,classifier_name,figure_index,plt_show=plt_show)


if __name__ =="__main__":
    from sklearn import datasets
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    X, y = datasets.make_classification(n_samples=100000, n_features=20,n_informative=2, n_redundant=10,random_state=42)    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.99,random_state=42)
    # param_range = np.logspace(-6, -1, 5)
    # validation_curve(SVC(),X,y,'gamma',param_range)
    plot_calibration_curve(x_train,y_train,x_test,y_test,SVC(),'SVC',1,plt_show=True)
