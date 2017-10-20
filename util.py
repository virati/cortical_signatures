"""

util.py

Store of useful code for (mostly) testing purposes.

"""



def print_dict(dictionary, ident = '', braces=1):
    """ Recursively prints nested dictionary keys."""

    for key, value in dictionary.iteritems():
        if isinstance(value, dict):
            print '%s%s%s%s' %(ident,braces*'[',key,braces*']') 
            print_dict(value, ident+'  ', braces+1)
        else:
            print "hi:", ident+'%s' % (key) # Edited to only show nested keys


def plot_3d_scalp(band=None):
	"""

	Plots 3D visualization of electrodes on the scalp.

	"""

    ax = plt.figure().add_subplot(111,projection='3d')
    egipos = mne.channels.read_montage('GSN-HydroCel-257')
    etrodes = egipos.pos
    
    ax.scatter(etrodes[:,0],etrodes[:,1],10*etrodes[:,2],c='c',s=300)
 
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
    # Get rid of the spines                         
    #ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    #ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0)) 
    #ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.set_xticks([])                               
    ax.set_yticks([])                               
    ax.set_zticks([])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    #plt.title(pt + ' ' + condit) # pt is patient, and condit is on vs. off target
    plt.title("Title")
    plt.show()