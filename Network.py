import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.animation as ani
import matplotlib.collections as collections
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import mpl_toolkits.mplot3d.art3d as art3d
from celluloid import Camera
import math
import random

# global variables used for units
milla = 10**(-3)
Mega = 10**(6)
pico = 10**(-12)
nano = 10**(-9)


# Simple class used for storing and accessing the excitatory and inhibitory layers
class cortex:
    def __init__(self, exc_layer, inh_layer):
        self.exc_layer = exc_layer
        self.inh_layer = inh_layer

# Class for the Neuron, can be used to create a neuron
class LIF:
    def __init__(self, all_synapse, spike=False, V = -70*milla, V_rest = -70*milla, V_reset= -80*milla,V_thresh= -54*milla, tau= 20*milla, t=0*milla, delta_t= 1*milla, R= 1, V_bg = 16.1*milla):
        # List of indexes of synapses incident on this neuron (in) and synapses this neurons 'axons' attach to (out)
        self.in_synapse = []
        self.out_synapse = []
        # Tells us co-ords of neurons its connected to (used to visualise connections between neurons)
        self.connectivity = {}
        # Tells us wether or not neuron is spiking
        self.spike = spike
        # Various volatage parameters and constants for the neuron
        self.V_rest = V_rest
        self.V_reset = V_reset
        self.V_thresh = V_thresh
        self.V = V
        self.V_bg = V_bg
        # Time constant of the neuron
        self.tau = tau
        # Resistance of the neuron
        self.R = R
        # Current time
        self.t = t
        # The time step this neuron will update according to
        self.delta_t = delta_t
        # Points to all_synapse dictionary so this neuron can access it
        self.all_synapse = all_synapse

    # Works out the rate of change of the voltage
    def gradient(self, current):
        
        grad = (1/self.tau)* ( (self.V_rest-self.V) + self.V_bg+(self.R*current) )
        return grad

    # Updates the voltage of the membrane of the neuron (also resets spike to False for the new timestep)
    def update_soma(self, current):
        self.spike = False
        grad = self.gradient(current)
        self.V += grad*self.delta_t
        self.t += self.delta_t
        if self.V >= self.V_thresh:
            self.V = self.V_reset
            self.spike = True
        return self.V 
    
    # Gets the post-synaptic current that another neuron activates in this neuron, given the synapse connecting the two
    def get_dendrite_I(self, synapse):
        g = synapse.get_g(self.spike)
        I = g*(synapse.E_s - self.V)
        return I

    # Gets the total input current(I) coming into this neuron from neurons its connected to (sums over dendrite current)
    def get_input_I(self):
        I = 0
        for neuron in self.in_synapse:
            synapse = self.all_synapse.get(neuron[0])[neuron[1]]
            I += self.get_dendrite_I(synapse)
        return I
    
    # All synapses on the end of this neurons axons get notified that this neuron has spiked
    def notify_spike(self):
        if self.spike:
            for neuron in self.out_synapse:
                self.all_synapse.get(neuron[0])[neuron[1]].spike()

# Class for the Synapse, can be used to create a synapse
class synapse:
    def __init__(self, s =1, g_bar_s =0.15, P =0.5, tau_s =10*milla, delta_t =1*milla, E_s= 0*milla):
        # Fraction of synapses conductance gates open
        self.s = s
        # Conductance constant
        self.g_bar_s = g_bar_s
        # Fraction of how many conductance gates open with a spike
        self.P = P
        # Synaptic time constant
        self.tau_s = tau_s
        # The size of the timestep
        self.delta_t = delta_t
        # Reversal potential of synapse gates
        self.E_s = E_s

    # If theres a spike we use this function to update s accordingly
    def spike(self):
        self.s+=self.P
        
    # This is used to retrieve the current conductance of the synapse, g
    def get_g(self, spike):

        self.s +=  self.delta_t *(-self.s)/self.tau_s 

        g = self.g_bar_s *self.s

        return g

# Class used to create and initialise a network
class initialise:
    def __init__(self, exc_res =(100,100), inh_res =(100,100), lower_bound_V = -70*milla, upper_bound_V = -40*milla, lmbda_exc=1, lmbda_intra=1, between_inh_neurons=5, updates=False, g_bar_s_exc=0.15, g_bar_s_inh =0.15):
        # The resolution of the cortex (x,y), for the excitatory and inhibitory layer 
        self.exc_res = exc_res
        self.inh_res = inh_res
        # Strict lower and upper bound on voltage we can initialise neurons with
        self.lower_bound_V = lower_bound_V
        self.upper_bound_V = upper_bound_V
        # Parameter for the exponential dist. (we use to penailise connecting to far away neurons)
        self.lmbda_exc = lmbda_exc
        self.lmbda_intra = lmbda_intra
        # The number of neurons we skip between each assignment of an inhibitory neuron
        self.between_inh_neurons = between_inh_neurons
        # Keeps record of coords filled with neurons in inh and exc layer
        self.inh_neurons = []
        self.exc_neurons = []
        # Allows for continued access of all_synapse while initialising
        self.all_synapse = False
        # If true prints certain updates about current computation to the command line
        self.updates = updates
        # Allows us to change value of g_bar_s from default for both excitatory and inhibitory synapses
        self.g_bar_s_exc = g_bar_s_exc
        self.g_bar_s_inh = g_bar_s_inh


    # Outputs random voltage strictly between our upper and lower bounds
    def ran_voltage(self):
        at_edge = True
        ran = random.uniform(self.lower_bound_V,self.upper_bound_V)
        while(at_edge == True):
            if ran == -70*milla:
                ran = random.uniform(-70*milla,-40*milla)
            elif ran == -40*milla:
                ran = random.uniform(-70*milla,-40*milla)
            else:
                at_edge = False
        return ran

    # Randomly returns a plus one or a minus one
    def ran_pls_minus(self):
        ran = random.choice([-1,1])
        return ran

    # Given two points calculates the euclidian distance
    def distance(self, point1, point2, con_type):
        if con_type == "exc_inh" or con_type == "inh_exc":
            dis = [(a - b)**2 for a, b in zip(point1, point2)]
            dis = math.sqrt(sum(dis))

        elif con_type == "exc_exc":
            x1, y1 = point1
            x2, y2 = point2
            w, h = self.exc_res
            dis = np.sqrt( min(abs(x1 - x2), w - abs(x1 - x2))**2 + min(abs(y1 - y2), h - abs(y1 - y2))**2 )
        else:
            raise Exception("Didn't give function valid connection type")
        return dis

    # Creates an np array containing all the synapses in the cortex
    def get_synapse_dict(self):
        all_synapse = {"exc": [], "inh": []}
        self.all_synapse = all_synapse
        return all_synapse

    # Creates neuron with randomised voltage within bounds
    def neuron(self, all_synapse):
        ran_V =self.ran_voltage()
        neuron = LIF(all_synapse, V=ran_V)
        return neuron

    # Creates np array containing a neuron in each entry (creates cortex)
    def cortex(self, all_synapse):
        cortex = [self.exc_layer(all_synapse),self.inh_layer(all_synapse)]
        return cortex
        
    # Creates np array containing a neuron in each entry (creating excitatory layer of cortex)
    def exc_layer(self, all_synapse):
        layer = np.empty(self.exc_res, dtype=object)
        x_len, y_len = self.exc_res
        for i in range(x_len):
            for j in range(y_len):
                neuron = self.neuron(all_synapse)
                neuron.connectivity.update({"layer":"exc"})
                layer[i,j] = neuron
                self.exc_neurons.append((i,j))
        return layer

    # Creates np array containing a neuron with a space of self.between_inh_neurons between each entry (creating inhibitory layer of cortex)
    def inh_layer(self, all_synapse):
        layer = np.empty(self.inh_res, dtype=object)
        x_len, y_len = self.inh_res
        counter = 0
        for i in range(x_len):
            for j in range(y_len):
                if counter == self.between_inh_neurons :
                    neuron = self.neuron(all_synapse)
                    neuron.connectivity.update({"layer":"inh"})
                    layer[i,j] = neuron
                    self.inh_neurons.append((i,j))
                    counter = 0
                else:
                    counter +=1
        return layer

    # Wires isotropically all connections of type excitratory layer to excitatory layer
    def wire_exc_exc_iso(self, layer):
        for current in self.exc_neurons:
            if self.updates == True:
                print("exc_exc",current)
            i, j = current
            connected_to = []
            for neuron in self.exc_neurons:
                if neuron == current:
                    pass
                else:
                    x, y = neuron
                    dis = self.distance(current, neuron, "exc_exc")
                    if random.expovariate(self.lmbda_exc) > dis:
                        self.all_synapse.get("exc").append(synapse(s=random.random(), E_s=0*milla, g_bar_s=self.g_bar_s_exc))
                        index = ["exc",len(self.all_synapse.get("exc"))-1]
                        layer[i,j].out_synapse.append(index)
                        layer[x,y].in_synapse.append(index)
                        connected_to.append((x,y))
                    else:
                        pass
            layer[i,j].connectivity.update({"exc_exc":connected_to})

    # Wires isotropically all connections of type excitatory layer to inhibitory layer
    def wire_exc_inh_iso(self, exc_layer, inh_layer):
        for current in self.exc_neurons:
            if self.updates == True:
                print("exc_inh",current)
            i, j = current
            connected_to = []
            for neuron in self.inh_neurons:
                x, y = neuron
                dis = self.distance(current, neuron, "exc_inh")
                if random.expovariate(self.lmbda_intra) > dis:
                    self.all_synapse.get("exc").append(synapse(s=random.random(), E_s=0*milla, g_bar_s=self.g_bar_s_exc))
                    index = ["exc",len(self.all_synapse.get("exc"))-1]
                    exc_layer[i,j].out_synapse.append(index)
                    inh_layer[x,y].in_synapse.append(index)
                    connected_to.append(neuron)
                else:
                    pass
            exc_layer[i,j].connectivity.update({"exc_inh":connected_to})

    # Wires isotropically all connections of type inhibitory layer to excitatory layer
    def wire_inh_exc_iso(self, exc_layer, inh_layer):
        for current in self.inh_neurons:
            if self.updates == True:
                print("inh_exc",current)
            i, j = current
            connected_to = []
            for neuron in self.exc_neurons:
                x, y = neuron
                dis = self.distance(current, neuron, "inh_exc")
                if random.expovariate(self.lmbda_intra) > dis:
                    self.all_synapse.get("inh").append(synapse(s=random.random(), E_s=-80*milla, g_bar_s=self.g_bar_s_inh))
                    index = ["inh",len(self.all_synapse.get("inh"))-1]
                    inh_layer[i,j].out_synapse.append(index)
                    exc_layer[x,y].in_synapse.append(index)
                    connected_to.append(neuron)
                else:
                    pass
            inh_layer[i,j].connectivity.update({"inh_exc":connected_to})

    # Wires the cortex isotrpically
    def wire_synapses_isotropic(self, cortex): 
        self.wire_exc_exc_iso(cortex.exc_layer)
        if self.updates==True:
            print("wired exc_exc")
        self.wire_exc_inh_iso(cortex.exc_layer, cortex.inh_layer)
        if self.updates==True:
            print("wired exc_inh")
        self.wire_inh_exc_iso(cortex.exc_layer, cortex.inh_layer)
        if self.updates==True:
            print("wired inh_exc")

# Class used to help facilitate the running of simulations
class run:
    def __init__(self, res, time_steps, no_of_layers =2):
        # The resolution we want for the layers
        self.res = res
        # The number of timesteps for which we will run the simulation
        self.time_steps = time_steps
        # The number of layers we want (included to facilitate easier modification in future)
        self.no_of_layers = no_of_layers
        # Attribute for matrix containing spiking information
        self.spike_matrix = self.init_spike_matrix(res, no_of_layers)
        # Attribute for matrix used to store state information about current while updating network
        self.I_matrix = self.init_I_matrix(res, no_of_layers)
        # Attribute for matrix containing voltage information of both layers over whole simulation
        self.V_with_t_matrix = self.init_V_with_t_matrix(res, no_of_layers, time_steps)

    # Initialises matrix to store spike times of all neurons 
    def init_spike_matrix(self, res, no_of_layers):
        x_len, y_len = res
        spike_matrix = np.empty((x_len,y_len,no_of_layers), dtype=object)
        for k in range(no_of_layers):
            for i in range(x_len):
                for j in range(y_len):
                    spike_matrix[i,j,k] = [] 
        return spike_matrix
    # Initialises matrix to store values of current at given timestep of all neurons 
    def init_I_matrix(self, res, no_of_layers):
        x_len, y_len = res
        I_matrix = np.empty((x_len, y_len, no_of_layers))
        return I_matrix
    # Initialises matrix to store values of voltage at every timestep for all neurons 
    def init_V_with_t_matrix(self, res, no_of_layers, time_steps):
        x_len, y_len = res
        V_with_t_matrix = np.empty((x_len, y_len, no_of_layers, time_steps))
        return V_with_t_matrix
    # Updates the I matrix for timestep
    def update_I(self,cortex):
        x_len, y_len = self.res
        for k in range(self.no_of_layers):
            for i in range(x_len):
                for j in range(y_len):
                    if k == 0:
                        self.I_matrix[i,j,k] = cortex.exc_layer[i,j].get_input_I()
                    else:
                        if cortex.inh_layer[i,j] != None:
                            self.I_matrix[i,j,k] = cortex.inh_layer[i,j].get_input_I()
                        else:
                            pass
    # Updates the V_with_t and spike matrix for given timestep
    def update_V_and_spike(self, cortex, t):
        x_len, y_len = self.res
        for k in range(self.no_of_layers):
            for i in range(x_len):
                for j in range(y_len):
                    if k == 0:
                        V = cortex.exc_layer[i,j].update_soma(self.I_matrix[i,j,k])
                        cortex.exc_layer[i,j].notify_spike()
                        self.V_with_t_matrix[i,j,k,t] = V
                        if cortex.exc_layer[i,j].spike == True:
                            self.spike_matrix[i,j,k].append(t)
                        else:
                            pass

                    else:
                        if cortex.inh_layer[i,j] != None:
                            V = cortex.inh_layer[i,j].update_soma(self.I_matrix[i,j,k])
                            cortex.inh_layer[i,j].notify_spike()
                            self.V_with_t_matrix[i,j,k,t] = V
                            if cortex.inh_layer[i,j].spike == True:
                                self.spike_matrix[i,j,k].append(t)
                            else:
                                pass
                        else:
                            pass
    # Updates all the neurons voltages but without recording in the V_with_t matrix
    def update_V_no_recording(self, cortex):
        x_len, y_len = self.res
        for k in range(self.no_of_layers):
            for i in range(x_len):
                for j in range(y_len):
                    if k == 0:
                        cortex.exc_layer[i,j].update_soma(self.I_matrix[i,j,k])
                        cortex.exc_layer[i,j].notify_spike()
                    else:
                        if cortex.inh_layer[i,j] != None:
                            cortex.inh_layer[i,j].update_soma(self.I_matrix[i,j,k])
                            cortex.inh_layer[i,j].notify_spike()
                        else:
                            pass
    # Runs the cortex for certain amount of time (stab_time) so that it reaches stable state
    def stabilise(self, cortex, stab_time=2000*milla):
        delta_t = cortex.exc_layer[0,0].delta_t
        time_steps = int(stab_time/delta_t)
        for _ in range(time_steps):
            self.update_I(cortex)
            self.update_V_no_recording(cortex)
    # Calculates average firing rate of a given neuron
    def avg_f_rate_neuron(self, neuron):
        spikes = 0
        for _ in range(self.time_steps):
            neuron.update_soma(0)
            if neuron.spike == True:
                spikes +=1
        time = self.time_steps * neuron.delta_t
        f_rate = spikes/time
        return f_rate
    # Calculates the average firing rate of given layer
    def avg_f_rate_layer(self, layer):
        if layer == "exc":
            layer_i = 0
        elif layer == "inh":
            layer_i = 1
        else:
            raise Exception("Not given valid layer name")
        
        total_spikes = 0
        x_len, y_len = self.res
        for i in range(x_len):
            for j in range(y_len):
                if self.spike_matrix[i,j,layer_i] != []:
                    no_of_spikes = len(self.spike_matrix[i,j,layer_i])
                    total_spikes += no_of_spikes
                else:
                    pass
        f_rate = total_spikes / self.time_steps
        return f_rate

    # Runs the simulation
    def run(self, g_bar_s_exc=0.15, g_bar_s_inh=0.15, lmbda_exc=1, lmbda_intra=1, between_inh_neurons=5, updates=False):
        init = initialise(exc_res=self.res, inh_res=self.res, updates=updates, g_bar_s_exc=g_bar_s_exc, g_bar_s_inh=g_bar_s_inh, lmbda_exc=lmbda_exc, lmbda_intra=lmbda_intra, between_inh_neurons=between_inh_neurons)
        all_synapse = init.get_synapse_dict()
        if updates == True:
            print("got synapses")
        layers = init.cortex(all_synapse)
        cortex_ins = cortex(layers[0],layers[1])
        if updates == True:
            print("got cortex")
        init.wire_synapses_isotropic(cortex_ins)
        if updates == True:
            print("done wiring")
        self.stabilise(cortex_ins)
        if updates == True:
            print("done stabilisation")

        for t in range(self.time_steps):
            if updates == True:
                print(t)
            self.update_I(cortex_ins)
            self.update_V_and_spike(cortex_ins,t)
        if updates == True:
            print("done running")

# Class containing more experimental ways of visualising the network,
# that were used extensively in construction to gauge network activity and performance
class exp_visualise:
    def __init__(self, filename, exc_layer_z =1, inh_layer_z=0):
        # Filename we want saving of plots to start with
        self.filename = filename
        # The z-value of where we want the excitatory and inhibitory layers placed
        self.exc_layer_z = exc_layer_z
        self.inh_layer_z = inh_layer_z

    # Plots the voltage of the neurons of neurons in exc layer with time
    def exc_voltage_time_graph(self, layer, cortex_voltage_time_matrix, show =False):
        x_len, y_len, time = cortex_voltage_time_matrix.shape
        x = np.empty(time)
        for i in range(0,time):
            x[i] = i*layer[0,0].delta_t
        for i in range(x_len):
            for j in range(y_len):
                y = np.empty(time)
                for t in range(time):
                    y[t] = cortex_voltage_time_matrix[i,j,t]
                r = random.random()
                b = random.random()
                g = random.random()
                colour = (r, g, b)     
                plt.plot(x,y, color = colour)
        plt.savefig(self.filename+'_voltage_time.png')
        if show == True:
            plt.show()

    # Plots animated view of exc layer of cortex
    def exc_ani_plane(self, layer, cortex_voltage_time_matrix, show =False):
        x_len, y_len, time = cortex_voltage_time_matrix

        fig, ax = plt.subplots() 
        camera = Camera(fig)
        V_min = layer[0,0].V_rest
        V_max = layer[0,0].V_thresh

        for i in range(time):
            plt.text(0.5, 1.01, "time= "+str(i)+" ms", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
            c = ax.imshow(cortex_voltage_time_matrix[:,:,i], vmin=V_min, vmax=V_max, extent=[0,y_len,x_len,0])
            camera.snap()

        fig.colorbar(c,ax=ax)
        animation = camera.animate()
        animation.save(self.filename +'_animated_cortex.gif')
        if show == True:
            plt.show()

    # Calculates list of coords for neurons and list of lines and colours for synapse connections needed for plotting in 2D
    def get_connections_and_neurons_2d(self, layer, s_type):
        x_len, y_len = layer.shape
        connections = []
        neurons_down = []
        neurons_across = []
        colors = []
        my_cmap = plt.get_cmap('jet')

        for i in range(x_len):
            for j in range(y_len):
                neurons_down.append(i)
                neurons_across.append(j)
                ls = layer[i,j].connectivity[s_type]
                c  = random.random()
                for neuron in ls:
                    neuron_0, neuron_1 = neuron
                    point = (neuron_1,neuron_0)
                    connections.append( ((j,i),point) )  
                    colors.append(my_cmap(c))       
        return neurons_across, neurons_down, connections, colors
    
    # Calculates list of coords for neurons and list of lines and colours for synapse connections needed for plotting in 3D
    def get_connections_and_neurons_3d(self, curr_layer, to_layer, s_type):
        curr_layer_type = s_type[0:3]
        if curr_layer_type == "exc":
            if s_type[4:7]=="exc":
                layer_set_curr = self.exc_layer_z
                layer_set_to = self.exc_layer_z
            else:
                layer_set_curr = self.exc_layer_z
                layer_set_to = self.inh_layer_z
        elif curr_layer_type == "inh" :
            layer_set_curr = self.inh_layer_z
            layer_set_to = self.exc_layer_z
        else:
            raise Exception("Not given a correct layer")
        
        x_len, y_len = curr_layer.shape
        connections = []
        neurons_down = []
        neurons_across = []
        colors = []
        my_cmap = plt.get_cmap('jet')
        layer_loc = []
        for i in range(x_len):
            for j in range(y_len):
                if curr_layer[i,j] != None:
                    neurons_down.append(i)
                    neurons_across.append(j)
                    layer_loc.append(0)
                    ls = curr_layer[i,j].connectivity[s_type]
                    c  = random.random()
                    for neuron in ls:
                        neuron_0, neuron_1 = neuron
                        point = (neuron_1, neuron_0, layer_set_to)
                        connections.append( ((j, i, layer_set_curr), point) )  
                        colors.append(my_cmap(c))      
                else:
                    pass 
        return connections, colors

    # Returns list containing list of y coords of neurons along with list of x coords of neurons (according to matrice/image coords of layer)
    def get_neurons_2d(self,layer):
        x_len, y_len = layer.shape
        neurons_down = []
        neurons_across = []
        for i in range(x_len):
            for j in range(y_len):
                neurons_down.append(i)
                neurons_across.append(j)
        return [neurons_across,neurons_down]

   # Returns list containing list of y coords of neurons along with list of x coords of neurons and the z coord of the layer (according to matrice/image coords of layer) 
    def get_neurons_3d(self,layer, exc_or_inh):
            x_len, y_len = layer.shape
            neurons_down = []
            neurons_across = []
            layer_lvl = []
            if exc_or_inh == "exc":
                layer_set = 1
            elif exc_or_inh == "inh":
                layer_set = 0
            else:
                raise Exception("Not given a correct layer")

            for i in range(x_len):
                for j in range(y_len):
                    if layer[i,j] != None:
                        neurons_down.append(i)
                        neurons_across.append(j)
                        layer_lvl.append(layer_set)
                    else:
                        pass
            return [neurons_across,neurons_down,layer_lvl]
    
    # Plots plane in 3d of layer, given the coords of neurons of that layer
    def plane(self, x_arr, y_arr, exc_or_inh, ax, alpha = 0.5):
        X,Y = np.meshgrid(x_arr, y_arr)
        if exc_or_inh == "exc":
            Z = np.ones(X.shape)*self.exc_layer_z
        elif exc_or_inh == "inh":
            Z = np.ones(X.shape)*self.inh_layer_z
        else:
            raise Exception("Not given a correct layer")
        ax.plot_surface(X, Y, Z, alpha=0.4)

    # Plots all exc_exc synapse connections at once in 2d
    def exc_exc_synapse_type_at_once_2d(self, layer, show =False, s_type = "exc_exc"):          
        neurons_across, neurons_down, connections, colors = self.get_connections_and_neurons_2d(layer, s_type)
        _, ax = plt.subplots()
        ax.plot(neurons_across,neurons_down, 'o', color = 'black')
        ax.invert_yaxis()
        ln_coll = collections.LineCollection(connections, colors=colors, alpha=0.75, linestyle='dashed')
        ax.add_collection(ln_coll)

        plt.savefig(self.filename+'_'+s_type+'_synapse_connections.png')
        if show == True:
            plt.show()

    # Animation of exc_exc synapse connections 
    def exc_exc_synapse_type_one_by_one(self, layer, show =False, s_type = "exc_exc"):
        fig, ax = plt.subplots()
        camera = Camera(fig)

        neuron_coords = self.get_neurons_2d(layer)

        connections = []
        my_cmap = plt.get_cmap('jet')
        x_len, y_len = layer.shape
        for i in range(x_len):
            for j in range(y_len):
                ax.plot(neuron_coords[0],neuron_coords[1],'o', color='black')
                ls = layer[i,j].connectivity[s_type]
                c  = random.random()
                connections = []
                for neuron in ls:
                    neuron_0, neuron_1 = neuron
                    point = (neuron_1,neuron_0)
                    connections.append(((j,i),point))
                ln_coll = collections.LineCollection(connections, color=my_cmap(c), alpha=0.75, linestyle='dashed')
                ax.add_collection(ln_coll) 
                point_x, point_y= zip(*ls)
                ax.plot(point_y,point_x,'b+',color = 'red')
                camera.snap()
        animation = camera.animate(interval=1000)
        animation.save(self.filename+'_'+s_type+'synapse_connections_animation.gif')
        if show == True:
            plt.show()

    # Given axes to plot on, plots the given synapse type in 3d
    def synapse_type_at_once_3d(self, exc_layer, inh_layer, s_type, ax, show =False):
        curr_layer_type = s_type[0:3]
        to_layer_type = s_type[4:]
        if curr_layer_type == "exc":
            if to_layer_type == "exc":
                curr_layer = exc_layer
                to_layer = exc_layer
            else:
                curr_layer = exc_layer
                to_layer = inh_layer
        elif curr_layer_type == "inh":
            curr_layer = inh_layer
            to_layer = exc_layer
        else:
            raise Exception("Not given a correct layer")
        
        connections, colors = self.get_connections_and_neurons_3d(curr_layer, to_layer, s_type)
        neurons_across_curr, neurons_down_curr, layer_loc_curr = self.get_neurons_3d(curr_layer,curr_layer_type)
        neurons_across_to, neurons_down_to, layer_loc_to = self.get_neurons_3d(to_layer,to_layer_type)
        ax.plot3D(neurons_across_curr, neurons_down_curr, layer_loc_curr, 'o', color = 'black')
        ax.plot3D(neurons_across_to, neurons_down_to, layer_loc_to, 'o', color = 'black')
        ax.invert_yaxis()
        ln_coll = art3d.Line3DCollection(connections, colors=colors, alpha=0.75, linestyle='dashed')
        ax.add_collection3d(ln_coll)

        x_len, y_len = exc_layer.shape
        xs = np.array(list(range(x_len)))
        ys = np.array(list(range(y_len)))
        self.plane(xs, ys, "exc", ax)
        self.plane(xs, ys, "inh", ax)

    # Plots all exc_exc synapse connections at once in 3d
    def exc_exc_synapse_type_at_once(self, exc_layer, inh_layer, show =False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"exc_exc",ax, show)
        if show == True:
            plt.show()  
    
    # Plots all exc_inh synapse connections at once in 3d
    def exc_inh_synapse_type_at_once(self, exc_layer, inh_layer, show =False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"exc_inh",ax, show)
        if show == True:
            plt.show()
    
    # Plots all inh_exc synapse connections at once in 3d
    def inh_exc_synapse_type_at_once(self, exc_layer, inh_layer, show =False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"inh_exc", ax, show)
        if show == True:
            plt.show()
    
    # Plots all synapse connections at once in 3d
    def all_synapse_type_at_once(self, exc_layer, inh_layer, show =False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"exc_inh",ax)
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"inh_exc", ax)
        self.synapse_type_at_once_3d(exc_layer,inh_layer,"exc_exc",ax)
        plt.savefig(self.filename+"all_synapse_connections.png")
        if show == True:
            plt.show()
        plt.clf()

    def cortex_voltage(self, exc_layer, inh_layer, V_with_t_matrix, show =False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        camera = Camera(fig)
        V_min = exc_layer[0,0].V_reset
        V_max = exc_layer[0,0].V_thresh

        exc_neurons = self.get_neurons_3d(exc_layer, "exc")
        inh_neurons = self.get_neurons_3d(inh_layer, "inh")

        _,_,_,time = V_with_t_matrix.shape
        for t in range(time):
            ax.text2D(0.05, 0.95, "time= "+str(t)+" ms", transform=ax.transAxes)
            def f(x,y, exc_or_inh):
                if exc_or_inh == "exc":
                    V = V_with_t_matrix[x,y,0,t]
                else :
                    V = V_with_t_matrix[x,y,1,t]
                return (V-V_min)/(V_max-V_min)
            z_exc = f(exc_neurons[0],exc_neurons[1], "exc")
            z_inh = f(inh_neurons[0],inh_neurons[1], "inh")

            ax.scatter(exc_neurons[0], exc_neurons[1], exc_neurons[2], 'o', c=z_exc, cmap='jet')
            ax.scatter(inh_neurons[0], inh_neurons[1], inh_neurons[2], 'o', c=z_inh, cmap='jet')
            ax.invert_yaxis()
            camera.snap()

        norm = colors.Normalize(vmin=V_min, vmax=V_max)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'))
        
        animation = camera.animate()
        animation.save(self.filename +'_animated_cortex_3D.gif')
        if show == True:
            plt.show()
        plt.clf()

# Contains plotting methods that are more static in nature, 
# so plotting techniques more aptly suited for plots to be put in a paper
class trad_visualise:
    def __init__(self):
        pass

    # Plots raster plot of just the excitatory neurons
    @staticmethod
    def raster_all_exc(spike_matrix, show = True, save = False):
        to_plot = []
        x_len, y_len, _ = spike_matrix.shape
        for i in range(x_len):
            for j in range(y_len):
                to_plot.append(spike_matrix[i,j,0])
        plt.eventplot(to_plot,colors='blue')
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+'_exc_neurons_raster.png')
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots raster plot of just the inhibitory neurons
    @staticmethod
    def raster_all_inh(spike_matrix, show = True, save = False):
        to_plot = []
        x_len, y_len, _ = spike_matrix.shape
        for i in range(x_len):
            for j in range(y_len):
                if spike_matrix[i,j,1] != []:
                    to_plot.append(spike_matrix[i,j,1])
                else:
                    pass
        plt.eventplot(to_plot,colors='red')
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+'_inh_neurons_raster.png')
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots raster plot of the excitatory and inhibitory neurons
    @staticmethod
    def raster_all_exc_and_inh(spike_matrix, show = True, save = False, show_edges = False, exc_gap = 9, inh_gap = 800):
        to_plot = []
        colors = []
        y_label = [[],[]]
        counter = 0
        x_len, y_len, _ = spike_matrix.shape
        gap = exc_gap
        for i in range(x_len):
            gap += 1
            if gap == exc_gap+1:
                y_label[0].append(counter)
                y_label[1].append("("+str(i)+",0)")
                gap = 0
            for j in range(y_len):
                counter += 1

                if (i==0)or(i==x_len-1)or(j==0)or(j==y_len-1):
                    if show_edges == True:
                        to_plot.append(spike_matrix[i,j,0])
                        colors.append('green')
                    else:
                        to_plot.append(spike_matrix[i,j,0])
                        colors.append('blue')
                else:
                    to_plot.append(spike_matrix[i,j,0])
                    colors.append('blue')

        gap = inh_gap
        for i in range(x_len):
            for j in range(y_len):
                if spike_matrix[i,j,1] != []:
                    counter +=1

                    to_plot.append(spike_matrix[i,j,1])
                    colors.append('red')
                   
                    gap += 1
                    if gap == inh_gap+1:
                        y_label[0].append(counter)
                        y_label[1].append("("+str(i)+","+str(j)+")")
                        gap = 0
                else:
                    pass
        plt.eventplot(to_plot, colors=colors)
        plt.xlabel("Time in ms")
        plt.ylabel("Co-ords of neurons in cortex")
        plt.yticks(y_label[0], y_label[1])
        plt.title("Raster plot of cortex")

        exc_patch = mpatches.Patch(color='blue', label='Exc neurons')
        inh_patch = mpatches.Patch(color='red', label='Inh neurons')
        patch_ls = []
        if show_edges == True:
            edge_patch = mpatches.Patch(color='green', label='Edge neurons')
            patch_ls.append(exc_patch)
            patch_ls.append(inh_patch)
            patch_ls.append(edge_patch)
        else:
            patch_ls.append(exc_patch)
            patch_ls.append(inh_patch)
        plt.legend(handles=patch_ls, bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+'_exc_and_inh_neurons_raster.png', bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Event plot of the firing rates of the excitatory and inhibitory neurons
    @staticmethod
    def f_rate_all_exc_and_inh(cortex, time_steps, spike_matrix, show=True, save = False, show_edges=False, exc_gap = 9, inh_gap = 800):
        f_rate = []
        colors = []
        y_label = [[],[]]
        counter = 0
        x_len, y_len, _ = spike_matrix.shape
        time = time_steps * cortex.exc_layer[0,0].delta_t

        gap = exc_gap
        for i in range(x_len):
            gap += 1
            if gap == exc_gap+1:
                y_label[0].append(counter)
                y_label[1].append("("+str(i)+",0)")
                gap = 0
            for j in range(y_len):
                counter += 1

                if (i==0)or(i==x_len-1)or(j==0)or(j==y_len-1):
                    if show_edges == True:
                        f_rate.append( [len(spike_matrix[i,j,0])/time] )
                        colors.append('green')
                    else:
                        f_rate.append( [len(spike_matrix[i,j,0])/time] )
                        colors.append('blue')
                else:
                    f_rate.append( [len(spike_matrix[i,j,0])/time] )
                    colors.append('blue')
        gap = inh_gap
        for i in range(x_len):
            for j in range(y_len):
                if spike_matrix[i,j,1] != []:
                    counter += 1

                    f_rate.append( [len(spike_matrix[i,j,1])/time] )
                    colors.append('red')

                    gap += 1
                    if gap == inh_gap+1:
                        y_label[0].append(counter)
                        y_label[1].append("("+str(i)+","+str(j)+")")
                        gap = 0
                    

        plt.eventplot(f_rate, colors=colors)
        plt.xlabel("Firing rate (kHz)")
        plt.ylabel("Co-ords of neurons in cortex")
        plt.yticks(y_label[0], y_label[1])
        plt.title("Firing rate of neurons in cortex")

        exc_patch = mpatches.Patch(color='blue', label='Exc neurons')
        inh_patch = mpatches.Patch(color='red', label='Inh neurons')
        patch_ls = []
        if show_edges == True:
            edge_patch = mpatches.Patch(color='green', label='Edge neurons')
            patch_ls.append(exc_patch)
            patch_ls.append(inh_patch)
            patch_ls.append(edge_patch)
        else:
            patch_ls.append(exc_patch)
            patch_ls.append(inh_patch)
        plt.legend(handles=patch_ls, bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)

        if save != False:
            if isinstance(save,str):
                plt.savefig(save+'_exc_and_inh_neurons_firing_rates.png', bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots curve of firing rate againts voltage for single neuron
    @staticmethod
    def f_V_curve(f_val=10, t_steps=10000, no_of_samples=10000, min_V=0*milla, max_V=25*milla, show=True, save = False):
        f_rates = []
        V_bg_ls = np.linspace(min_V,max_V,num=no_of_samples)
        for V in V_bg_ls:
            neuron = LIF([], V_bg=V)
            runner = run((0,0), t_steps)
            f_rate = runner.avg_f_rate_neuron(neuron)
            f_rates.append(f_rate)
        inter = np.interp(f_val, f_rates, V_bg_ls)
        plt.plot([0,inter],[f_val,f_val], "red")
        plt.plot([inter,inter],[f_val,0], "red")
        plt.scatter(inter,f_val, color = "red")
        plt.plot(V_bg_ls, f_rates, "black")
        plt.annotate("Background V = "+str(inter)[0:6],(inter,10))
        plt.xlabel("Background Voltage V")
        plt.ylabel("Firing rate kHz")
        plt.title("Firing Rate Against Background Voltage For A Neuron")
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+'_f_V_curve.png', bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots graph of average firing rate for the excitatory and inhibitory neurons
    # for a list of given values for g_bar_s of the excitatory and inhibitory neurons
    @staticmethod
    def synaptic_strength_graph(strength_ls, cortex_res= (50,50), time_steps=10000, show=True, save = False):
        exc_f_rate = []
        inh_f_rate = []
        for strength in strength_ls:
            strength_exc, strength_inh = strength
            runner = run(cortex_res, time_steps)
            runner.run(g_bar_s_exc=strength_exc, g_bar_s_inh=strength_inh, updates=True)
            exc_f_rate.append(runner.avg_f_rate_layer("exc"))
            inh_f_rate.append(runner.avg_f_rate_layer("inh"))
        xs_exc, xs_inh = zip(*strength_ls)
        plt.scatter(xs_exc, exc_f_rate, color="black")
        plt.scatter(xs_exc, inh_f_rate, color="red")
        plt.xlabel("g_bar_s")
        plt.ylabel("Avg firing rate Hz")
        plt.title("Firing rate against g_bar_s for the cortex")

        exc_patch = mpatches.Patch(color='black', label='Exc neurons')
        inh_patch = mpatches.Patch(color='red', label='Inh neurons')
        patch_ls = [exc_patch,inh_patch]
        plt.legend(handles=patch_ls, bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0)
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+"Firing_rate_v_Synaptic_strength.png", bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots the cortex topoligically and colours neurons according to their firing rate
    @staticmethod
    def f_rate_tpgl(spike_matrix, time_steps, delta_t=1*milla, show=True, save = False):
        x_len, y_len, z = spike_matrix.shape
        points = []
        heats = []
        time = time_steps * delta_t
        min_f_rate = len(spike_matrix[0,0,0])/time
        max_f_rate = len(spike_matrix[0,0,0])/time
        for k in range(z):
            if k == 0:
                level = 1
            elif k == 1:
                level = 0
            else:
                raise Exception("Wrong number of levels")
            for i in range(x_len):
                for j in range(y_len):
                    if spike_matrix[i,j,k] != []:
                        f_rate = len(spike_matrix[i,j,k])/time
                        points.append((i,j,level))
                        heats.append(f_rate)
                        if min_f_rate > f_rate:
                            min_f_rate = f_rate
                        if max_f_rate < f_rate:
                            max_f_rate = f_rate
        for i in range(len(heats)):
            f_rate = heats[i]
            heats[i] = (f_rate-min_f_rate)/(max_f_rate-min_f_rate)

        fig = plt.figure()
        ax = fig.gca(projection='3d')
        plt.title("Topological firing rate (Hz)")
        ax.scatter(*zip(*points), 'o', c=heats, cmap='jet')
        norm = colors.Normalize(vmin=min_f_rate, vmax=max_f_rate)
        fig.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'))
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+"Firing_rate_topologically.png", bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()
    
    # Plots the excitatory neurons topoligically and colours them according to their firing rate
    @staticmethod
    def f_rate_tpgl_exc(spike_matrix, time_steps, delta_t=1*milla, show=True, save = False):
        x_len, y_len, z = spike_matrix.shape
        points = []
        heats = []
        time = time_steps * delta_t
        min_f_rate = len(spike_matrix[0,0,0])/time
        max_f_rate = len(spike_matrix[0,0,0])/time
        for i in range(x_len):
            for j in range(y_len):
                f_rate = len(spike_matrix[i,j,0])/time
                points.append((i,j))
                heats.append(f_rate)
                if min_f_rate > f_rate:
                    min_f_rate = f_rate
                if max_f_rate < f_rate:
                    max_f_rate = f_rate
        if max_f_rate-min_f_rate == 0:
            return False
        for i in range(len(heats)):
            f_rate = heats[i]
            heats[i] = (f_rate-min_f_rate)/(max_f_rate-min_f_rate)
        
        plt.scatter(*zip(*points), c=heats, cmap='jet')
        plt.title("Topological firing rate, Excitatory layer (Hz)")
        norm = colors.Normalize(vmin=min_f_rate, vmax=max_f_rate)
        plt.colorbar(cm.ScalarMappable(norm=norm, cmap='jet'))
        if save != False:
            if isinstance(save,str):
                plt.savefig(save+"Firing_rate_topologically.png", bbox_inches="tight")
            else:
                raise Exception("save isnt a string")
        if show == True:
            plt.show()
        plt.clf()

    # Plots the animated spiking activity of the cortex
    @staticmethod
    def spiking_live(spike_matrix, time_steps, offset=0, save = False):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        camera = Camera(fig)
        xs = []
        ys = []
        zs = []
        x_len, y_len, z = spike_matrix.shape
        for k in range(z):
            if k == 0:
                level = 1
            elif k == 1:
                level = 0
            else:
                raise Exception("Wrong number of levels")
            for i in range(x_len):
                for j in range(y_len):
                    if spike_matrix[i,j,k] != []:
                        xs.append(i)
                        ys.append(j)
                        zs.append(level)

        for t in range(offset,time_steps+offset):
            colors = []
            for k in range(z):
                for i in range(x_len):
                    for j in range(y_len):
                        if spike_matrix[i,j,k] != []:
                            if t in spike_matrix[i,j,k]:
                                colors.append("red")
                            else:
                                colors.append("black")
            ax.text2D(0.05, 0.95, "time= "+str(t)+" ms", transform=ax.transAxes)
            ax.scatter(xs, ys, zs, 'o', c=colors)
            ax.invert_yaxis()
            camera.snap()
        
        animation = camera.animate()

        if save != False:
            if isinstance(save,str):
                animation.save(save +'_spiking_live.gif')
            else:
                raise Exception("save isnt a string")
        plt.clf()

# Class for translation between cortex to retina
class translate:
    def __init__(self,k=10,eps=2,w_0=1):
        self.k = k
        self.eps = eps
        self.w_0 = w_0
        # ^ theese are constants
    
    # For co-ords x,y in cortex returns complex no. representing point in retina
    def point_cortex_to_retina(self,x,y):
        z = (self.w_0 * np.sqrt(self.eps))* np.exp( (x+y*1j)* np.sqrt( (math.pi * self.eps) / (4*self.k) ))
        return z 

    # Creates table containing retina co-ords for cortex height x and width y
    def trans_tab(self,res):
        x, y = res
        table = np.empty((x,y,2))
        for i in range(x):
            for j in range(y):
                z = self.point_cortex_to_retina(i,j)
                table[i,j,0] = z.real
                table[i,j,1] = z.imag
        return table
    
    # Returns list of x and y retinal co-ords whose corresponding cortex co-ords are spiking
    def retinal_points_spikes_t(self, table, time_t, spike_matrix, layer = 0):
        x = np.array([])
        y = np.array([])
        x_len, y_len, _ = spike_matrix.shape
        for i in range(x_len):
            for j in range(y_len):
                if time_t in spike_matrix[i,j,layer]:
                    x = np.append(x, table[i,j,0])
                    y = np.append(y, table[i,j,1])
        return x, y

    # Plots the animated retinal image of given spiking data from a simulation
    def retinal_points_spikes(self, table, spike_matrix, time_steps, layer=0, offset=0, save = False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        camera = Camera(fig)
        for t in range(offset, offset + time_steps):
            xs, ys = self.retinal_points_spikes_t(table, t, spike_matrix, layer=layer)
            ax.scatter(xs, ys, c="black")
            ax.invert_yaxis()
            plt.text(0.5, 1.01, "time= "+str(t)+" ms", horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
            camera.snap()
        plt.xlabel("Retinal Translation of Cortical Activity")
        animation = camera.animate()
        if save != False:
            if isinstance(save,str):
                animation.save(save +'_spiking_live.gif')
            else:
                raise Exception("save isnt a string")
        plt.clf()

    # Plots the summed retinal image of given spiking data from a simulation
    def retinal_points_spikes_avg(self, table, spike_matrix, time_steps, layer=0, offset=0, save = False):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for t in range(offset, offset + time_steps):
            xs, ys = self.retinal_points_spikes_t(table, t, spike_matrix, layer=layer)
            ax.scatter(xs, ys, c="black")
            ax.invert_yaxis()
        plt.xlabel("Retinal Translation of Cortical Activity (Over "+str(time_steps)+" timesteps)")
        if save != False:
            if isinstance(save,str):
                plt.savefig(save +'_spiking_avg.png')
            else:
                raise Exception("save isnt a string")
        plt.clf()

# Class used to help generate random parameters (used during large parameter searches)
class ran_param:
    def __init__(self, res):
        self.lmbda_exc = self.ran_lmbda_exc()
        self.lmbda_intra = self.ran_lmbda_intra()
        self.g_exc = self.ran_g_exc()
        self.g_inh = self.ran_g_inh()
        self.between_inh_neurons = self.ran_between_inh_neurons(res)
 
    def ran_lmbda_exc(self):
        return random.uniform(0.5,3)
    def ran_lmbda_intra(self):
        return random.uniform(0.5,3)
    def ran_g_exc(self):
        return random.uniform(0.01,0.5)
    def ran_g_inh(self):
        smaller = False
        while smaller == False:
            g_inh = random.uniform(0.01,0.5)
            if g_inh < self.g_exc:
                smaller = True 
        return g_inh
    def ran_between_inh_neurons(self, res):
        x_len, y_len = res
        return random.uniform(0,0.25*x_len*y_len)


## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
################## ################## ################## ################## ################## ################## ################## ################## ################## ################## #################
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## Following used for random parameter search:
# for trial in range(1000):
#     res = (50,50)
#     time_steps = 500
#     params = ran_param(res)
#     runner = run(res,time_steps)
#     runner.run(g_bar_s_exc=params.g_exc, g_bar_s_inh=params.g_inh, lmbda_exc=params.lmbda_exc, lmbda_intra=params.lmbda_intra, between_inh_neurons=params.between_inh_neurons, updates=True)
#     name = "g_exc="+str(params.g_exc)+", g_inh="+str(params.g_inh)+", lmbda_exc="+str(params.lmbda_exc)+", lmbda_intra="+str(params.lmbda_intra)+", between_inh_neurons="+str(params.between_inh_neurons)+"_"
#     trans = translate()
#     table = trans.trans_tab(res)
#     trans.retinal_points_spikes(table, runner.spike_matrix, 500, offset=0, save=name)
#     trad_visualise.f_rate_tpgl_exc(runner.spike_matrix,time_steps, show=False, save=name)
#     trad_visualise.raster_all_exc_and_inh(runner.spike_matrix, show=False, save=name)
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#[0.03,0.135,0.24,0.27]
#[0.3,0.4,0.5]

## Following used for running simulations for different values of g_bar_s for excitatory neurons
for g in [0.03,0.135,0.24,0.27]:
    res = (50,50)
    time_steps = 500
    runner = run(res,time_steps)
    runner.run(g_bar_s_exc=g, g_bar_s_inh=0.03, updates=True)
    # runner.run(g_bar_s_exc=g, g_bar_s_inh=0.03, updates=True)
    # name = "g_exc="+str(g)+"_"+"g_inh="+str(0.03)
    name = "g_exc="+str(g)+"_"+"g_inh="+str(0.03)
    # trans = translate()
    # table = trans.trans_tab(res)
    # trans.retinal_points_spikes(table, runner.spike_matrix, 500, offset=0, save=name)
    # trans.retinal_points_spikes_avg(table, runner.spike_matrix, 25, offset=0, save=name)
    # trad_visualise.f_rate_tpgl_exc(runner.spike_matrix,time_steps, show=False, save=name)
    # trad_visualise.raster_all_exc_and_inh(runner.spike_matrix, show=False, save=name)
    trad_visualise.spiking_live(runner.spike_matrix, time_steps, save=name)
## ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------