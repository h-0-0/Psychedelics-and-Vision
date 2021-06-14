import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import math

class tools:
    # Contains various usefull tools
    def __init__(self):
        pass

    # Given a matrix, start_point and end_point, 'colours' in matrix entries to 1 
    # on line from start_point to end_point
    def colour_line(self, Matrix, start_point,end_point):
        x_a, y_a = start_point
        x_b, y_b = end_point

        if x_a <= x_b :
            x_1 = x_a
            y_1 = y_a
            x_2 = x_b
            y_2 = y_b
        else:
            x_1 = x_b
            y_1 = y_b
            x_2 = x_a
            y_2 = y_a

        if (x_2 - x_1) == 0:
            for i in range(y_1,y_2+1):
                Matrix[x_1,i] = 1
        else:
            m = (y_2-y_1) / (x_2-x_1)
            c = ( y_1*x_2 - y_2*x_1 ) / (x_2-x_1)
            to_colour = []
            for i in range(x_2-x_1):
                y_new = round( m * (x_1+i) + c )
                point = (x_1+i, y_new)
                to_colour.append(point)
        
            for point in to_colour:
                i, j = point
                Matrix[i,j] = 1

    # Creates and returns grid of points for hexagons
    def hex_corners(self, b, res=(100,100)):
        Matrix = np.zeros(res)
        x_len, y_len = res
        shift = False
        for i in reversed( range(0, x_len, round( b / np.sqrt(2) ) ) ):
            if shift == False:
                for j in range(0, y_len, b):
                    Matrix[i,j] = 1
                shift = True
            else:
                for j in range( round(b/2), y_len, b):
                    Matrix[i,j] = 1
                shift = False
        return Matrix

    # Given matrix finds bottom left entry set as 1 in matrix
    def hex_start(self, Matrix):
        x_len, y_len = Matrix.shape
        found = False
        x_list = [x for x in reversed(range(x_len))]
        y_list = [x for x in range(y_len)]
        i = 0
        j = 0
        while (found == False):
            ind_1 = x_list[i]
            ind_2 = y_list[j]
            if Matrix[ ind_1, ind_2 ] == 1 :
                out = (ind_1, ind_2)
                found = True
            else:
                if j == (y_len-1):
                    i +=1
                    j =0
                else:
                    j +=1
        return out

    # Find next right horizontal 1 in given matrix from current point
    def hex_right_horz(self, Matrix, point):
        x_len, y_len = Matrix.shape
        if point != False:
            x, y = point
            found = False

            while (found ==False):
                y +=1
                if (0<=x<x_len) and (0<=y<y_len):
                    if y == y_len:
                        next_point = False
                    elif Matrix[x,y] == 1:
                        next_point = (x,y)
                        found = True
                    else:
                        pass
                else:
                    next_point = False
                    break
        else:
            next_point = False
        return next_point
    
    # Find point on upwards diagonal to the right from current 1 in hexagonal grid of ones
    def hex_right_diag(self, Matrix, point, b, is_up):
        x, y = point
        x_len, y_len = Matrix.shape
        across = int(b/2)
        up = int(b/np.sqrt(2))
        new_y = y + across
        if is_up == True:
            new_x = x - up
        else:
            new_x = x + up

        next_point = False
        if is_up == True :
            found = False
            for i in range(3):
                for j in range(3):
                    if (new_x-i <0) or (new_y+j >=y_len): 
                        next_point = False
                        break
                    elif Matrix[new_x-i,new_y+j] == 1:
                        next_point = (new_x-i, new_y+j)
                        found = True
                        break
                    else:
                        pass
                if found == True:
                    break
        else:
            found = False
            for i in range(2):
                for j in range(2):
                    if (new_x+i >= x_len) or (new_y+j >=y_len): 
                        next_point = False
                        found = True
                        break
                    elif Matrix[new_x+i,new_y+j] == 1:
                        next_point = (new_x+i, new_y+j)
                        found = True
                        break
                    else:
                        pass
                if found == True:
                    break
        return next_point

    # Draws a hexagons onto a grid of hex points 
    def draw_hex(self, Matrix, b):
        x_len, _ = Matrix.shape
        
        start = tools.hex_start(Matrix)
        start_points = []
        to_draw = []
        for i in reversed(range(x_len)):
            if Matrix[i,start[1]]==1:
                start_points.append( (i,start[1]) )

        current_point = start
        go_until = math.ceil(x_len/ ((5/2)*b))
        for i in start_points:
            current_point = i
            for j in range(go_until):
                next_point = tools.hex_right_horz(Matrix,current_point)
                if next_point !=False:
                    # tools.colour_line(Matrix,current_point,next_point)
                    to_draw.append( (current_point,next_point) )

                    current_point = next_point
                    next_point_up = tools.hex_right_diag(Matrix, current_point, b, True)
                    next_point_down = tools.hex_right_diag(Matrix, current_point, b, False)
                    if (next_point_up !=False):
                        # tools.colour_line(Matrix,current_point,next_point_up)
                        to_draw.append( (current_point,next_point_up) )

                        current_point_up = next_point_up
                        next_point_up = tools.hex_right_horz(Matrix,current_point_up)

                        if (next_point_up !=False):
                            # tools.colour_line(Matrix,current_point_up,next_point_up)
                            to_draw.append( (current_point_up,next_point_up) )

                            current_point_up = next_point_up
                            next_point = tools.hex_right_diag(Matrix,current_point_up,b,False)
                            if next_point != False:
                                # tools.colour_line(Matrix, current_point_up, next_point)
                                to_draw.append( (current_point_up,next_point) )
                
                    if (next_point_down !=False):
                        # tools.colour_line(Matrix,current_point,next_point_down)
                        to_draw.append( (current_point,next_point_down) )

                        current_point_down = next_point_down
                        next_point_down = tools.hex_right_horz(Matrix,current_point_down)

                        if (next_point_down !=False):
                            # tools.colour_line(Matrix,current_point_down,next_point_down)
                            to_draw.append( (current_point_down,next_point_down) )

                            current_point_down = next_point_down
                            next_point = tools.hex_right_diag(Matrix,current_point_down,b,True)
                            if next_point !=False:
                                # tools.colour_line(Matrix, current_point_down, next_point)
                                to_draw.append( (current_point_down,next_point) )
                current_point = next_point
        for i in to_draw:
            cur ,nex = i
            tools.colour_line(Matrix, cur, nex)
        return Matrix

    # Finds first point on row, given row and matrix
    def first_one_in_x(self, Matrix, row):
        _, y_len = Matrix.shape
        start = False
        for y in range(y_len):
            print(y)
            if Matrix[row,y] == 1:
                start = (row,y)
                break
        return start
    
    # Gets rid of dots in centre of hexagons
    def hex_remove_dots(self, Matrix, b):
        x_len, y_len = Matrix.shape
        shift = False
        for i in range(0, x_len, round( b / np.sqrt(2) ) ):
            at_end = False
            beg = True 
            while at_end == False:
                if shift == False:
                    if beg == True:
                        current = 2*b
                        Matrix[i,current] = 0
                        beg = False
                    else:
                        current = current + 3*b
                        if current >= y_len:
                            at_end =True
                            shift = True
                        else:
                            Matrix[i, current] = 0
                else:
                    if beg == True:
                        current = round(b/2)
                        Matrix[i,current] = 0
                        beg = False
                    else:
                        current = current + 3*b
                        if current >= y_len:
                                at_end =True
                                shift = False
                        else:
                                Matrix[i, current] = 0
        return Matrix
tools = tools()

class translate:
    # Class for translation between cortex to retina given k and eps(ilon)
    def __init__(self,k,eps,w_0):
        self.k = k
        self.eps = eps
        self.w_0 = w_0
        # ^ theese are constants
    
    # For co-ords x,y in cortex returns complex no. representing point in retina
    def point_cortex_to_retina(self,x,y):
        z = (self.w_0 * np.sqrt(self.eps))* np.exp( (x+y*1j)* np.sqrt( (math.pi * self.eps) / (4*self.k) ))
        return z 

    # Creates table containing retina co-ords for cortex height x and width y
    def trans_tab(self,x,y):
        table = np.empty((x,y,2))
        for i in range(x):
            for j in range(y):
                z = self.point_cortex_to_retina(i,j)
                table[i,j,0] = z.real
                table[i,j,1] = z.imag
        return table
    
    # Returns list of x and y retinal co-ords whose corresponding cortex co-ords are value 1
    def retinal_points(self,table,cortex):
        x = np.array([])
        y = np.array([])
        n_1, n_2, _ = table.shape
        for i in range(n_1):
            for j in range(n_2):
                if cortex[i,j] == 1:
                    x = np.append(x, table[i,j,0])
                    y = np.append(y, table[i,j,1])
        return x, y


class draw:
    # Class for drawing what we see in the retina
    def __init__(self,file_name, pic_height=100, pic_width=100):
        self.file_name = file_name
        self.pic_height = pic_height
        self.pic_width = pic_width

    # Creates and saves plot of given x and y co-ords
    def plot(self,x,y):
        plt.plot(x,y,'o')
        plt.savefig(self.file_name+'_plot')
    
    # Given a matrix produces and saves plot of said matrix
    def plot_matrix(self, Matrix, rename = False):
        plt.matshow(Matrix)
        if rename == False :
            plt.savefig(self.file_name+'_matrix')
        else:
            plt.savefig(rename+'_matrix')

    # Creates matrix and of retinal co-ords
    def make_matrix(self, x, y):
        x_min = np.amin(x)
        x_max = np.amax(x)

        y_min = np.amin(y)
        y_max = np.amax(y)

        bin_x = (x_max-x_min)/(self.pic_width-1)
        bin_y = (y_max-y_min)/(self.pic_height-1)

        image = np.zeros((self.pic_height,self.pic_width))
        x_coord = np.empty((len(x)),dtype='int')
        y_coord = np.empty((len(y)),dtype='int')

        for i in range(len(x)):
            x_coord[i] = int( (x[i]-x_min)/bin_x )
            y_coord[i] = int( (y[i]-y_min)/bin_y )
       
        for i in range(len(x)):
            image[ x_coord[i],y_coord[i] ] = 1
        
        return image
    
    # Given a matrix produces and saves image of said matrix
    def save_image(self, Matrix, rename = False):
        if rename == False:
            img.imsave(self.file_name+'.jpg',Matrix)
        else:
            img.imsave(rename+'.jpg',Matrix)


class fake_cortex:
    # Creates fake(test) representaions of the activty in the cortex 
    def __init__(self, x, y, bar_gap=10, bar_width=6, line_gap=1, hex_b = 5):
        self.x = x
        self.y = y
        self.cortex = np.zeros((x,y))
        self.bar_gap = bar_gap
        self.bar_width = bar_width
        self.line_gap = line_gap+1
        self.hex_b = hex_b
    
    def reset(self):
        self.cortex = np.zeros((self.x,self.y))


    # Creates a cortex pattern corresponding to funnel form constant
    def funnel(self):
        x_consider = []
        count=0
        reset = self.bar_width +self.bar_gap
        for i in range(self.x):
            if count<= self.bar_width:
                x_consider.append(i)
                count+=1
            elif count<= reset:
                count+=1
            else:
                count=0
        
        y_consider = range(self.y)[::self.line_gap]

        for i in x_consider:
            for j in y_consider:
                self.cortex[i,j] = 1

    # Creates a cortex pattern corresponding to tunnel form constant
    def tunnel(self):
        y_consider = []
        count=0
        reset = self.bar_width +self.bar_gap
        for i in range(self.y):
            if count<= self.bar_width:
                y_consider.append(i)
                count+=1
            elif count<= reset:
                count+=1
            else:
                count=0
        
        x_consider = range(self.x)[::self.line_gap]

        for i in x_consider:
            for j in y_consider:
                self.cortex[i,j] = 1
    
    # Creates a cortex pattern corresponding to cobweb form constant
    def cobweb(self):
        x_consider = range(self.x)[::self.bar_gap]
        y_consider = range(self.y)[::self.bar_gap]

        for i in x_consider:
            for j in range(self.y):
                self.cortex[i,j] = 1
        
        for i in y_consider:
            for j in range(self.x):
                self.cortex[j,i] = 1

    # Creates a cortex pattern corresponding to lattice form constant
    def lattice(self):
        self.cortex = tools.hex_corners(self.hex_b,res=(self.x,self.y))
        tools.draw_hex(self.cortex, self.hex_b)
        tools.hex_remove_dots(self.cortex, self.hex_b)





# Creates a form constant on cortex and then gives corresponding co-ords in retina
def create_f_con_points(f_con,table_height,table_width,bar_gap,bar_width,line_gap,k,eps,w_0):
    cor = fake_cortex(table_height,table_width,bar_gap,bar_width,line_gap)
    # drawing = draw(f_con)

    if f_con == 'funnel':
        cor.funnel()
    elif f_con == 'tunnel':
        cor.tunnel()
    elif f_con == 'cobweb' :
        cor.cobweb()
    else:
        pass

    draw.plot_matrix(cor.cortex)

    k_eps = translate(k,eps,w_0)
    table = k_eps.trans_tab(table_height,table_width)
    x,y = k_eps.retinal_points(table,cor.cortex)

    cor.reset()

    return x,y
               

f_con = 'cobweb'
draw = draw(f_con) 
x,y = create_f_con_points(f_con,        100,          100,      10,      6,        1,       10,  2,   1  )
image = draw.make_matrix(x,y)
draw.save_image(image)
