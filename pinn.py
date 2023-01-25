import tensorflow as tf
from keras.models import load_model
import scipy.optimize
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os 
import random

# visualization
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

# gif
import os
from PIL import Image
from IPython.display import Image as Img
from IPython.display import display

class basic_setting:
    def __init__(self, space_range, time_range, the_number_of_equations=1, initial_condition=False, boundary_condition=False, the_number_of_collocation_points=30, widths=(3, 3)):
        r"""
        [Args]
            space_range : `list`
            time_range : `list`
            the_number_of_equations : `int`
        
        [Example]
        >>> pinn.basic_setting([(0.0, 2.0)], [], 1) # X=(0.0, 2.0)
        >>> pinn.basic_setting([(0.0, 2.0), (-1.0, 2.0)], [], 2) # X=(0.0, 2.0), Y=(-1.0, 2.0)
        >>> pinn.basic_setting([(0.0, 2.0), (0.0, 1.0)], (0.0, 1.0), 1) # X=(0.0, 2.0), Y=(0.0, 1.0), T=(0.0, 1.0)
        """
        self.space_range = space_range
        self.time_range = time_range
        self.the_number_of_equations = the_number_of_equations
        self.space_dimension = len(self.space_range)
        self.time_existence = int(len(self.time_range)/2)
        self.initial_condition = initial_condition
        self.boundary_condition = boundary_condition
        self.the_number_of_collocation_points = the_number_of_collocation_points
        self.widths = widths
        self.loss_u = []
        self.loss_v = []
        self.loss_u_ic = []
        self.loss_v_ic = []
        self.loss_u_bc = []
        self.loss_v_bc = []

        assert type(self.space_range) is list , "input list into 1st component"
        assert type(self.time_range) is list, "input list into 2nd component"
        assert self.space_dimension in [0, 1, 2, 3], "space dimension should be either 0, or 1, or 2, or 3"
        assert self.time_existence in [0, 1], "time existence should be either 0 or 1"
        assert self.space_dimension + self.time_existence > 0, "space dimension + time existence > 0"
        assert type(self.the_number_of_equations) is int, "input the number of equations as integer"
        assert self.the_number_of_equations > 0, "the number of equations > 0"
        assert type(self.widths) is tuple, "input two widths of two layers as tuple"
        assert len(self.widths) == 2, "input two widths of two layers as tuple"
        
        if self.space_dimension == 1:
            assert len(self.space_range[0]) == 2, "input minimum and maximum for x"

            self.x_min = self.space_range[0][0]
            self.x_max = self.space_range[0][1]
            message = "Domain: ({}, {})".format(self.x_min, self.x_max)

        elif self.space_dimension == 2:
            assert len(self.space_range[0]) == 2, "input minimum and maximum for x"
            assert len(self.space_range[1]) == 2, "input minimum and maximum for y"

            self.x_min = self.space_range[0][0]
            self.x_max = self.space_range[0][1]
            self.y_min = self.space_range[1][0]
            self.y_max = self.space_range[1][1]
            message = "Domain: ({}, {})X({}, {})".format(self.x_min, self.x_max, self.y_min, self.y_max)
            
        elif self.space_dimension == 3:
            assert len(self.space_range[0]) == 2, "input minimum and maximum for x"
            assert len(self.space_range[1]) == 2, "input minimum and maximum for y"
            assert len(self.space_range[2]) == 2, "input minimum and maximum for z"

            self.x_min = self.space_range[0][0]
            self.x_max = self.space_range[0][1]
            self.y_min = self.space_range[1][0]
            self.y_max = self.space_range[1][1]
            self.z_min = self.space_range[2][0]
            self.z_max = self.space_range[2][1]
            message = "Domain: ({}, {})X({}, {})X({}, {})".format(self.x_min, self.x_max, self.y_min, self.y_max, self.z_min, self.z_max)

        if self.time_existence == 1:

            self.t_min = time_range[0]
            self.t_max = time_range[1]
            if self.space_dimension == 0:
                print("Domain: [{}, {})".format(self.t_min, self.t_max))
            else: 
                print(message + "X[{}, {})".format(self.t_min, self.t_max))
        
        elif self.time_existence == 0:
            self.t_min = 0.0
            self.t_max = 0.0
            print(message)
        
        if not os.path.exists("Figure"):
            os.makedirs("Figure")

        if not os.path.exists("Data"):
            os.makedirs("Data")

class creation(basic_setting):
    @staticmethod
    def open_interval(minimum=0.0, maximum=1.0, the_number_of_points=30, to_tf=False):
        r"""
        An open interval from minimum to maximum.

        [Args]
            minimum : `float`
            maximum : `float`
            the_number_of_points : `int`
            to_tf : `bool`

        [Example]
        >>> pinn.creation.open_interval(0.0, 2.0, to_tf=True)
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([1.00806  , 1.6581106, 1.4972426, 0.435453], dtype=float32)>
        """
        assert maximum >= minimum, "maximum >= minimum"
        
        if maximum > minimum:
            result = (np.random.random_sample((the_number_of_points,))*(maximum - minimum) + minimum).reshape(-1, )
            result[result == 0.0] = (maximum - minimum)/the_number_of_points

        elif maximum == minimum:
            result = np.ones(the_number_of_points)*maximum

        if to_tf == True:
            return tf.convert_to_tensor(result, dtype=tf.float32) 
        else:
            return result
    
    @staticmethod
    def half_interval(minimum=0.0, maximum=1.0, the_number_of_points=30, to_tf=False):
        r"""
        An half interval (left closed) from minimum to maximum.

        [Args]
            minimum : `float`
            maximum : `float`
            the_number_of_points : `int`
            to_tf : `bool`

        [Example]
        >>> pinn.creation.half_interval(minimum=0.0, maximum=1.0, the_number_of_points=4, to_tf=True)
        <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.9253717 , 0.6767667 , 0.        , 0.34709215], dtype=float32)>
        """
        assert maximum > minimum, "maximum > minimum"
        
        result = (np.random.random_sample((the_number_of_points - 1,))*(maximum - minimum) + minimum).reshape(-1, )
        result = np.insert(result, np.random.choice(the_number_of_points - 1, size=1), minimum)

        if to_tf == True:
            return tf.convert_to_tensor(result, dtype=tf.float32) 
        else:
            return result
    
    def input_set(self, the_number_of_points=30, time=False, IC=False, BC=False, to_tf=False):
        r"""
        [Args]
            the_number_of_points : `int`
            time : `bool`
            IC : `bool`
            BC : `bool`
            to_tf : `bool`

        [Example]
        >>> setting = pinn.basic_setting([(0.0, 2.0)], [])
        >>> train = pinn.creation.input_set(setting, the_number_of_points=4, to_tf=True)
        >>>    
        >>> setting = pinn.basic_setting([(0.0, 2.0), (0.0, 1.0)], [])
        >>> train = pinn.creation.input_set(setting, the_number_of_points=4, to_tf=True)
        >>>
        >>> setting = pinn.basic_setting([(0.0, 2.0), (0.0, 1.0)], [0.0, 0.1])
        >>> train = pinn.creation.input_set(setting, the_number_of_points=4, IC=True, to_tf=True)
        """
        assert IC*BC == 0, "IC and BC cannot be True simultaneously"

        if BC == False:
            # making space
            space = creation.open_interval(minimum=self.space_range[0][0], maximum=self.space_range[0][1], the_number_of_points=the_number_of_points)
            
            for range_for_one_variable in self.space_range[1:]:
                space = np.vstack((space, creation.open_interval(minimum=range_for_one_variable[0], maximum=range_for_one_variable[1], the_number_of_points=the_number_of_points)))

            if IC == True:
                assert self.time_existence == 1, "input minimum and maximum for t"

                result = np.vstack((space, creation.open_interval(minimum=0.0, maximum=0.0, the_number_of_points=the_number_of_points)))
                
                if to_tf == True:
                    result = result.T

            else:
                if time == False:
                    result = space
                else: 
                    assert self.time_existence == 1, "input minimum and maximum for t"

                    result = np.vstack((space, creation.open_interval(minimum=self.t_min, maximum=self.t_max, the_number_of_points=the_number_of_points)))
        else:
            zero = np.zeros(the_number_of_points)
            
            # making boundary of the space
            if self.space_dimension == 1:
                space_bdy = np.array([self.x_min, self.x_max])

            elif self.space_dimension == 2:
                x = creation.open_interval(minimum=self.x_min, maximum=self.x_max, the_number_of_points=the_number_of_points)
                y = creation.open_interval(minimum=self.y_min, maximum=self.y_max, the_number_of_points=the_number_of_points)

                upper_bdy_x = np.stack((x, zero + self.y_min)).T
                lower_bdy_x = np.stack((x, zero + self.y_max)).T
                upper_bdy_y = np.stack((zero + self.x_min, y)).T
                lower_bdy_y = np.stack((zero + self.x_max, y)).T

                space_bdy = np.vstack((upper_bdy_x, lower_bdy_x, upper_bdy_y, lower_bdy_y))

                idx = np.random.choice(np.shape(space_bdy)[0], size=the_number_of_points, replace=False)
                space_bdy = space_bdy[idx].T

            elif self.space_dimension == 3:
                print("TODO")
                # x = creation.open_interval(minimum=self.x_min, maximum=self.x_max, the_number_of_points=the_number_of_points)
                # y = creation.open_interval(minimum=self.y_min, maximum=self.y_max, the_number_of_points=the_number_of_points)
                # z = creation.open_interval(minimum=self.z_min, maximum=self.z_max, the_number_of_points=the_number_of_points)

            if time == True:
                assert self.time_existence == 1, "input minimum and maximum for t"

                if self.space_dimension == 1:
                    extended_space_bdy = np.array([random.choice(space_bdy) for dummy in range(the_number_of_points)])
                    result = np.vstack((extended_space_bdy, creation.open_interval(minimum=self.t_min, maximum=self.t_max, the_number_of_points=the_number_of_points)))

                elif self.space_dimension == 2:
                    result = np.vstack((space_bdy, creation.open_interval(minimum=self.t_min, maximum=self.t_max, the_number_of_points=the_number_of_points)))

                elif self.space_dimension == 3:
                    print("TODO")
            else:
                result = space_bdy
        
        if to_tf == True:
            if self.space_dimension == 1 and BC == True and time == False:
                result = tf.convert_to_tensor(result, dtype=tf.float32)
                return tf.reshape(result, [2, 1])

            if IC == True:
                return tf.convert_to_tensor(result, dtype=tf.float32)

            else:
                return tf.convert_to_tensor(result.T, dtype=tf.float32)
        else:
            return result
    
    def train_and_test_sets(self, proportion_of_initial_and_boundary_sets=0.1):
        r"""
        Constructing train set and test set
        'space', 'space X time', 'Neumann boundary', 'Neumann boundary X time' ==> train
        'initial', 'Dirichlet boundary', 'Dirichlet boundary X time' ==> test

        [Args]
            proportion_of_initial_and_boundary_sets : `float`, between 0.0 and 1.0
        """ 
        assert 0.0 < proportion_of_initial_and_boundary_sets <= 1.0, "0 < proportion of the number of test set <= 1"

        try:
            the_number_of_initial_and_boundary_set = int(self.the_number_of_collocation_points*proportion_of_initial_and_boundary_sets)
        except:
            the_number_of_initial_and_boundary_set = 1

        # train set
        if self.boundary_condition == "Neumann":
            train_inner = creation.input_set(self, the_number_of_points=self.the_number_of_collocation_points, time=self.time_existence)
            train_boundary = creation.input_set(self, the_number_of_points=the_number_of_initial_and_boundary_set, time=self.time_existence, BC=True)
            train = np.hstack((train_inner, train_boundary)).T
            np.random.shuffle(train)
            train = train.T

        else:
            train = creation.input_set(self, the_number_of_points=self.the_number_of_collocation_points, time=self.time_existence)

        # test set 
        if self.initial_condition == True and self.boundary_condition == "Dirichlet":
            test_initial = creation.input_set(self, the_number_of_points=the_number_of_initial_and_boundary_set, IC=True)
            test_boundary = creation.input_set(self, the_number_of_points=the_number_of_initial_and_boundary_set, time=self.time_existence, BC=True)
            test = np.hstack((test_initial, test_boundary)).T
            np.random.shuffle(test)
            test = test.T

        elif self.initial_condition == True and self.boundary_condition != "Dirichlet":
            test = creation.input_set(self, the_number_of_points=the_number_of_initial_and_boundary_set, IC=True)

        elif self.initial_condition == False and self.boundary_condition == "Dirichlet":
            test = creation.input_set(self, the_number_of_points=the_number_of_initial_and_boundary_set, time=self.time_existence, BC=True)

        else:
            print("There is no test set. Either Neumann boundary condition or Dirichlet boundary condition is needed.")
            return None
        
        return train, test
    
    @staticmethod
    def get_permutations_1d(x_list, fixed_time=0.0):
        for x in x_list:
            yield [x, fixed_time]

    
    @staticmethod
    def get_permutations_2d(x_list, y_list, fixed_time=0.0):
        for x in x_list:
            for y in y_list:
                yield [x, y, fixed_time]
    
    @staticmethod
    def get_permutations_3d(x_list, y_list, z_list, fixed_time=0.0):
        for x in x_list:
            for y in y_list:
                for z in z_list:
                    yield [x, y, z, fixed_time]

    def prediction_set_for_fixed_time(self, fixed_time=0.0, the_number_of_points=30):
        r"""
        [Args]
            fixed_time : `float`
            the_number_of_points : `int`
        """
        assert self.time_existence == 1, "time does not exist"

        if self.space_dimension == 1:
            X = np.linspace(self.x_min, self.x_max, the_number_of_points, endpoint=True)

            return np.array(list(creation.get_permutations_1d(X, fixed_time)))

        elif self.space_dimension == 2:
            X = np.linspace(self.x_min, self.x_max, the_number_of_points, endpoint=True)
            Y = np.linspace(self.y_min, self.y_max, the_number_of_points, endpoint=True)

            return np.array(list(creation.get_permutations_2d(X, Y, fixed_time)))

class visualization(basic_setting):
    def _of_train_and_test_set(self, the_number_of_points=30):
        r"""
        [Args]
            the_number_of_points : `int`
        
        [Example]
        >>> setting = pinn.basic_setting([[0.0, 2.0], [0.0, 1.0]], [0.0, 1.0], 1)
        >>> pinn.visualization._of_train_and_test_set(setting, the_number_of_points=500)
        """
        assert 0 < self.space_dimension + self.time_existence < 4, "four dimensions can not be visualized"

        if self.space_dimension == 3:
            points = creation.input_set(self, the_number_of_points=the_number_of_points)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title('Domain')
            ax.set_xlabel('$x$', labelpad=10, fontdict={'size': 10})
            ax.set_ylabel('$y$', labelpad=10, fontdict={'size': 10})
            ax.set_zlabel('$t$', labelpad=10, fontdict={'size': 10}, rotation = 0)
            ax.scatter3D(points[0], points[1], points[2], 'gray')

        elif self.space_dimension == 2 and self.time_existence == 1:
            points = creation.input_set(self, the_number_of_points=the_number_of_points, time=True)
            points_IC = creation.input_set(self, the_number_of_points=the_number_of_points, IC=True)

            fig = plt.figure()
            ax = plt.axes(projection='3d')
            ax.set_title('Domain')
            ax.set_xlabel('$x$', labelpad=10, fontdict={'size': 10})
            ax.set_ylabel('$y$', labelpad=10, fontdict={'size': 10})
            ax.set_zlabel('$t$', labelpad=10, fontdict={'size': 10}, rotation = 0)
            ax.scatter3D(points[0], points[1], points[2], 'gray')
            ax.scatter3D(points_IC[0], points_IC[1], points_IC[2], 'gray')

        elif self.space_dimension == 2:
            points = creation.input_set(self, the_number_of_points=the_number_of_points)
            
            fig = plt.figure()        
            plt.grid()                
            plt.title('Domain')              
            plt.xlabel('$x$', fontsize=11)              
            plt.ylabel('$y$', fontsize=11, rotation=0)  
            plt.scatter(points[0], points[1])
            plt.show
        
        elif self.space_dimension == 1 and self.time_existence == 1:
            points = creation.input_set(self, the_number_of_points=the_number_of_points, time=True)
            points_IC = creation.input_set(self, the_number_of_points=the_number_of_points, IC=True)
            
            fig = plt.figure()        
            plt.grid()                
            plt.title('Domain')              
            plt.xlabel('$x$', fontsize=11)              
            plt.ylabel('$t$', fontsize=11, rotation=0)  
            plt.scatter(points[0], points[1])
            plt.scatter(points_IC[0], points_IC[1])
            plt.show
        
        elif self.space_dimension == 1 and self.time_existence == 0:
            points = creation.input_set(self, the_number_of_points=the_number_of_points)
            
            fig = plt.figure()        
            plt.grid()                
            plt.title('Domain')              
            plt.xlabel('$x$', fontsize=11)              
            plt.scatter(points, np.zeros(the_number_of_points))
            plt.show
        
        else:
            print("In case it is not implemented")
    
    def _of_prediction_set_with_heatmap(self, data, bar_min=0.0, bar_max=0.0, fixed_time=0.0, title='u', save_folder='u'):
        fig = plt.figure(figsize=(5, 4))
        ax = sns.heatmap(data, vmin=bar_min, vmax=bar_max, cmap=sns.color_palette("RdPu", 100))
            
        ax.set_title(title + "$({},x,y)$".format(np.round(fixed_time, 4)) , fontsize=16)
        ax.set_xlabel('$x$', fontsize=12)
        ax.set_ylabel('$y$', fontsize=12, rotation=0)

        x_labels = [item.get_text() for item in ax.get_xticklabels()]
        for i, x in zip(range(len(ax.get_xticklabels())), np.round(np.linspace(self.x_min, self.x_max, num=len(ax.get_xticklabels())), 2)):
            x_labels[i] = str(x)
        ax.set_xticklabels(x_labels)

        y_labels = [item.get_text() for item in ax.get_yticklabels()]
        for i, y in zip(range(len(ax.get_yticklabels())), np.round(np.linspace(self.y_max, self.y_min, num=len(ax.get_yticklabels())), 2)):
            y_labels[i] = str(y)
        ax.set_yticklabels(y_labels)

        plt.show()
        
        if not os.path.exists("Figure/{}".format(save_folder)):
            os.makedirs("Figure/{}".format(save_folder))

        fig.savefig("Figure/{}/{}({},x,y).png".format(save_folder, save_folder, np.round(fixed_time, 4)), dpi=1000)

        plt.close()

    def _of_loss_function_record(self, labels=["$R_{u}$"]):
        epoch = np.arange(self.final_epoch)

        fig = plt.figure(figsize=(5, 4))

        if self.the_number_of_equations == 1:
            plt.plot(epoch, self.loss_u, label=labels[0])

            if self.initial_condition is not False:
                plt.plot(epoch, self.loss_u_ic, label=labels[1])
            if self.boundary_condition is not False:
                plt.plot(epoch, self.loss_u_bc, label=labels[2])

        elif self.the_number_of_equations == 2:
            plt.plot(epoch, self.loss_u, label=labels[0])
            plt.plot(epoch, self.loss_v, label=labels[1])

            if self.initial_condition is not False:
                plt.plot(epoch, self.loss_u_ic, label=labels[2])
                plt.plot(epoch, self.loss_v_ic, label=labels[3])
            if self.boundary_condition is not False:
                plt.plot(epoch, self.loss_u_bc, label=labels[4])
                plt.plot(epoch, self.loss_v_bc, label=labels[5])
            
        plt.legend()
        plt.show()

    def _of_prediction_set_with_3D_for_fixed_time(self, data, z_min=0.0, z_max=0.0, fixed_time=0.0, title='u', save_folder='u'):
        X, Y = np.meshgrid(data.columns, data.index)

        fig = plt.figure(figsize=(10, 20))
        ax = fig.add_subplot(1, 1, 1, projection='3d')
        ax.set_title(title)
        ax.text2D(0.0, 1.0, "$t$={}".format(np.round(fixed_time, 4)), transform=ax.transAxes)
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        ax.set_zlim(z_min, z_max)
        plt.ylabel('$y$', labelpad=10, fontdict={'size': 10})
        plt.xlabel('$x$', labelpad=10, fontdict={'size': 10})
        ax.plot_trisurf(X.reshape(-1,),
                        Y.reshape(-1,),
                        data.values.reshape(-1,),
                        edgecolor='none', cmap='gray', alpha=0.6, vmin=0, vmax=8)

        plt.show()
        
        if not os.path.exists("Figure/{}".format(save_folder)):
            os.makedirs("Figure/{}".format(save_folder))
        
        plt.savefig("Figure/{}/u({},x,y).png".format(save_folder, np.round(fixed_time, 4)), dpi=500, bbox_inches = 'tight', pad_inches=0)

        plt.close()
        
    # gif
    def generate_gif(self, folder_name="u", duration=500):
        assert self.time_existence == 1, "time does not exist"
        
        img_list = os.listdir("Figure/{}".format(folder_name))
        img_list = ["Figure/{}".format(folder_name) + '/' + x for x in img_list]
        images = [Image.open(x) for x in img_list]
        
        im = images[0]
        im.save('Figure/{}/{} animation from t={} to t={}.gif'.format(folder_name, folder_name, self.t_min, self.t_max), save_all=True, append_images=images[1:], loop=0xff, duration=duration)
        # loop 반복 횟수
        # duration 프레임 전환 속도 (500 = 0.5초)
        return Img(url='Figure/{}/{} animation from t={} to t={}.gif'.format(folder_name, folder_name, self.t_min, self.t_max))

class derivative(basic_setting):
    def dim1(self, function, domain, to_array=False):
        r"""
        [Args]
            function : `function`
            domain : `float` or `tensor` or `numpy.array`
            to_array : `bool`
        
        [Returns]
            `tuple` including x, y, dy_dx, d2y_dxx

        [Example]
        >>> setting = pinn.basic_setting([[0.0, 2.0]], [], 1) 
        >>> train = pinn.creation.input_set(setting, the_number_of_points=5, to_tf=True)
        >>> # non constant function
        >>> def f(x):
        >>>    return x**3
        >>> pinn.derivative.dim1(setting, f, train)
        >>> # constant function, the inputted data should be numpy.array
        >>> def C(x):
        >>>     return tf.constant(2.0)     # the value of the function should be converted to tensor
        >>> pinn.derivative.dim1(setting, C, train)

        [Note]
            Diverse data can be inputted except for constant functions. For example,
            4.0
            [4.0]
            [4.0, 1.0]
            [[4.0], [1.0], [3.0]]
            tf.convert_to_tensor([4.0, 1.0], dtype=tf.float32)
            tf.convert_to_tensor([[4.0], [1.0], [3.0]], dtype=tf.float32)
        """
        assert self.space_dimension == 1 and self.time_existence == 0, "only one dimensional space can be handled"

        x = tf.Variable(domain, trainable=True)

        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                y = function(x)
            dy_dx = tape1.gradient(y, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2y_dxx = tape2.gradient(dy_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        try:
            result = tf.convert_to_tensor([x.numpy(), y.numpy(), dy_dx.numpy(), d2y_dxx.numpy()], dtype=tf.float32)
        except:
            y = tf.ones([domain.size, ])*(y.numpy())
            result = tf.convert_to_tensor([x.numpy(), y.numpy(), dy_dx.numpy(), d2y_dxx.numpy()], dtype=tf.float32)

        if to_array == True:
            return result.numpy()
        else:
            return result 
    
    def dim2(self, function, domain, to_array=False):
        r"""
        [Args]
            function : `function`
            domain : `numpy.array`
            to_array : `bool`
        
        [Returns]
            `tuple` including x, y, u, du_dx, du_dy, d2u_dxx, d2u_dxy, d2u_dyx, d2u_dyy
        
        [Example]
        >>> def function(x, y):
        >>>     return x**3 + y*2 + 2*x*y 
        >>> # time not involved
        >>> setting = pinn.basic_setting([[-1.0, 0.0], [0.0, 1.0]], [], 1) 
        >>> train = pinn.creation.input_set(setting, the_number_of_points=5)
        >>> pinn.derivative.dim2(setting, function, train, to_array=True)
        >>> # time involved
        >>> setting = pinn.basic_setting([[-1.0, 0.0]], [0.0, 1.0], 1) 
        >>> train = pinn.creation.input_set(setting, the_number_of_points=5, time=True)
        >>> pinn.derivative.dim2(setting, function, train)

        [Note]
            The inputted data can be only `numpy.array` not as in the one dimension.
        """
        assert self.space_dimension + self.time_existence == 2, "only two dimensional space can be handled"

        x = tf.Variable(domain[0], trainable=True)
        y = tf.Variable(domain[1], trainable=True)

        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                u = function(x, y)
            du_dx = tape1.gradient(u, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            du_dy = tape1.gradient(u, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dxx = tape2.gradient(du_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dxy = tape2.gradient(du_dx, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dyx = tape2.gradient(du_dy, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dyy = tape2.gradient(du_dy, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        
        try:
            result = tf.convert_to_tensor([x.numpy(), y.numpy(), u.numpy(), 
                                           du_dx.numpy(), du_dy.numpy(), 
                                           d2u_dxx.numpy(), d2u_dxy.numpy(), 
                                           d2u_dyx.numpy(), d2u_dyy.numpy()], dtype=tf.float32)
        except:
            u = tf.ones([domain[0].size, ])*(u.numpy())
            result = tf.convert_to_tensor([x.numpy(), y.numpy(), u.numpy(), 
                                           du_dx.numpy(), du_dy.numpy(), 
                                           d2u_dxx.numpy(), d2u_dxy.numpy(), 
                                           d2u_dyx.numpy(), d2u_dyy.numpy()], dtype=tf.float32)

        if to_array == True:
            return result.numpy()
        else:
            return result 

    def dim3(self, function, domain, to_array=False):
        r"""
        [Args]
            function : `function`
            domain : `numpy.array`
            to_array : `bool`
        
        [Returns]
            `tuple` including x, y, z, u, du_dx, du_dy, du_dz, d2u_dxx, d2u_dxy, d2u_dxz, d2u_dyx, d2u_dyy, d2u_dyz, d2u_dzx, d2u_dzy, d2u_dzz

        [Example]
        >>> # time not involved
        >>> setting = pinn.basic_setting([[0.0, 2.0], [-1.0, 2.0], [-3.0, -1.0]], [], 1) 
        >>> train = pinn.creation.input_set(setting, the_number_of_points=5)
        >>> def f(x, y, z):
        >>>     return x*y*z + z**2
        >>> pinn.derivative.dim3(setting, f, train)
        >>> # time involved
        >>> setting = pinn.basic_setting([[0.0, 2.0], [-1.0, 2.0]], [0.0, 1.0], 1) 
        >>> train = pinn.creation.input_set(setting, the_number_of_points=5, time=True)
        >>> def C(x, y, z):
        >>>     return tf.constant(-2.0)
        >>> pinn.derivative.dim3(setting, C, train)
        """
        assert self.space_dimension + self.time_existence == 3, "only three dimensional space can be handled"

        x = tf.Variable(domain[0], trainable=True)
        y = tf.Variable(domain[1], trainable=True)
        z = tf.Variable(domain[2], trainable=True)

        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                u = function(x, y, z)
            du_dx = tape1.gradient(u, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            du_dy = tape1.gradient(u, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            du_dz = tape1.gradient(u, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dxx = tape2.gradient(du_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dxy = tape2.gradient(du_dx, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dxz = tape2.gradient(du_dx, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dyx = tape2.gradient(du_dy, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dyy = tape2.gradient(du_dy, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dyz = tape2.gradient(du_dy, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dzx = tape2.gradient(du_dz, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dzy = tape2.gradient(du_dz, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
        d2u_dzz = tape2.gradient(du_dz, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)

        try:
            result= tf.convert_to_tensor([x.numpy(), y.numpy(), z.numpy(), u.numpy(), 
                                          du_dx.numpy(), du_dy.numpy(), du_dz.numpy(), 
                                          d2u_dxx.numpy(), d2u_dxy.numpy(), d2u_dxz.numpy(), 
                                          d2u_dyx.numpy(), d2u_dyy.numpy(), d2u_dyz.numpy(), 
                                          d2u_dzx.numpy(), d2u_dzy.numpy(), d2u_dzz.numpy()], dtype=tf.float32)
        except:
            u = tf.ones([domain[0].size, ])*(u.numpy())
            result= tf.convert_to_tensor([x.numpy(), y.numpy(), z.numpy(), u.numpy(), 
                                        du_dx.numpy(), du_dy.numpy(), du_dz.numpy(), 
                                        d2u_dxx.numpy(), d2u_dxy.numpy(), d2u_dxz.numpy(), 
                                        d2u_dyx.numpy(), d2u_dyy.numpy(), d2u_dyz.numpy(), 
                                        d2u_dzx.numpy(), d2u_dzy.numpy(), d2u_dzz.numpy()], dtype=tf.float32)

        if to_array == True:
            return result.numpy()
        else:
            return result 
    
    def dim3_neural_network(self, neural_network, train, to_array=False, omit=False):
        r"""
        [Example]
        >>> train = pinn.creation.input_set(setting, the_number_of_points=3, time=True)
        >>> derivatives = pinn.derivative.dim3_neural_network(setting, NN, train, to_array=True)
        """
        assert self.space_dimension + self.time_existence == 3, "only three dimensional space can be handled"

        if self.the_number_of_equations == 1:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:
                    x = tf.Variable(train[0], trainable=True)
                    y = tf.Variable(train[1], trainable=True)
                    z = tf.Variable(train[2], trainable=True)
                    u = tf.transpose(neural_network(tf.transpose(tf.stack([x, y, z], axis=0))))[0]
                du_dx = tape1.gradient(u, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                du_dy = tape1.gradient(u, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                du_dz = tape1.gradient(u, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2u_dxx = tape2.gradient(du_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2u_dyy = tape2.gradient(du_dy, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2u_dzz = tape2.gradient(du_dz, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            if omit == True:
                d2u_dxy = u
                d2u_dxz = u
                d2u_dyx = u
                d2u_dyz = u
                d2u_dzx = u
                d2u_dzy = u
            else:
                d2u_dxy = tape2.gradient(du_dx, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dxz = tape2.gradient(du_dx, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dyx = tape2.gradient(du_dy, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dyz = tape2.gradient(du_dy, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dzx = tape2.gradient(du_dz, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dzy = tape2.gradient(du_dz, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            result = tf.convert_to_tensor([x.numpy(), y.numpy(), z.numpy(), u.numpy(), 
                                           du_dx.numpy(), du_dy.numpy(), du_dz.numpy(), 
                                           d2u_dxx.numpy(), d2u_dxy.numpy(), d2u_dxz.numpy(), 
                                           d2u_dyx.numpy(), d2u_dyy.numpy(), d2u_dyz.numpy(), 
                                           d2u_dzx.numpy(), d2u_dzy.numpy(), d2u_dzz.numpy()], dtype=tf.float32)
            return result

        elif self.the_number_of_equations == 2:
            with tf.GradientTape(persistent=True) as tape2:
                with tf.GradientTape(persistent=True) as tape1:
                    x = tf.Variable(train[0], trainable=True)
                    y = tf.Variable(train[1], trainable=True)
                    z = tf.Variable(train[2], trainable=True)
                    uv = tf.transpose(neural_network(tf.transpose(tf.stack([x, y, z], axis=0))))
                    u = uv[0]
                    v = uv[1]
                du_dx = tape1.gradient(u, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                dv_dx = tape1.gradient(v, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                du_dy = tape1.gradient(u, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                dv_dy = tape1.gradient(v, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                du_dz = tape1.gradient(u, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                dv_dz = tape1.gradient(v, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            d2u_dxx = tape2.gradient(du_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2v_dxx = tape2.gradient(dv_dx, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2u_dyy = tape2.gradient(du_dy, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2v_dyy = tape2.gradient(dv_dy, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2u_dzz = tape2.gradient(du_dz, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            d2v_dzz = tape2.gradient(dv_dz, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)

            if omit == True:
                d2u_dxy = u
                d2v_dxy = v
                d2u_dxz = u
                d2v_dxz = v
                d2u_dyx = u
                d2v_dyx = v
                d2u_dyz = u
                d2v_dyz = v
                d2u_dzx = u
                d2v_dzx = v
                d2u_dzy = u
                d2v_dzy = v

            else:
                d2u_dxy = tape2.gradient(du_dx, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dxy = tape2.gradient(dv_dx, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dxz = tape2.gradient(du_dx, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dxz = tape2.gradient(dv_dx, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dyx = tape2.gradient(du_dy, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dyx = tape2.gradient(dv_dy, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dyz = tape2.gradient(du_dy, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dyz = tape2.gradient(dv_dy, z, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dzx = tape2.gradient(du_dz, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dzx = tape2.gradient(dv_dz, x, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2u_dzy = tape2.gradient(du_dz, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                d2v_dzy = tape2.gradient(dv_dz, y, unconnected_gradients=tf.UnconnectedGradients.ZERO)
                
                

            result = tf.convert_to_tensor([x.numpy(), y.numpy(), z.numpy(), u.numpy(), v.numpy(), 
                                            du_dx.numpy(), du_dy.numpy(), du_dz.numpy(), 
                                            dv_dx.numpy(), dv_dy.numpy(), dv_dz.numpy(),
                                            d2u_dxx.numpy(), d2u_dxy.numpy(), d2u_dxz.numpy(), 
                                            d2v_dxx.numpy(), d2v_dxy.numpy(), d2v_dxz.numpy(), 
                                            d2u_dyx.numpy(), d2u_dyy.numpy(), d2u_dyz.numpy(), 
                                            d2v_dyx.numpy(), d2v_dyy.numpy(), d2v_dyz.numpy(), 
                                            d2u_dzx.numpy(), d2u_dzy.numpy(), d2u_dzz.numpy(),
                                            d2v_dzx.numpy(), d2v_dzy.numpy(), d2v_dzz.numpy()], dtype=tf.float32)

        if to_array == True:
            return result.numpy()
        else:
            return result             
    
class normal_derivative(basic_setting):
    def indices_of_boundary(self, derivatives):
        if self.space_dimension == 1:
            x = derivatives[0].numpy()
            # indices_x_min = np.hstack((np.where(x == self.x_min)[0], np.where(x == self.x_max)[0]))
            indices_x_min = np.where(x == self.x_min)[0]
            indices_x_max = np.where(x == self.x_max)[0]
            all_indices = np.hstack((indices_x_min, indices_x_max))
            
            return indices_x_min, indices_x_max, all_indices

        elif self.space_dimension == 2:
            x = derivatives[0].numpy()
            y = derivatives[1].numpy()
            indices_x_min = np.where(x == self.x_min)[0]
            indices_x_max = np.where(x == self.x_max)[0]
            indices_y_min = np.where(y == self.y_min)[0]
            indices_y_max = np.where(y == self.y_max)[0]
            all_indices = np.hstack((indices_x_min, indices_x_max, indices_y_min, indices_y_max))

            return indices_x_min, indices_x_max, indices_y_min, indices_y_max, all_indices

        elif self.space_dimension == 3:
            x = derivatives[0].numpy()
            y = derivatives[1].numpy()
            z = derivatives[2].numpy()
            indices_x_min = np.where(x == self.x_min)[0]
            indices_x_max = np.where(x == self.x_max)[0]
            indices_y_min = np.where(y == self.y_min)[0]
            indices_y_max = np.where(y == self.y_max)[0]
            indices_z_min = np.where(z == self.z_min)[0]
            indices_z_max = np.where(z == self.z_max)[0]
            all_indices = np.hstack((indices_x_min, indices_x_max, indices_y_min, indices_y_max, indices_z_min, indices_z_max))

            return indices_x_min, indices_x_max, indices_y_min, indices_y_max, indices_z_min, indices_z_max, all_indices

    def _on_bdy_dim2_neural_network(self, indices, du_dx, du_dy): 
        du_dx = du_dx.numpy()
        du_dy = du_dy.numpy()

        du_dn = np.hstack((-du_dx[indices[0]], du_dx[indices[1]], -du_dy[indices[2]], du_dy[indices[3]]))

        return du_dn 

class learning(basic_setting):
    def run(self, epochs=(5000, 10000), learning_rate=0.001, Adam_stop=0.1, loss_function=None, neural_network=None, print_loss=False, proportion_of_initial_and_boundary_sets=0.1, title="learned model"):
        r"""
        [Args]
            epochs : `tuple`
            learning_rate : `float`
            loss_function : `function`
            neural_network : `neural network`
            print_loss : `bool`
            proportion_of_test_set : `float`
        """
        assert len(epochs) == 2, "Input epochs for the optimizer Adam and L-BFGS-B as tuple"

        # Constructing train set and test set
        train_and_test_sets = creation.train_and_test_sets(self, proportion_of_initial_and_boundary_sets=proportion_of_initial_and_boundary_sets)
        train = train_and_test_sets[0]
        test = train_and_test_sets[1]

        # Constructing optimization with Adam
        total_loss_record_Adam = []
        optm = tf.keras.optimizers.Adam(learning_rate = learning_rate)

        for itr in tqdm(range(epochs[0])):
            with tf.GradientTape() as tape:
                train_loss = loss_function(train, test)
                total_loss_record_Adam.append(train_loss)
                        
                grad_w = tape.gradient(train_loss, neural_network.trainable_variables)
                optm.apply_gradients(zip(grad_w, neural_network.trainable_variables))
        
            if print_loss == True:
                if itr % 1000 == 0:
                    print(train_loss.numpy())
            
            if train_loss.numpy() < Adam_stop:
                break

        print("Optimization with Adam is done. The value of the loss function is {}.\n".format(total_loss_record_Adam[-1]))

        # Constructing optimization with L-BFGS-B
        def vec_weight():
            # vectorize weights
            weight_vec = []

            # Loop over all weights
            for v in neural_network.trainable_variables:
                weight_vec.extend(v.numpy().flatten())

            weight_vec = tf.convert_to_tensor(weight_vec)
            return weight_vec

        def restore_weight(weight_vec):
            # restore weight vector to model weights
            index = 0
            for v in neural_network.trainable_variables:
                v_shape = v.shape

                # weight matrices
                if len(v_shape) == 2:
                    size_w = v_shape[0] * v_shape[1]
                    updated_val = tf.reshape(weight_vec[index:index + size_w], (v_shape[0], v_shape[1]))
                    index += size_w

                # bias vectors
                elif len(v_shape) == 1:
                    updated_val = weight_vec[index:index + v_shape[0]]
                    index += v_shape[0]

                # assign variables (Casting necessary since scipy requires float64 type)
                v.assign(tf.cast(updated_val, dtype=tf.float32))
        
        total_loss_record_L_BFGS_B = []

        def loss_grad(w):
            # update weights in model
            restore_weight(w)

            for itr in tqdm(range(epochs[1])):
                with tf.GradientTape() as tape:
                    train_loss = loss_function(train, test)
                    total_loss_record_L_BFGS_B.append(train_loss)
                            
                    grad_w = tape.gradient(train_loss, neural_network.trainable_variables)
                    optm.apply_gradients(zip(grad_w, neural_network.trainable_variables))
            
            # vectorize gradients
            grad_vec = []
            for g in grad_w:
                grad_vec.extend(g.numpy().flatten())

            # gradient list to array
            # scipy-routines requires 64-bit floats
            loss = train_loss.numpy().astype(np.float64)
            grad_vec = np.array(grad_vec, dtype=np.float64)

            return loss, grad_vec
        
        scipy.optimize.minimize(fun=loss_grad, x0=vec_weight().numpy(), jac=True, method='L-BFGS-B',
                                options={'maxiter': 10000, 'maxfun': 50000, 'maxcor': 50, 'maxls': 50, 'ftol': 1.0 * np.finfo(float).eps})

        print("Optimization with L-BFGS-B is done. The value of the loss function is {}.\n".format(total_loss_record_L_BFGS_B[-1]))

        # Visualization
        epoch_Adam = len(total_loss_record_Adam)
        epoch_L_BFGS_B = len(total_loss_record_L_BFGS_B)

        self.final_epoch = epoch_Adam + epoch_L_BFGS_B

        fig = plt.figure(figsize = (5, 4))
        plt.plot(range(1, epoch_Adam + 1), total_loss_record_Adam, label="Adam")
        plt.plot(range(epoch_Adam, epoch_Adam + epoch_L_BFGS_B), total_loss_record_L_BFGS_B, label="L-BFGS-B")
        plt.xlabel('epoch', fontsize=11)             
        plt.ylabel('total loss $\mathcal{L}$', fontsize=11) 
        plt.legend()
        plt.show()
        fig.savefig("Figure/{}.png".format(title), dpi=1000)
        
        # Saving the loss functions
        total_loss_record = [loss.numpy() for loss in total_loss_record_Adam + total_loss_record_L_BFGS_B]
        
        if not os.path.exists("Data/loss function.csv"):
            df = pd.DataFrame(total_loss_record, columns=[str(self.widths)])
            df.to_csv("Data/loss function.csv", index=False)
        else:
            try:
                df = pd.read_csv("Data/loss function.csv")
                df[str(self.widths)] = total_loss_record
                df.to_csv("Data/loss function.csv", index=False)
                print("The train loss was saved.")
            except:
                print("WARNING: The train loss could not be saved since epochs are different from that of the preexisting csv file.")
        
        # Saving the learned model
        try:
            neural_network.save('Data/{}.tf'.format(title))
            print("\nThe learned model was saved. You can ignore the warring saying 'the compiled metrics have yet to be built'.")
        except:
            print("WARNING: The learned model could not be saved.")
    
    def prediction_for_fixed_time(self, fixed_time=0.0, the_number_of_points=50, neural_network=None, from_saved_model={'file_name':False, 'loss':False}):
        r"""
        [Args]
            fixed_time : `float`
            the_number_of_points : `int`
            neural_network : `neural network`
            from_saved_model : `False` or `dictionary`,
        """
        assert self.time_existence == 1, "time does not exist"

        if from_saved_model.get('file_name') != False and from_saved_model.get('loss') != False:
            neural_network = load_model('Data/{}.tf'.format(from_saved_model.get('file_name')), custom_objects={'loss': from_saved_model.get('loss')}, compile=False)

        prediction_set = creation.prediction_set_for_fixed_time(self, fixed_time=fixed_time, the_number_of_points=the_number_of_points)

        if self.the_number_of_equations == 1:
            predicted_u = neural_network.predict(prediction_set).ravel().reshape(the_number_of_points, the_number_of_points).T

            df_predicted_u = pd.DataFrame(data=predicted_u, columns=prediction_set.T[1][:the_number_of_points], index=prediction_set.T[1][:the_number_of_points])

            reverse_df_predicted_u = df_predicted_u.iloc[::-1]

            u_min, u_max = df_predicted_u.values.min().min(), df_predicted_u.values.max().max()

            print("prediction for t={} is completed.".format(np.round(fixed_time, 4)))

            return reverse_df_predicted_u, u_min, u_max

        elif self.the_number_of_equations == 2:
            predicted_uv = neural_network.predict(prediction_set).ravel()
            predicted_u = predicted_uv[0::2].reshape(the_number_of_points, the_number_of_points).T
            predicted_v = predicted_uv[1::2].reshape(the_number_of_points, the_number_of_points).T

            df_predicted_u = pd.DataFrame(data=predicted_u, columns=prediction_set.T[1][:the_number_of_points], index=prediction_set.T[1][:the_number_of_points])
            df_predicted_v = pd.DataFrame(data=predicted_v, columns=prediction_set.T[1][:the_number_of_points], index=prediction_set.T[1][:the_number_of_points])

            reverse_df_predicted_u = df_predicted_u.iloc[::-1]
            reverse_df_predicted_v = df_predicted_v.iloc[::-1]

            u_min, u_max = df_predicted_u.values.min().min(), df_predicted_u.values.max().max()
            v_min, v_max = df_predicted_v.values.min().min(), df_predicted_v.values.max().max()

            print("prediction for t={} is completed.".format(np.round(fixed_time, 4)))

            return reverse_df_predicted_u, reverse_df_predicted_v, u_min, u_max, v_min, v_max
    
    def prediction_for_time(self, the_number_of_frames=3, the_number_of_points=50, neural_network=None, from_saved_model={'file_name':False, 'loss':False}):
        r"""
        [Args]  
            the_number_of_frames : `int`
            the_number_of_points : `in`
            neural_network : `neural network`
            from_saved_model : `dict` 
            'file_name' : `False` or `string`
            'loss': `False` or `function`
        """
        assert the_number_of_frames > 1, "the number of frames > 1"
        assert self.time_existence == 1, "time does not exist"

        file_name = from_saved_model.get('file_name')
        loss = from_saved_model.get('loss')

        time = np.linspace(self.t_min, self.t_max, num=the_number_of_frames)

        if file_name != False and loss != False:
            prediction_for_t0 = learning.prediction_for_fixed_time(self, fixed_time=0.0, the_number_of_points=the_number_of_points, from_saved_model={'file_name': file_name, 'loss': loss})  
        else:
            prediction_for_t0 = learning.prediction_for_fixed_time(self, fixed_time=0.0, the_number_of_points=the_number_of_points, neural_network=neural_network)  
      
        if self.the_number_of_equations == 1:
            u = [prediction_for_t0[0]]
            u_min = prediction_for_t0[1] 
            u_max = prediction_for_t0[2]    

            for t in time[1:]:
                if file_name != False and loss != False:
                    prediction_for_t = learning.prediction_for_fixed_time(self, fixed_time=t, the_number_of_points=the_number_of_points, from_saved_model={'file_name': file_name, 'loss': loss})  
                else:
                    prediction_for_t = learning.prediction_for_fixed_time(self, fixed_time=t, the_number_of_points=the_number_of_points, neural_network=neural_network)   

                u.append(prediction_for_t[0])
                u_min = min(u_min, prediction_for_t[1])
                u_max = max(u_max, prediction_for_t[2])
            
            return time, u, u_min, u_max

        elif self.the_number_of_equations == 2:
            u = [prediction_for_t0[0]]
            v = [prediction_for_t0[1]]
            u_min = prediction_for_t0[2] 
            u_max = prediction_for_t0[3] 
            v_min = prediction_for_t0[4] 
            v_max = prediction_for_t0[5] 

            for t in time[1:]:
                if file_name != False and loss != False:
                    prediction_for_t = learning.prediction_for_fixed_time(self, fixed_time=t, the_number_of_points=the_number_of_points, from_saved_model={'file_name': file_name, 'loss': loss})  
                else:
                    prediction_for_t = learning.prediction_for_fixed_time(self, fixed_time=t, the_number_of_points=the_number_of_points, neural_network=neural_network) 

                u.append(prediction_for_t[0])
                v.append(prediction_for_t[1])
                u_min = min(u_min, prediction_for_t[2])
                u_max = max(u_max, prediction_for_t[3])
                v_min = min(v_min, prediction_for_t[4])
                v_max = max(v_max, prediction_for_t[5])
            
            return time, u, v, u_min, u_max, v_min, v_max