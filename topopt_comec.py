# A 377 LINE GUI CODE FOR TOPOLOGY OPTIMIZATION BY LUC PREVOST, 2021
from __future__ import division
import os
from matplotlib import colors
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import *
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import optimizer2D, optimizer3D

def main():
    root = Tk()
    gui = Window(root)
    gui.root.mainloop()
    return None

class Window:
    def __init__(self, root):
        self.root = root
        # Windows title
        self.root.title("Topopt Comec")
        # Windows size
        self.root.geometry('950x500')
        # Initialization
        # (self.nelxyz, self.volfrac) = ([40, 60, 0], 0.3)
        # (self.v, self.r, self.c) = ('-', 0, [0, 0, 0])
        # (self.fx, self.fy, self.fz) = ([20, 20, 0], [60, 0, 0], [0, 0, 0])
        # (self.a, self.fv) = (['↑', '↓', '-'], [0.1, 0.1, 0])
        # (self.sx, self.sy, self.sz) = ([0, 40, 0, 0], [60, 60, 0, 0], [0, 0, 0, 0])
        # (self.dim) = ('XYZ', 'XYZ', '-', '-')
        # (self.E, self.nu) = (1, 0.25)
        # (self.save_results, self.n_it) = (0, 2)
        
        (self.nelxyz, self.volfrac) = ([10, 10, 15], 0.3)
        (self.v, self.r, self.c) = ('-', 0, [0, 0, 0])
        (self.fx, self.fy, self.fz) = ([5, 5, 0], [5, 5, 0], [0, 5, 0])
        (self.a, self.fv) = (['Z:<', 'Z:>', '-'], [0.1, 0.1, 0])
        (self.sx, self.sy, self.sz) = ([0, 0, 10, 10], [0, 10, 0, 10], [0, 0, 0, 0])
        (self.dim) = ('XYZ', 'XYZ', 'XYZ', 'XYZ')
        (self.E, self.nu) = (1, 0.25)
        (self.save_results, self.n_it) = (0, 1)
        
        # Title
        Label(self.root, text = "Linear topology optimization for compliant mechanisms", font=("Arial", 20), fg = 'black').grid(row=0, column=0, columnspan = 20)
        Label(self.root, text = "Parameters", font=("Arial", 20), fg = 'black').grid(row=0, column=20, columnspan = 7)
        # Dimensions (1-3)
        Label(self.root, text = "Dimensions", font=("Arial", 12)).grid(row=1, column=20, columnspan = 7) 
        self.nelx_entry = Entry(self.root, width = 5)
        self.nelx_entry.grid(row=2, column = 20)
        self.nelx_entry.insert(0, self.nelxyz[0])
        Label(self.root, text = "x").grid(row=2, column=21)
        self.nely_entry = Entry(self.root, width = 5)
        self.nely_entry.grid(row=2, column = 22)
        self.nely_entry.insert(0, self.nelxyz[1])
        Label(self.root, text = "x").grid(row=2, column=23)
        self.nelz_entry = Entry(self.root, width = 5)
        self.nelz_entry.grid(row=2, column = 24)
        self.nelz_entry.insert(0, self.nelxyz[2])
        Label(self.root, text = "Volume fraction").grid(row=3, column=20, columnspan = 3)
        self.volfrac_entry = Entry(self.root, width = 5)
        self.volfrac_entry.grid(row=3, column = 23)
        self.volfrac_entry.insert(0, self.volfrac)
        # Void area (4-6)
        shapes = ['-', '□', '○']
        Label(self.root, text = "Void area", font=("Arial", 12)).grid(row=4, column=20, columnspan = 7)
        Label(self.root, text = "Shape").grid(row=5, column=20)
        self.v_entry = ttk.Combobox(self.root, values=shapes, width = 3, state = 'readonly')
        self.v_entry.current(0)
        self.v_entry.grid(row = 5, column = 21)
        Label(self.root, text = "Radius").grid(row=5, column=23)
        self.r_entry = Entry(self.root, width = 5)
        self.r_entry.grid(row=5, column = 24)
        self.r_entry.insert(0, self.r)
        Label(self.root, text = "Center").grid(row=6, column=20)
        self.cx_entry = Entry(self.root, width = 5)
        self.cx_entry.grid(row=6, column = 21)
        self.cx_entry.insert(0, self.c[0])
        self.cy_entry = Entry(self.root, width = 5)
        self.cy_entry.grid(row=6, column = 22)
        self.cy_entry.insert(0, self.c[1])
        self.cz_entry = Entry(self.root, width = 5)
        self.cz_entry.grid(row=6, column = 23)
        self.cz_entry.insert(0, self.c[2])
        # Forces (7-10)
        arrows = ['-', 'X:→', 'X:←', 'Y:↑', 'Y:↓', 'Z:<', 'Z:>']
        Label(self.root, text = "Forces", font=("Arial", 12)).grid(row=7, column=20, columnspan = 7)
        Label(self.root, text = "In", fg='#ff0000').grid(row=8, column=20)
        self.fix_entry = Entry(self.root, width = 5)
        self.fix_entry.grid(row=8, column = 21)
        self.fix_entry.insert(0, self.fx[0])
        self.fiy_entry = Entry(self.root, width = 5)
        self.fiy_entry.grid(row=8, column = 22)
        self.fiy_entry.insert(0, self.fy[0])
        self.fiz_entry = Entry(self.root, width = 5)
        self.fiz_entry.grid(row=8, column = 23)
        self.fiz_entry.insert(0, self.fz[0])
        self.a1_entry = tk.StringVar() 
        self.a1_entry = ttk.Combobox(self.root, values=arrows, width = 4, state = 'readonly')
        self.a1_entry.current(3)
        self.a1_entry.grid(row = 8, column = 24)
        self.v1_entry = Entry(self.root, width = 5)
        self.v1_entry.grid(row=8, column = 25)
        self.v1_entry.insert(0, self.fv[0])
        Label(self.root, text = "N/m").grid(row=8, column=26)
        Label(self.root, text = "Out1", fg='#0000ff').grid(row=9, column=20)
        self.fo1x_entry = Entry(self.root, width = 5)
        self.fo1x_entry.grid(row=9, column = 21)
        self.fo1x_entry.insert(0, self.fx[1])
        self.fo1y_entry = Entry(self.root, width = 5)
        self.fo1y_entry.grid(row=9, column = 22)
        self.fo1y_entry.insert(0, self.fy[1])
        self.fo1z_entry = Entry(self.root, width = 5)
        self.fo1z_entry.grid(row=9, column = 23)
        self.fo1z_entry.insert(0, self.fz[1])
        self.a2_entry = ttk.Combobox(self.root, values=arrows, width = 4, state = 'readonly')
        self.a2_entry.current(4)
        self.a2_entry.grid(row = 9, column = 24)
        self.v2_entry = Entry(self.root, width = 5)
        self.v2_entry.grid(row=9, column = 25)
        self.v2_entry.insert(0, self.fv[1])
        Label(self.root, text = "N/m").grid(row=9, column=26)
        Label(self.root, text = "Out2", fg='#0000ff').grid(row=10, column=20)
        self.fo2x_entry = Entry(self.root, width = 5)
        self.fo2x_entry.grid(row=10, column = 21)
        self.fo2x_entry.insert(0, self.fx[2])
        self.fo2y_entry = Entry(self.root, width = 5)
        self.fo2y_entry.grid(row=10, column = 22)
        self.fo2y_entry.insert(0, self.fy[2])
        self.fo2z_entry = Entry(self.root, width = 5)
        self.fo2z_entry.grid(row=10, column = 23)
        self.fo2z_entry.insert(0, self.fz[2])
        self.a3_entry = ttk.Combobox(self.root, values=arrows, width = 4, state = 'readonly')
        self.a3_entry.current(0)
        self.a3_entry.grid(row = 10, column = 24)
        self.v3_entry = Entry(self.root, width = 5)
        self.v3_entry.grid(row=10, column = 25)
        self.v3_entry.insert(0, self.fv[2])
        Label(self.root, text = "N/m").grid(row=10, column=26)
        # Supports (11-15)
        dim = ['-', 'X', 'Y', 'Z', 'XYZ']
        Label(self.root, text = "Supports location", font=("Arial", 12)).grid(row=11, column=20, columnspan = 7)
        Label(self.root, text = "▲1").grid(row=12, column=20)
        self.s1x_entry = Entry(self.root, width = 5)
        self.s1x_entry.grid(row=12, column = 21)
        self.s1x_entry.insert(0, self.sx[0])
        self.s1y_entry = Entry(self.root, width = 5)
        self.s1y_entry.grid(row=12, column = 22)
        self.s1y_entry.insert(0, self.sy[0])
        self.s1z_entry = Entry(self.root, width = 5)
        self.s1z_entry.grid(row=12, column = 23)
        self.s1z_entry.insert(0, self.sz[0])
        self.d1_entry = ttk.Combobox(self.root, values=dim, width = 4, state = 'readonly')
        self.d1_entry.current(4)
        self.d1_entry.grid(row = 12, column = 24)
        Label(self.root, text = "▲2").grid(row=13, column=20)
        self.s2x_entry = Entry(self.root, width = 5)
        self.s2x_entry.grid(row=13, column = 21)
        self.s2x_entry.insert(0, self.sx[1])
        self.s2y_entry = Entry(self.root, width = 5)
        self.s2y_entry.grid(row=13, column = 22)
        self.s2y_entry.insert(0, self.sy[1])
        self.s2z_entry = Entry(self.root, width = 5)
        self.s2z_entry.grid(row=13, column = 23)
        self.s2z_entry.insert(0, self.sz[1])
        self.d2_entry = ttk.Combobox(self.root, values=dim, width = 4, state = 'readonly')
        self.d2_entry.current(4)
        self.d2_entry.grid(row = 13, column = 24)
        Label(self.root, text = "▲3").grid(row=14, column=20)
        self.s3x_entry = Entry(self.root, width = 5)
        self.s3x_entry.grid(row=14, column = 21)
        self.s3x_entry.insert(0, self.sx[2])
        self.s3y_entry = Entry(self.root, width = 5)
        self.s3y_entry.grid(row=14, column = 22)
        self.s3y_entry.insert(0, self.sy[2])
        self.s3z_entry = Entry(self.root, width = 5)
        self.s3z_entry.grid(row=14, column = 23)
        self.s3z_entry.insert(0, self.sz[2])
        self.d3_entry = ttk.Combobox(self.root, values=dim, width = 4, state = 'readonly')
        self.d3_entry.current(0)
        self.d3_entry.grid(row = 14, column = 24)
        Label(self.root, text = "▲4").grid(row=15, column=20)
        self.s4x_entry = Entry(self.root, width = 5)
        self.s4x_entry.grid(row=15, column = 21)
        self.s4x_entry.insert(0, self.sx[3])
        self.s4y_entry = Entry(self.root, width = 5)
        self.s4y_entry.grid(row=15, column = 22)
        self.s4y_entry.insert(0, self.sy[3])
        self.s4z_entry = Entry(self.root, width = 5)
        self.s4z_entry.grid(row=15, column = 23)
        self.s4z_entry.insert(0, self.sz[3])
        self.d4_entry = ttk.Combobox(self.root, values=dim, width = 4, state = 'readonly')
        self.d4_entry.current(0)
        self.d4_entry.grid(row = 15, column = 24)
        # Material (16-18)
        Label(self.root, text = "Material", font=("Arial", 12)).grid(row=16, column=20, columnspan = 7)
        Label(self.root, text = "Young's modulus").grid(row=17, column=20, columnspan = 3)
        self.E_entry = Entry(self.root, width = 5)
        self.E_entry.grid(row=17, column = 23)
        self.E_entry.insert(0, self.E)
        Label(self.root, text = "N/m\N{superscript two}").grid(row=17, column=24)
        Label(self.root, text = "Poisson's ratio").grid(row=18, column=20, columnspan = 3)
        self.nu_entry = Entry(self.root, width = 5)
        self.nu_entry.grid(row=18, column = 23)
        self.nu_entry.insert(0, self.nu)
        # Save results (19)
        self.save_result_entry = IntVar()
        C = Checkbutton(self.root, text = "Save results", font=("Arial", 12), variable = self.save_result_entry, onvalue = 1, offvalue = 0, height=1, width = 10).grid(row=19, column=20, columnspan = 4)
        # Calculate (20)
        Label(self.root, text = "Num iteration", font=("Arial", 12)).grid(row=20, column=20, columnspan = 3)
        self.n_it_entry = Entry(self.root, width = 5)
        self.n_it_entry.grid(row=20, column = 23)
        self.n_it_entry.insert(0, self.n_it)
        B = Button(self.root, text="Calculate", bg='yellow', font=("Arial", 12), command = self.update_values)
        B.grid(row=19, column=24, columnspan = 3, rowspan = 2)
        self.root.bind("<Return>", self.update_values)
        # self.plot_values_2D(optimizer2D.optimize(self.nelxyz, self.volfrac, self.c, self.r, self.v,
        #     self.fx, self.fy, self.a, self.fv, self.sx, self.sy, self.dim,
        #     self.E, self.nu, self.n_it))
        self.plot_values_3D(optimizer3D.optimize(self.nelxyz, self.volfrac, self.c, self.r, self.v,
            self.fx, self.fy, self.fz, self.a, self.fv, self.sx, self.sy, self.sz, self.dim,
            self.E, self.nu, self.n_it))
        pass
    
    def update_values(self, event=None):
        self.nelxyz = [int(self.nelx_entry.get()), int(self.nely_entry.get()), int(self.nelz_entry.get())]
        (self.volfrac, self.v, self.r) = (float(self.volfrac_entry.get()), str(self.v_entry.get()), float(self.r_entry.get()))
        self.c = [int(self.cx_entry.get()), int(self.cy_entry.get()), int(self.cz_entry.get())]
        self.fx = [int(self.fix_entry.get()), int(self.fo1x_entry.get()), int(self.fo2x_entry.get())]
        self.fy = [int(self.fiy_entry.get()), int(self.fo1y_entry.get()), int(self.fo2y_entry.get())]
        self.fz = [int(self.fiz_entry.get()), int(self.fo1z_entry.get()), int(self.fo2z_entry.get())]
        self.a = [str(self.a1_entry.get()), str(self.a2_entry.get()), str(self.a3_entry.get())]
        self.sx = [int(self.s1x_entry.get()), int(self.s2x_entry.get()), int(self.s3x_entry.get()), int(self.s4x_entry.get())]
        self.sy = [int(self.s1y_entry.get()), int(self.s2y_entry.get()), int(self.s3y_entry.get()), int(self.s4y_entry.get())]
        self.sz = [int(self.s1z_entry.get()), int(self.s2z_entry.get()), int(self.s3z_entry.get()), int(self.s4z_entry.get())]
        self.dim = [str(self.d1_entry.get()), str(self.d2_entry.get()), str(self.d3_entry.get()), str(self.d4_entry.get())]
        (self.E, self.nu) = (float(self.E_entry.get()), float(self.nu_entry.get()))
        (self.save_results, self.n_it) = (bool(self.save_result_entry.get()), int(self.n_it_entry.get()))
        if self.check_entry():
            if self.nelxyz[2]==0:
                self.plot_values_2D(optimizer2D.optimize(self.nelxyz, self.volfrac, self.c, self.r, self.v,
                    self.fx, self.fy, self.a, self.fv, self.sx, self.sy, self.dim,
                    self.E, self.nu, self.n_it))
            else:
                self.plot_values_3D(optimizer3D.optimize(self.nelxyz, self.volfrac, self.c, self.r, self.v,
                    self.fx, self.fy, self.fz, self.a, self.fv, self.sx, self.sy, self.sz, self.dim,
                    self.E, self.nu, self.n_it))
        return None

    def check_entry(self, ):
        text_error = ""
        # Dimensions
        for nel in self.nelxyz:
            if nel<0: text_error = "The number of element in a dimension is lower than 0."
        for nel in self.nelxyz:
            if nel>200: text_error = "The number of element in a dimension is bigger than 200."
        if self.volfrac>1: text_error = "The volume fraction is bigger than 1."
        elif self.volfrac<0: text_error = "The volume fraction is lower than 0."
        # Void area
        elif self.r<0: text_error = "The radius is lower than 0."
        # Forces
        if self.a[0]=='-': text_error = "There is no input force."
        elif self.a[1]==self.a[2]=='-': text_error = "There is no output force."
        for f in self.fx:
            if f<0 or f>self.nelxyz[0]: text_error = "A force is outside the domain."
        for f in self.fy:
            if f<0 or f>self.nelxyz[1]: text_error = "A force is outside the domain."
        for f in self.fz:
            if f<0 or f>self.nelxyz[2]: text_error = "A force is outside the domain."
        if self.a[0] == '-': text_error = "The input force has no direction"
        if self.a[1]=='-' and self.a[2]=='-': text_error = "The outpout force has no direction."
        # Supports
        if self.dim[0]==self.dim[1]==self.dim[2]==self.dim[3]=='-': text_error = "There isn't any support."
        for s in self.sx:
            if s <0 or s>self.nelxyz[0]: text_error = "A support is outside the domain."
        for s in self.sy:
            if s <0 or s>self.nelxyz[1]: text_error = "A support is outside the domain."
        for s in self.sz:
            if s <0 or s>self.nelxyz[2]: text_error = "A support is outside the domain."
        # Material
        if self.E<0 or self.E>10: text_error = "The Young's modulus is not between 0 and 10."
        if self.nu<0 or self.nu>1: text_error = "The Poisson's ratio is not between 0 and 1."
        # Compute
        if self.n_it<0: text_error = "The number of iteration is lower than 0."
        # Return
        if text_error == "":
            return True
        else:
            messagebox.showerror("Input error", text_error)
            return False

    def plot_values_2D(self, xPhys):
        plt.ion() # Ensure that redrawing is possible
        fig, ax = plt.subplots()
        # Frame
        (fx, fy) = (int(self.nelxyz[0]/20), int(self.nelxyz[1]/20))
        plt.scatter([-fx, self.nelxyz[0]+fx], [-fy, self.nelxyz[1]+fy], color = "white")
        plt.xlabel("X")
        plt.ylabel("Y")
        # Mechanism
        im = ax.imshow(-xPhys.reshape((self.nelxyz[0], self.nelxyz[1])).T, cmap='gray',\
                interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        im.set_array(-xPhys.reshape((self.nelxyz[0], self.nelxyz[1])).T)
        # Forces
        col = 'r'
        for i in range(len(self.a)):
            if self.a[i] != '-':
                if i>0: col = 'b'
                if self.a[i] == 'X:→': (lx, ly) = (self.nelxyz[0]/6, 0)
                if self.a[i] == 'X:←': (lx, ly) = (-self.nelxyz[0]/6, 0)
                if self.a[i] == 'Y:↑': (lx, ly) = (0, self.nelxyz[1]/6)
                if self.a[i] == 'Y:↓': (lx, ly) = (0, -self.nelxyz[1]/6)
                plt.quiver(self.fx[i], self.fy[i], lx, ly, color=[col], scale=100)
        # Supports
        (sx, sy) = ([], [])
        for i in range(len(self.dim)):
            if self.dim[i] != '-':
                sx.append(self.sx[i])
                sy.append(self.sy[i])
        plt.scatter(sx, sy, s = 5*self.nelxyz[0], marker = "^", color = "black")
        # Tkinter
        chart = FigureCanvasTkAgg(fig, self.root)
        chart.get_tk_widget().grid(row = 1, column = 0, rowspan = 20, columnspan = 20, sticky=E+W+N+S)
        # Save data
        dirname = 'Results'
        if self.save_results:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                print("Directory " , dirname ,  " Created ")
            fig.savefig('Results/%s.png' % ('compliant_mechanism_2D'), bbox_inches='tight', dpi=200)
        return None

    def plot_values_3D(self, xPhys):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        (x, y, z) = ([], [], [])
        for k in range(self.nelxyz[2]):
            for j in range(self.nelxyz[1]):
                for i in range(self.nelxyz[0]):
                    if xPhys[j + i*self.nelxyz[1] + k*self.nelxyz[0]*self.nelxyz[1]] > 0.5:
                        x.append(i)
                        y.append(j)
                        z.append(k)
        ax.scatter(x, y, z, s=6000/min(self.nelxyz[0], self.nelxyz[1], self.nelxyz[2]), marker='s', c='black')
        ax.view_init(elev = self.nelxyz[2], azim = 60)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        # Plot forces
        col = 'r'
        for i in range(len(self.a)):
            if self.a[i] != '-':
                if i>0: col = 'b'
                if self.a[i] == 'X:→': (lx, ly, lz) = (self.nelxyz[0]/6, 0, 0)
                if self.a[i] == 'X:←': (lx, ly, lz) = (-self.nelxyz[0]/6, 0, 0)
                if self.a[i] == 'Y:↑': (lx, ly, lz) = (0, self.nelxyz[1]/6, 0)
                if self.a[i] == 'Y:↓': (lx, ly, lz) = (0, -self.nelxyz[1]/6, 0)
                if self.a[i] == 'Z:<': (lx, ly, lz) = (0, 0, self.nelxyz[1]/6)
                if self.a[i] == 'Z:>': (lx, ly, lz) = (0, 0, -self.nelxyz[1]/6)
                plt.quiver(self.fx[i], self.fy[i], self.fz[i],lx, ly, lz, linewidths=10, color=[col])
        # Supports
        (sx, sy, sz) = ([], [], [])
        for i in range(len(self.dim)):
            if self.dim[i] != '-':
                sx.append(self.sx[i])
                sy.append(self.sy[i])
                sz.append(self.sz[i])
        ax.scatter(sx, sy, sz, s = 5*min(self.nelxyz[0], self.nelxyz[1], self.nelxyz[2]), marker = "^", color = "black")
        # Tkinter
        chart = FigureCanvasTkAgg(fig, self.root)
        chart.get_tk_widget().grid(row = 1, column = 0, rowspan = 20, columnspan = 20, sticky=E+W+N+S)
        # Save data
        dirname = 'Results'
        if self.save_results:
            if not os.path.exists(dirname):
                os.makedirs(dirname)
                print("Directory " , dirname ,  " Created ")
            fig.savefig('Results/%s.png' % ('compliant_mechanism_3D'), bbox_inches='tight', dpi=200)
        return None

if __name__ == '__main__':
    main()